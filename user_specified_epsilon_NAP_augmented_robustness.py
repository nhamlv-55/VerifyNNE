import json
import os
from typing import List, Tuple, Union, Dict, Set
import numpy as np
import matplotlib.pyplot as plt
from maraboupy import Marabou, MarabouCore, MarabouUtils
from vnncomp2021.benchmarks.cifar10_resnet.pytorch_model.attack_pgd import attack_pgd
from utils import load_vnnlib, Pattern, get_pattern, parse_raw_idx
import datetime
from onnx2pytorch import ConvertModel
import onnx
import torch
import tempfile
import subprocess
import datetime
PATH = 'vnncomp2021/benchmarks/mnistfc/mnist-net_256x4.onnx'
MAX_TIME = 1200  # in seconds
np.random.seed(42)
MARABOU = '/home/nle/opt/Marabou/build/bin/Marabou'

BENCHMARK_PATH = 'datasets/MNIST/prop_14_0.05.vnnlib'
input, eps, true_label, adv_label = load_vnnlib(BENCHMARK_PATH)
print("true label", true_label)
print("eps", eps)
TARGET_INPUT = input.numpy()
print(TARGET_INPUT[:10])

BENCHMARK_FOLDER = 'vnncomp2021/benchmarks/mnistfc'

PATTERN_PATH = 'datasets/MNIST/mnist_relu_patterns_0.json'
with open(PATTERN_PATH, "r") as f:
    STABLE_PATTERNS = json.load(f)

NETWORK = Marabou.read_onnx(PATH)
PYTORCH_NETWORK = ConvertModel(onnx.load(PATH))
print("Checking loaded network:")
print("true label:", true_label)
true_output = PYTORCH_NETWORK(input.resize(1, 28, 28))
print("true_output", true_output)
START_PATTERN = Pattern()
# START_PATTERN.from_check_list(STABLE_PATTERNS[str(true_label)]["stable_idx"],
#                               STABLE_PATTERNS[str(true_label)]["val"])

def pgd_attack(input, true_label, target, eps):
    pertubation = attack_pgd(PYTORCH_NETWORK, input.resize(1, 1, 28, 28), 
                            torch.tensor([true_label], dtype= torch.int64) , 
                            multi_targeted=False, 
                            target = torch.tensor([target], dtype = torch.int64),
                            epsilon=eps, alpha=0.075, attack_iters=100, num_restarts=5, upper_limit=1, lower_limit=0, use_adam=True)

    attack_image = input.resize(1, 1, 28, 28)+pertubation
    attack_output = PYTORCH_NETWORK(attack_image)
    # print(attack_output, torch.argmax(attack_output))
    if torch.argmax(attack_output).item() == target:
        return attack_image
    
    return None
    
def add_relu_constraints(network: Marabou.MarabouNetwork,
                         pattern: Pattern) -> None:
    """
    Add stable relus constraints to the Marabou network
    """
    print("adding relu constraints... There are {} fixed relus".format(pattern.n_fixed_relus))
    print("adding positive relu")
    for i in range(len(pattern.activated)):
        marabou_idx = pattern.activated[i]+28*28+10
        print(marabou_idx)
        #input of ReLU must be >=0
        network.setLowerBound(marabou_idx, 0.000001)

    print("adding negative relu")
    for i in range(len(pattern.deactivated)):
        marabou_idx = pattern.deactivated[i]+28*28+10
        print(marabou_idx)
        constraint = MarabouUtils.Equation(MarabouCore.Equation.LE)
        constraint.addAddend(1, marabou_idx)
        constraint.setScalar(-0.000001)

        network.addEquation(constraint)

        #output of ReLU must be 0
        network.setLowerBound(marabou_idx+256, -0.000001)
        network.setUpperBound(marabou_idx+256, 0.000001)



def check_pattern(network: Marabou.MarabouNetwork, prop_name: str,
                  pattern: Pattern,
                  label: int, other_label: int, add_output_constraints: bool) -> Tuple[str, int, Union[List[float], None]]:
    print("--------CHECK PATTERN: output_{} is always less than output_{} ? --------".format(label, other_label))
    # network = init_network()
    add_relu_constraints(network, pattern)
    offset = network.outputVars[0][0][0]
    print(offset)
    # add output constraint

    if add_output_constraints:
        print("adding output constraints...")
        for l in range(10):
            if l==other_label: continue
            constraint = MarabouUtils.Equation(MarabouCore.Equation.GE)
            constraint.addAddend(1, other_label+offset)
            constraint.addAddend(-1, l+offset)
            constraint.setScalar(0.00001)

            network.addEquation(constraint)
    else:
        print("skip adding output constraints")


    try:
        print("save query to a file")
        tf = tempfile.NamedTemporaryFile(delete=False)
        network.saveQuery(tf.name)

        print("solve saved query using Marabou binary")

        cmd = [MARABOU, 
               "--input-query={}".format(tf.name),
               "--blas-threads={}".format(4),
               "--num-workers={}".format(8), 
               "--timeout={}".format(MAX_TIME),
               "--snc",
               "--export-assignment", 
            #    "--prove-unsat"
               ]
        start_time = datetime.datetime.now()
        p = subprocess.run(cmd, stdout=subprocess.PIPE)
        finish_time = datetime.datetime.now()
        running_time = (finish_time - start_time).seconds
        print(p.returncode)
        stdout: List[str] = p.stdout.decode('utf-8').split("\n")
        print("stdout snippet:")
        print(stdout[-5:])
        exit_code:str = stdout[-2]

        # exit_code: str
        # vals: Dict[int, float]
        # exit_code, vals, stats = network.solve( filename="{}_{}_vs_{}".format(prop_name, label, other_label),
        #                                         options=M_OPTIONS, 
        #                                         )
        # running_time: int = stats.getTotalTimeInMicro()
        print(exit_code)
        if exit_code=="sat":
            print("Running time:{}".format(running_time))

            cex: List[float] = [0]*784
        #     for idx in range(len(cex)):
        #         cex[idx] = vals[idx]

        #     print(vals)

            return exit_code, running_time, cex
    except Exception as e:
        print(e)
        if exit_code not in ["SAT", "UNSAT"]:
            print("THE QUERY CANNOT BE SOLVED")
        return exit_code, -1, None

    return exit_code, running_time, None




def findCEX(x: List[float], NAP: Pattern, target_epsilon: float, use_attack: bool = True)->Union[List[List[float]]]:
    """
    find some CEXs that follows a NAP but is classified differently
    if use_attack is true, then first we will use an adv. method to find adv. example and check if they follow the NAP
    """
    found_cex:List[List[float]] = []
    if use_attack:
        print("Running adv. attack to find cexs...")
        n_cex = 100
        n = Marabou.read_onnx(PATH)
        for i in range(n_cex):
            cex = pgd_attack(torch.Tensor(x), true_label=true_label, target=0, eps=target_epsilon)
            if cex is not None:
                cex = cex.flatten().numpy().tolist()
                cex_pattern = get_pattern(n, cex)
                if cex_pattern <= NAP:
                    print("*", sep="")
                    found_cex.append(cex)
                else:
                    print("-", sep="")
        print("found {} cex using pgd".format(len(found_cex)))

    if len(found_cex)>0:
        return found_cex
    else: #pgd doesnt find anything. maybe we are robust?
        print("Running Marabou to find cexs...")
        for adv_label in range(1):
            if adv_label == true_label: continue
            network = Marabou.read_onnx(PATH) 
            print(network.numVars)
            # check if the benchmarks are really violated
            for i in range(len(x)):
                network.setLowerBound(i, max(0, x[i] - target_epsilon))
                network.setUpperBound(i, min(1, x[i] + target_epsilon))
            now = datetime.datetime.now()
            try:
                exit_code, running_time, cex = check_pattern(network, prop_name="", pattern = START_PATTERN,
                                                    label=true_label, other_label=adv_label, 
                                                    add_output_constraints=True)
            except Exception as e:
                print(e)
                exit_code = "error"
                cex = None
            after_solve = datetime.datetime.now()
            my_time = after_solve - now
            print(exit_code)
            if exit_code=="sat":
                print("found cex!")
                return [cex]
                break
            elif exit_code=="unsat":
                print("no cex!")
            else:
                print("error!")
            print(exit_code, running_time, my_time)

        return []


def find_core_NAP(NAPs: List[Pattern])->Pattern:
    """
    given a list of NAP, build the one using the intersection of all
    """
    result = Pattern()
    all_A: List[Set[int]] = [set(p.activated) for p in NAPs]
    print("all_A", all_A)
    all_D: List[Set[int]] = [set(p.deactivated) for p in NAPs]
    new_A: Set[int] = set.intersection(*all_A)
    new_D: Set[int] = set.intersection(*all_D)

    result.activated = new_A
    result.deactivated = new_D

    return result

def find_NAP(input: List[float], start_pattern: Pattern, target_epsilon: float) -> Pattern:
    NAP_bar = get_pattern(NETWORK, input)
    
    print("NAP_bar:{}".format(NAP_bar))
    print("len NAP_bar:{}".format(NAP_bar.n_fixed_relus))
    NAP_star = START_PATTERN
    while True:
        cex = findCEX(input, NAP_star, target_epsilon)
        if cex is None:
            return NAP_star
        else:
            cex_patterns: List[Pattern] = []
            for c in cex:
                cex_patterns.append(get_pattern(NETWORK, c))

            core_pattern = find_core_NAP(cex_patterns)
            print("core pattern", core_pattern)
            NAP_star.strengthen(NAP_bar, core_pattern)
        





def main():
    EPS = 0.05
    #check that the target input is not robust at the desired epsilon
    print("checking to see if the input is not robust at the desired epsilon")
    cs = findCEX(TARGET_INPUT, START_PATTERN, EPS, use_attack=False)
    if len(cs)==0:
        print("input is already robust at epsilon")
        return
    NAP_star = find_NAP(TARGET_INPUT, START_PATTERN, EPS)
    print(NAP_star)
main()