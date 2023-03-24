import json
import os
from typing import List, Tuple, Union, Dict
import numpy as np
import matplotlib.pyplot as plt
from maraboupy import Marabou, MarabouCore, MarabouUtils
from utils import load_vnnlib, Pattern, get_pattern, parse_raw_idx
import datetime

PATH = 'vnncomp2021/benchmarks/mnistfc/mnist-net_256x4.onnx'
MAX_TIME = 1200  # in seconds
np.random.seed(42)
M_OPTIONS: MarabouCore.Options = Marabou.createOptions(verbosity=0,
                                                       initialSplits=1,
                                                       timeoutInSeconds=MAX_TIME,
                                                       snc=True,
                                                       numWorkers=4,
                                                       )


BENCHMARK_PATH = 'datasets/MNIST/prop_7_0.05.vnnlib'
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

START_PATTERN = Pattern()
# START_PATTERN.from_check_list(STABLE_PATTERNS[str(true_label)]["stable_idx"],
#                               STABLE_PATTERNS[str(true_label)]["val"])


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
        print("start solving")
        exit_code: str
        vals: Dict[int, float]
        exit_code, vals, stats = network.solve( filename="{}_{}_vs_{}".format(prop_name, label, other_label),
                                                options=M_OPTIONS, 
                                                )
        running_time: int = stats.getTotalTimeInMicro()
        print(exit_code)
        if exit_code=="sat":
            print("double check output")
            print("Running time:{}".format(running_time))

            cex: List[float] = [0]*784
            for idx in range(len(cex)):
                cex[idx] = vals[idx]

            print(vals)

            return exit_code, running_time, cex
    except Exception as e:
        print(e)
        if exit_code not in ["sat", "unsat"]:
            print("THE QUERY CANNOT BE SOLVED")
        return exit_code, -1, None

    return exit_code, running_time, None




def findCEX(x: List[float], NAP: Pattern, target_epsilon: float)->Union[List[float], None]:
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
            return cex
            break
        elif exit_code=="unsat":
            print("no cex!")
        else:
            print("error!")
        print(exit_code, running_time, my_time)

    return None


def find_NAP(input: List[float], start_pattern: Pattern, target_epsilon: float) -> Pattern:
    NAP_bar = get_pattern(NETWORK, input)
    
    print("NAP_bar:{}".format(NAP_bar))
    print("len NAP_bar:{}".format(NAP_bar.n_fixed_relus))
    NAP_star = START_PATTERN
    while True:
        cex = findCEX(input, NAP_star, target_epsilon)
        print("cex", cex)
        if cex is None:
            return NAP_star
        else:
            cex_pattern = get_pattern(NETWORK, cex)
            NAP_star.strengthen(NAP_bar, cex_pattern)
        





def main():
    NAP_star = find_NAP(TARGET_INPUT, START_PATTERN, 0.03)
    print(NAP_star)
main()