from fileinput import filename
import json
import os
from typing import List, Tuple

import matplotlib.pyplot as plt
import torch
import torchvision
from maraboupy import Marabou, MarabouCore, MarabouUtils
from utils import load_vnnlib
import datetime
PATH = 'vnncomp2021/benchmarks/mnistfc/mnist-net_256x6.onnx'
MAX_TIME = 600  # in seconds

M_OPTIONS: MarabouCore.Options = Marabou.createOptions(verbosity=0,
                                                       initialSplits=4,
                                                       timeoutInSeconds=MAX_TIME,
                                                       snc=True,
                                                       numWorkers=32,
                                                       )


RESULT_CSV = open("MNIST_ROBUST_256x6.csv", 'a')
# load benchmarks from vnncomp
# BENCHMARK_PATH = 'fig1.json'
# RAW = json.load(open(BENCHMARK_PATH, "r"))
# TARGET_INPUT = []
# for r in RAW["target"]:
#     TARGET_INPUT.extend(r)
# print(len(TARGET_INPUT))

BENCHMARK_PATH = 'datasets/MNIST/prop_7_0.05.vnnlib'
input, eps, true_label, adv_label = load_vnnlib(BENCHMARK_PATH)
print("true label", true_label)
print("eps", eps)
TARGET_INPUT = input.numpy()
print(TARGET_INPUT[:10])

BENCHMARK_FOLDER = 'vnncomp2021/benchmarks/mnistfc'

PATTERN_PATH = 'datasets/MNIST/MNIST_256x6_relu_patterns_0.json'
with open(PATTERN_PATH, "r") as f:
    STABLE_PATTERNS = json.load(f)


def add_relu_constraints(network: Marabou.MarabouNetwork,
                         relu_check_list: List[int],
                         relu_val: List[int]) -> None:
    """
    Add stable relus constraints to the Marabou network
    """
    for i in range(len(relu_check_list)):
        layer, idx, marabou_idx = parse_raw_idx(relu_check_list[i])
        # print(layer, idx, marabou_idx, marabou_idx+256)
        # print(relu_val[i])
        if relu_val[i] < 2000 : #should be correct all the way to delta =0.5
            constraint = MarabouUtils.Equation(MarabouCore.Equation.LE)
            constraint.addAddend(1, marabou_idx)
            constraint.setScalar(0)


            #output of ReLU must be 0
            network.setLowerBound(marabou_idx+256, 0)
            network.setUpperBound(marabou_idx+256, 0)

            #input of ReLU must be <=0
            # network.setUpperBound(marabou_idx, 0)

        else:
            #input of ReLU must be >=0
            constraint = MarabouUtils.Equation(MarabouCore.Equation.GE)
            constraint.addAddend(1, marabou_idx)
            constraint.setScalar(0)
            # network.setLowerBound(marabou_idx, 0)

            #output of relu must be the same as input
            # positive_phase_con = MarabouUtils.Equation(MarabouCore.Equation.EQ)
            # positive_phase_con.addAddend(1, marabou_idx)
            # positive_phase_con.addAddend(-1, marabou_idx+256)
            # positive_phase_con.setScalar(0)
            # print(positive_phase_con)
            # network.addEquation(positive_phase_con)
        network.addEquation(constraint)
    # # return network


def check_pattern(network: Marabou.MarabouNetwork, prop_name: str,
                  relu_check_list: List[int], relu_val: List[int],
                  label: int, other_label: int) -> Tuple[str, int]:
    print("--------CHECK PATTERN: output_{} is always less than output_{} ? --------".format(label, other_label))
    print("number of fixed relus:", len(relu_check_list))
    # network = init_network()
    add_relu_constraints(network, relu_check_list, relu_val)
    offset = network.outputVars[0][0][0]
    print(offset)
    # add output constraint

    for l in range(10):
        if l==other_label: continue
        constraint = MarabouUtils.Equation(MarabouCore.Equation.GE)
        constraint.addAddend(1, other_label+offset)
        constraint.addAddend(-1, l+offset)
        constraint.setScalar(0.00001)

        network.addEquation(constraint)
    network.saveQuery("testQuery")
    try:
        print("start solving")
        exit_code, vals, stats = network.solve( filename="{}_{}_vs_{}".format(prop_name, label, other_label),
                                                options=M_OPTIONS, 
                                                )
        running_time: int = stats.getTotalTimeInMicro()
        if exit_code=="sat":
            for idx, r in enumerate(relu_check_list):
                marabou_idx = parse_raw_idx(r)[-1]
                print(marabou_idx, vals[marabou_idx], relu_val[idx])

            print("double check output")
            for o in network.outputVars[0][0]:
                print(o, vals[o])
            print(label+offset, vals[label+offset])
            print(other_label+offset, vals[other_label+offset])

            print("Running time:{}".format(running_time))
        return exit_code, running_time
    except Exception:
        if exit_code not in ["sat", "unsat"]:
            print("THE QUERY CANNOT BE SOLVED")
        return exit_code, -1




def parse_raw_idx(raw_idx: int) -> Tuple[int, int, int]:
    """
    only for MNIST 256xk network:
    """
    n_relus = 256
    offset = 28*28+10
    layer = raw_idx // n_relus
    idx = raw_idx % n_relus
    marabou_idx = 2*n_relus*layer + idx + offset
    return layer, idx, marabou_idx


def run_benchmark(benchmark: str):
    global true_label
    RESULT_CSV.write(benchmark+"\n")
    RESULT_CSV.flush()
    table = ["UNK"]*10
    network_path = PATH
    x = TARGET_INPUT
    eps = 0.05
    true_label = true_label
    fig = plt.figure
    plt.imshow(x.reshape(28, 28), cmap='gray')
    plt.savefig("Prop.png")   
    for adv_label in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
        if adv_label == true_label: continue
        RESULT_CSV.write("vs {}".format(adv_label))
        RESULT_CSV.flush() 
        network = Marabou.read_onnx(network_path)
        print(network.numVars)
        # check if the benchmarks are really violated
        checking_eps = eps
        print("Checking with eps = {}".format(checking_eps))
        print("len x", len(x))
        for i in range(len(x)):
            network.setLowerBound(i, max(0, x[i]-checking_eps) )
            network.setUpperBound(i, min(1, x[i]+checking_eps))
            #check a ball centered at vector 0
            # network.setLowerBound(i, 0.5 - checking_eps)
            # network.setUpperBound(i, 0.5 + checking_eps)
        relu_check_list = STABLE_PATTERNS[str(true_label)]["stable_idx"]
        relu_val = STABLE_PATTERNS[str(true_label)]["val"]
        now = datetime.datetime.now()
        exit_code, running_time = check_pattern(network, prop_name="", relu_check_list = relu_check_list,
                                                relu_val = relu_val, label=true_label, other_label=adv_label)
        after_solve = datetime.datetime.now()
        my_time = after_solve - now
        if exit_code=="sat":
            table[adv_label] = "SAT:{} {}".format(running_time/10**6, my_time)
        elif exit_code=="unsat":
            table[adv_label] = "UNS:{} {}".format(running_time/10**6, my_time)
        else:
            table[adv_label] = exit_code
        RESULT_CSV.write(table[adv_label]+"\n")
        RESULT_CSV.write(str(datetime.datetime.now()))
        RESULT_CSV.write("\n")
        RESULT_CSV.flush()
        print(exit_code, running_time, my_time)
    print(table)


def main():
    violated = run_benchmark("prop_7_0.05.vnnlib")

main()