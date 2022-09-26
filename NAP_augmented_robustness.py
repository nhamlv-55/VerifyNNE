import json
import os
from typing import List, Tuple

import matplotlib.pyplot as plt
import torch
import torchvision
from maraboupy import Marabou, MarabouCore, MarabouUtils
from torch import Tensor
import pandas
from ReluPatterns import Patterns
from utils import load_vnnlib
import datetime
PATH = 'vnncomp2021/benchmarks/mnistfc/mnist-net_256x4.onnx'
MAX_TIME = 300  # in seconds
M_OPTIONS: MarabouCore.Options = Marabou.createOptions(verbosity=0,
                                                       timeoutInSeconds=MAX_TIME,
                                                       snc=True,
                                                       numWorkers=8
                                                       )


RESULT_CSV = open("NAP_AUG_ROBUST.csv", "w")
# load benchmarks from vnncomp
BENCHMARK_PATH = 'datasets/Marabou_MNIST_vnncomp21.csv'
BENCHMARK_FOLDER = 'vnncomp2021/benchmarks/mnistfc'

PATTERN_PATH = 'datasets/MNIST/mnist_relu_patterns_0.json'
with open(PATTERN_PATH, "r") as f:
    STABLE_PATTERNS = json.load(f)


def add_relu_constraints(network: Marabou.MarabouNetwork,
                         relu_check_list: List[int],
                         relu_val: List[int]) -> Marabou.MarabouNetwork:
    """
    Add stable relus constraints to the Marabou network
    """
    for i in range(len(relu_check_list)):
        layer, idx, marabou_idx = parse_raw_idx(relu_check_list[i])
        # print(layer, idx, marabou_idx, marabou_idx+256)
        # print(relu_val[i])
        if relu_val[i] == 0:
            constraint = MarabouUtils.Equation(MarabouCore.Equation.LE)
            constraint.addAddend(1, marabou_idx)
            constraint.setScalar(0)

            network.setLowerBound(marabou_idx+256, 0)
            network.setUpperBound(marabou_idx+256, 0)
        else:
            constraint = MarabouUtils.Equation(MarabouCore.Equation.GE)
            constraint.addAddend(1, marabou_idx)
            constraint.setScalar(0)
        network.addEquation(constraint)

    return network


def check_pattern(network: Marabou.MarabouNetwork,
                  relu_check_list: List[int], relu_val: List[int],
                  label: int, other_label: int) -> Tuple[str, int]:
    print("--------CHECK PATTERN: output_{} is always less than output_{} ? --------".format(label, other_label))
    # network = init_network()
    network = add_relu_constraints(network, relu_check_list, relu_val)
    offset = network.outputVars[0][0][0]
    print(offset)
    # add output constraint
    constraint = MarabouUtils.Equation(MarabouCore.Equation.GE)

    constraint.addAddend(1, other_label+offset)
    constraint.addAddend(-1, label+offset)
    constraint.setScalar(0)
    network.addEquation(constraint)
    network.saveQuery("log_query2")
    # return 
    try:
        print("start solving")
        exit_code, vals, stats = network.solve(options=M_OPTIONS)
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
    only for MNIST 256x4 network:
    """
    n_relus = 256
    offset = 28*28+10
    layer = raw_idx // n_relus
    idx = raw_idx % n_relus
    marabou_idx = 2*n_relus*layer + idx + offset
    return layer, idx, marabou_idx


def run_benchmark(benchmark: str):
    _, network_path, prop_path, _, res, _ = benchmark.strip().split(',')
    network_path = os.path.join(
        BENCHMARK_FOLDER, os.path.basename(network_path))
    if res == "violated":
        RESULT_CSV.write(benchmark+"\n")
        RESULT_CSV.flush()
        table = ["UNK"]*10
        prop_path = os.path.basename(prop_path)
        x, eps, true_label, adv_labels = load_vnnlib(
            os.path.join(BENCHMARK_FOLDER, prop_path))
        print(prop_path, true_label, adv_labels)
        RESULT_CSV.write(prop_path+"\n")
        RESULT_CSV.write(str(true_label)+"\n")
        RESULT_CSV.write(str(adv_labels)+"\n")
        RESULT_CSV.flush()
        for adv_label in adv_labels:
            network = Marabou.read_onnx(network_path)
            # check if the benchmarks are really violated
            checking_eps = eps
            print("Checking with eps = {}".format(checking_eps))
            for i in range(len(x)):
                network.setLowerBound(i, max((x[i]-checking_eps), 0))
                network.setUpperBound(i, min((x[i]+checking_eps), 1))

            relu_check_list = STABLE_PATTERNS[str(true_label)]["stable_idx"]
            relu_val = STABLE_PATTERNS[str(true_label)]["val"]
            exit_code, running_time = check_pattern(network, relu_check_list = relu_check_list,
                                                    relu_val = relu_val, label=true_label, other_label=adv_label)
            if exit_code=="sat":
                table[adv_label] = "SAT:{}".format(running_time/10**6)
            elif exit_code=="unsat":
                table[adv_label] = "UNS:{}".format(running_time/10**6)
            else:
                table[adv_label] = exit_code
            RESULT_CSV.write(table[adv_label]+"\n")
            RESULT_CSV.flush()
            print(exit_code, running_time)
        print(table)
        # fig = plt.figure
        # plt.imshow(x.resize(28, 28), cmap='gray')
        # plt.savefig(prop_path+".png")

def main():
    with open(BENCHMARK_PATH, "r") as f:
        """
        a line is
        <dataset>, <network>, <property>, <overhead>, <result>, <time>
        mnistfc,./benchmarks/mnistfc/mnist-net_256x4.onnx,./benchmarks/mnistfc/prop_14_0.05.vnnlib,3.210934332,violated,18.468217869
        """
        benchmarks = f.readlines()
    for b in benchmarks:
        violated = run_benchmark(b)

main()