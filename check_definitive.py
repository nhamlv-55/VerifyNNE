import json
import os
from typing import List, Tuple, Set

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
                                                       numWorkers=6
                                                       )

PATTERN_PATH = 'datasets/MNIST/mnist_relu_patterns_0.json'

RESULT_CSV = open("DEF_NAP.csv", "w")
with open(PATTERN_PATH, "r") as f:
    STABLE_PATTERNS = json.load(f)

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
        if relu_val[i] < 2500:
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

def load_NAP(label:str)->Tuple[Set[int], Set[int]]:
    A:Set[int] = set()
    D:Set[int] = set()
    relu_check_list = STABLE_PATTERNS[label]["stable_idx"]
    relu_val = STABLE_PATTERNS[label]["val"]

    assert len(relu_check_list) == len(relu_val)
    for r_idx, r in enumerate(relu_check_list):
        if relu_val[r_idx]==0:
            D.add(r)
        else:
            A.add(r)
    return A, D
network = None
for label1 in range(10):
    for label2 in range(label1+1, 10):
        NAP1 = load_NAP(str(label1))
        NAP2 = load_NAP(str(label2))
        print(len(NAP1[0]), len(NAP2[1]), len(NAP1[0].intersection(NAP2[1])))
        print(len(NAP1[1]), len(NAP2[0]), len(NAP1[1].intersection(NAP2[0])))
        if len(NAP1[0].intersection(NAP2[1]))>0 or len(NAP1[1].intersection(NAP2[0]))>0:
            RESULT_CSV.write("{} and {} are definitive!\n".format(label1, label2))
            RESULT_CSV.flush()
            print(label1, label2, "are definitive")
            continue
        else:
            #check using marabou
            label1 = str(label1)
            label2 = str(label2)
            pattern1 = STABLE_PATTERNS[label1]["stable_idx"]
            value1 = STABLE_PATTERNS[label1]["val"]

            pattern2 = STABLE_PATTERNS[label2]["stable_idx"]
            value2 = STABLE_PATTERNS[label2]["val"]

            #init network and set bounds
            if network is not None:
                del network
            network = Marabou.read_onnx(PATH)
            for i in range(28*28):
                network.setLowerBound(i, 0)
                network.setUpperBound(i, 1)
            
            #add relu

            network = add_relu_constraints(network, pattern1, value1)
            network = add_relu_constraints(network, pattern2, value2)

            exit_code, vals, stats = network.solve(options=M_OPTIONS)

            running_time: int = stats.getTotalTimeInMicro()
            if exit_code=="sat":
                RESULT_CSV.write("{} and {} are not definitive!\n".format(label1, label2))
                RESULT_CSV.flush()
                # for idx, r in enumerate(relu_check_list):
                #     marabou_idx = parse_raw_idx(r)[-1]
                #     print(marabou_idx, vals[marabou_idx], relu_val[idx])

                # print("double check output")
                # for o in network.outputVars[0][0]:
                #     print(o, vals[o])
                # print(label+offset, vals[label+offset])
                # print(other_label+offset, vals[other_label+offset])

                # print("Running time:{}".format(running_time))
            elif exit_code=="unsat":
                RESULT_CSV.write("{} and {} are definitive!\n".format(label1, label2))
                RESULT_CSV.flush()