"""
This script will try to check if a ReLU pattern is good enough to prove the outcome of an ACAS network

"""

import re
from typing import Set, Tuple, List
import numpy as np
from maraboupy import Marabou, MarabouCore, MarabouUtils


PATH = '/home/nle/workspace/VerifyNNE/datasets/ACAS/acas_nets/ACASXU_run2a_1_3_batch_2000.nnet'
EPSILON = 0.005


relu_check_list = [1, 2, 20, 27, 29, 30, 43, 59, 61, 63, 65, 69, 73, 87, 96, 99, 104, 106, 132, 141, 145, 148, 152, 153, 164, 165, 166, 174, 175, 176, 178, 179, 182, 185, 186, 188, 192, 201, 205, 207, 208, 211, 220, 221, 223, 226, 232, 243]
relu_val = [2000, 0, 0, 0, 2000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2000, 0, 0, 2000, 0, 0, 0, 2000, 0, 0, 0, 0, 2000, 2000, 0, 0, 0, 2000, 0, 0, 0, 0, 2000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2000, 2000]

# relu_check_list = [1, 2, 19, 20, 27, 29, 30, 43, 59, 61, 63, 65, 69, 73, 87, 96, 99, 104, 106, 114, 132, 141, 145, 148, 151, 152, 153, 164, 165, 166, 172, 174, 175, 176, 178, 179, 180, 182, 184, 185, 186, 188, 192, 201, 202, 205, 207, 208, 210, 211, 213, 218, 220, 221, 223, 226, 232, 241, 243]
# relu_val = [2000, 0, 8, 0, 0, 2000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2000, 0, 0, 2000, 1, 0, 0, 0, 2000, 5, 0, 0, 0, 0, 2000, 8, 2000, 0, 0, 0, 2000, 1990, 0, 4, 0, 0, 0, 2000, 0, 7, 0, 0, 0, 1, 0, 1, 5, 0, 0, 0, 0, 2000, 1997, 2000]

#relu_check_list = [2, 27, 29, 30, 63, 65, 69, 73, 87, 99, 104, 132, 148, 165, 179, 188, 207, 208, 232]
#relu_val = [0, 0, 2000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2000, 0, 2000, 0, 0, 0, 2000]

assert(len(relu_check_list)==len(relu_val))

def init_network():
    network = Marabou.read_nnet(PATH)

    print("output nodes:", network.outputVars)
    print("input nodes:", network.inputVars)
    print("relu list:")

    for r in network.reluList:
        print(r)
    network.setLowerBound(0, -1)
    network.setUpperBound(0, 1)
    network.setLowerBound(1, -1)
    network.setUpperBound(1, 1)
    network.setLowerBound(2, -1)
    network.setUpperBound(2, 1)
    network.setLowerBound(3, -1)
    network.setUpperBound(3, 1)
    network.setLowerBound(4, -1)
    network.setUpperBound(4, 1)

    return network

def add_relu_constraints(network: Marabou.MarabouNetworkNNet, 
                        relu_check_list: List[int], 
                        relu_val: List[int])->Marabou.MarabouNetworkNNet:
    for i in range(len(relu_check_list)):
        layer, idx, marabou_idx = parse_raw_idx(relu_check_list[i])
        print(layer, idx, marabou_idx)
        if relu_val[i] == 0:
            constraint = MarabouUtils.Equation(MarabouCore.Equation.LE)
            constraint.addAddend(1, marabou_idx)
            constraint.setScalar(-0.001)
        else:
            constraint = MarabouUtils.Equation(MarabouCore.Equation.GE)
            constraint.addAddend(1, marabou_idx)
            constraint.setScalar(0.001)
        network.addEquation(constraint)

    return network

def parse_raw_idx(raw_idx: int)->Tuple[int, int, int]:
    """
    only for ACAS network:
    """
    n_relus = 50
    offset = 5
    layer = raw_idx // n_relus
    idx = raw_idx % n_relus
    marabou_idx =  2*n_relus*layer + idx + offset
    return layer, idx, marabou_idx

def find_one_assignment(relu_check_list: List[int], relu_val: List[int])->None:
    network = init_network()
    network = add_relu_constraints(network, relu_check_list, relu_val)    
    exitCode, vals, stats = network.solve()
    assert(exitCode=="sat")    
    for idx, r in enumerate(relu_check_list):
        marabou_idx = parse_raw_idx(r)[-1]
        print(marabou_idx, vals[marabou_idx], relu_val[idx])

def check_pattern(relu_check_list: List[int], relu_val: List[int], label: int, other_label: int)->None:
    print("--------CHECK PATTERN--------")
    network = init_network()
    network = add_relu_constraints(network, relu_check_list, relu_val)    
    offset = network.outputVars[0][0][0]
    print("offset", offset)
    #add output constraint
    # logit of 0 - logit of 1 > 0
    constraint = MarabouUtils.Equation(MarabouCore.Equation.GE)
    constraint.addAddend(1, label+offset)
    constraint.addAddend(-1, other_label+offset)
    constraint.setScalar(0.0002)
    network.addEquation(constraint)

    exitCode, vals, stats = network.solve()

    for idx, r in enumerate(relu_check_list):
        marabou_idx = parse_raw_idx(r)[-1]
        print(marabou_idx, vals[marabou_idx], relu_val[idx])

    print("double check output")
    for o in network.outputVars[0][0]:
        print(o, vals[o])
    print(label+offset, vals[label+offset])
    print(other_label+offset, vals[other_label+offset])
# find_one_assignment(relu_check_list, relu_val)
check_pattern(relu_check_list, relu_val, label=0, other_label = 1)