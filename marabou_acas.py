"""
This script will try to check if a ReLU pattern is good enough to prove the outcome of an ACAS network

"""

import json
from typing import Tuple, List
import numpy as np
from maraboupy import Marabou, MarabouCore, MarabouUtils
import logging
import pandas

PATH = 'datasets/ACAS/acas_nets/ACASXU_run2a_1_3_batch_2000.nnet'
EPSILON = 0.005
PATTERN_PATH = 'datasets/ACAS/relu_patterns.json'
MAX_TIME = 300 #in seconds
M_OPTIONS: MarabouCore.Options = Marabou.createOptions(verbosity=0,
                                  timeoutInSeconds=MAX_TIME
                                  )


with open(PATTERN_PATH, "r") as f:
    STABLE_PATTERNS = json.load(f)


def init_network()->Marabou.MarabouNetworkNNet:
    network:Marabou.MarabouNetworkNNet = Marabou.read_nnet(PATH)

    print("output nodes:", network.outputVars)
    print("input nodes:", network.inputVars)

    logging.debug("relu list:")
    for r in network.reluList:
        logging.debug(r)
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
    """
    Add stable relus constraints to the Marabou network
    """
    for i in range(len(relu_check_list)):
        layer, idx, marabou_idx = parse_raw_idx(relu_check_list[i])
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

def check_pattern(relu_check_list: List[int], relu_val: List[int], label: int, other_label: int)->Tuple[str, int]:
    """
    In ACAS, the prediction is the label with smallest value.
    So we check that label - other_label < 0 forall input
    by finding assignments for label - other_label >=0
    """
    print("--------CHECK PATTERN: output_{} is always less than output_{} ? --------".format(label, other_label))
    network = init_network()
    network = add_relu_constraints(network, relu_check_list, relu_val)    
    offset = network.outputVars[0][0][0]

    #add output constraint
    constraint = MarabouUtils.Equation(MarabouCore.Equation.GE)
    constraint.addAddend(1, label+offset)
    constraint.addAddend(-1, other_label+offset)
    constraint.setScalar(-0.0001)
    network.addEquation(constraint)

    try:
        exit_code: str    
        exit_code, vals, stats = network.solve(options=M_OPTIONS)
        running_time:int = stats.getTotalTimeInMicro()
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
def main():
    res = [[-1]*5 for i in range(5)]
    print(res)
    for label in STABLE_PATTERNS:
        print("For label {}, check if its stable RELU pattern guarantees the output")
        for other_label in range(5):
            if other_label == int(label):
                continue
            relu_check_list = STABLE_PATTERNS[label]["stable_idx"]
            relu_val = STABLE_PATTERNS[label]["val"] 
            exit_code, running_time = check_pattern(relu_check_list, relu_val, label=int(label), other_label = other_label)
            if exit_code=="sat":
                res[int(label)][other_label] = "SAT:{}".format(running_time/10**6)
            elif exit_code=="unsat":
                res[int(label)][other_label] = "UNS:{}".format(running_time/10**6)
            else:
                res[int(label)][other_label] = exit_code

    res = pandas.DataFrame(res)
    print(res)
main()