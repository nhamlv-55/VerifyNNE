"""
This script will parse the dumped Marabou Input Query into a more
human-friendly form
"""
import sys
from typing import List, Tuple
from enum import Enum
input_file = sys.argv[1]


class blockType(Enum):
    OPEN = 1
    INPUT_NODE_INFOS = 2
    OUTPUT_NODE_INFOS = 3
    LBS = 4
    UBS = 5
    LINEAR = 6
    NONLINEAR = 7

def print_block(block: List[str], block_type: blockType) -> None:
    print("-"*10)
    print(block_type)
    if block_type == block_type.INPUT_NODE_INFOS:
        for l in block:
            print(l.strip())
    elif block_type == block_type.LINEAR:
        for l in block:
            tokens: List[str] = l.split(",")
            addends: List[Tuple[int, float]] = []
            str_rep: List[str] = []
            for idx in range(1, len(tokens), 2):
                var_idx, coeff = (int(tokens[idx]), float(tokens[idx+1]))
                addends.append((var_idx, coeff))
                str_rep.append("{:.4f}*v{}".format(coeff, var_idx))

            assert addends[-1][1] == -1
            line = "eq{}: v{} = {} + {}".format(tokens[0], #eq number
                                                addends[-1][0],  # output
                                                # weightedsum
                                                " + ".join(str_rep[1:-1]),
                                                addends[0][1]  # bias
                                                )
            print(line)
    elif block_type == block_type.NONLINEAR:
        for l in block:
            tokens: List[str] = l.strip().split(",")
            node_type:str = tokens[1]
            if node_type=="relu":
                assert len(tokens)==4
                print("eq{}: v{} = relu(v{})".format(tokens[0], tokens[2], tokens[3]))
    else:
        for l in block:
            print(l.strip())


with open(input_file, "r") as f:
    raw = f.readlines()

n_neurons: int = int(raw[0].strip())
n_lower_bounds: int = int(raw[1].strip())
n_upper_bounds: int = int(raw[2].strip())
n_eqs: int = int(raw[3].strip())
n_constraints: int = int(raw[4].strip())

n_inputs: int = int(raw[5].strip())
input_node_lines = raw[5:5+n_inputs+1]
print_block(input_node_lines, blockType.INPUT_NODE_INFOS)


n_outputs = int(raw[5+n_inputs+1].strip())
output_node_lines = raw[5+n_inputs+1: 5+n_inputs+1+n_outputs+1]
print_block(output_node_lines, blockType.OUTPUT_NODE_INFOS)


# lowerbounds

current_line: int = 5+n_inputs+1+n_outputs+1
lb_lines = raw[current_line:current_line+n_lower_bounds]
print_block(lb_lines, blockType.LBS)

# upperbounds
current_line += n_lower_bounds
ub_lines = raw[current_line:current_line+n_upper_bounds]
print_block(ub_lines, blockType.UBS)

# linear consts:
current_line += n_upper_bounds
linear_lines = raw[current_line: current_line+n_eqs]
print_block(linear_lines, blockType.LINEAR)

#nonlinear
current_line +=n_eqs
nonlin_lines = raw[current_line:]
print_block(nonlin_lines, blockType.NONLINEAR)