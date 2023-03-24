"""
Utility class to parse ACAS
"""
from __future__ import annotations
from typing import Any, Dict, Optional, Tuple, List
import torch
import os
from pysmt.smtlib.parser import SmtLibParser, SmtLibCommand
from pysmt.fnode import FNode
from maraboupy import MarabouCore, Marabou
import numpy as np
from io import FileIO, TextIOWrapper
import json
import random

class Pattern:
    def __init__(self):
        self.activated: List[int] = []
        self.deactivated: List[int] = []
        self.n_fixed_relus = 0 

    def finalize(self):
        self.activated.sort()
        self.deactivated.sort()
        self.n_fixed_relus = len(self.activated) + len(self.deactivated)

    def from_check_list(self, relu_check_list: List[int], relu_val: List[int]):
        assert(len(relu_check_list) == len(relu_val))
        for i in range(len(relu_check_list)):
            if relu_val[i] < 2000:
                self.deactivated.append(relu_check_list[i])
            else:
                self.activated.append(relu_check_list[i])

        self.finalize()

    def from_Marabou_output(self, mReluList: List[Tuple[int, int]], outputDict: Dict[int, float]):
        offset = 28*28+10
        for i, o in mReluList:
            if outputDict[i] > 0:
                self.activated.append(i-offset)
            else:
                self.deactivated.append(i-offset)
         
        self.finalize()

    def strengthen(self, NAP_bar: Pattern, cex_pattern: Pattern)->None:
        print("NAP_bar", NAP_bar)
        print("cex_pattern", cex_pattern)
        assert(set(cex_pattern.activated) >= set(self.activated))
        assert(set(cex_pattern.deactivated) >= set(self.deactivated))
        delta_A = set(NAP_bar.activated).intersection( set(cex_pattern.activated) - set(self.activated))
        delta_D = set(NAP_bar.deactivated).intersection( set(cex_pattern.deactivated) - set(self.deactivated))
        delta_A = sorted(list(delta_A))
        delta_D = sorted(list(delta_D))
        print("delta_A:{}".format(delta_A))
        print("delta D:{}".format(delta_D))
        print("Strenthen NAP using CEX...")
        rand = random.randint(0, 1)
        if rand==0:
            # flip_neuron = random.randint(0, len(delta_D)-1)
            flip_neuron = delta_D[-1]
            assert flip_neuron not in self.activated, "rand_neuron is in self.activated"
            self.activated.append(flip_neuron)
        else:
            # flip_neuron = random.randint(0, len(delta_A)-1)
            flip_neuron = delta_A[-1]
            assert flip_neuron not in self.deactivated, "rand_neuron is in self.deactivated"
            self.deactivated.append(flip_neuron)
        print("flip {}".format(flip_neuron))
        self.finalize()

    def __str__(self) -> str:
        return "A:{}\nD:{}".format(self.activated, self.deactivated) 


def get_pattern(marabou_network: Marabou.MarabouNetwork, input: Any)->Pattern:
    _, outputDict = marabou_network.evaluate(input)
    pred_label = np.argmax(_)
    print("pred label: {}".format(pred_label))
    
    pattern = Pattern()
    pattern.from_Marabou_output(marabou_network.reluList, outputDict)

    return pattern    

"""
A helper function to set the default bound for all nodes in the Marabou Input Query
"""
def set_default_bound(ipq: MarabouCore.InputQuery, range: List[int], l: float, u: float)->None:
    for v in range:
        ipq.setLowerBound(v, l)
        ipq.setUpperBound(v, u)
    return ipq
"""
A helper function to convert entries in a saliency map to either -1 or 1.
"""

def normalize_sm(sm, k: int) -> Tuple[Any, Any, Any]:
    """
    expect sm to be flatten already
    """
    top_k_idx = np.argpartition(sm, -k)[-k:].tolist()
    top_lowk_idx = np.argpartition(sm, k)[:k].tolist()
    print(top_k_idx)
    print(top_lowk_idx)
    new_sm = np.array([0]*28*28)
    for idx in top_k_idx:
        new_sm[idx] = 1
    for idx in top_lowk_idx:
        new_sm[idx] = -1
    return new_sm, top_k_idx, top_lowk_idx

"""
A helper function to write numpy array to file
"""
def _write(o: Any, f: TextIOWrapper):
    print(o)
    if type(o) == torch.Tensor:
        f.write(np.array2string(o.detach().numpy())+"\n")
    else:
        f.write(o.str()+"\n")


class CommaString(object):
    """ A full string separated by commas. """

    def __init__(self, text: str):
        self.text = text
        return

    def __str__(self):
        return self.text

    def __repr__(self):
        return self.text

    def has_next_comma(self) -> bool:
        return ',' in self.text

    def _read_next(self) -> str:
        """
        :return: the raw string of next token before comma
        """
        if self.has_next_comma():
            token, self.text = self.text.split(',', maxsplit=1)
        else:
            token, self.text = self.text, ''
        return token.strip()

    def read_next_as_int(self) -> int:
        return int(self._read_next())

    def read_next_as_float(self) -> float:
        return float(self._read_next())

    def read_next_as_bool(self) -> bool:
        """ Parse the next token before comma as boolean, 1/0 for true/false. """
        num = self.read_next_as_int()
        assert num == 1 or num == 0, f'The should-be-bool number is {num}.'
        return bool(num)


"""
Utility class to read vnncomp benchmark
"""

def parse_raw_idx(raw_idx: int) -> Tuple[int, int, int]:
    """
    only for MNIST 256xk network:
    """
    offset = 28*28+10
    n_relus = 256
    layer = raw_idx // n_relus
    idx = raw_idx % n_relus
    marabou_idx = 2*n_relus*layer + idx + offset
    return layer, idx, marabou_idx


def load_vnnlib(path: str) -> Tuple[torch.Tensor, float, int, List[int]]:
    # a hack to detect `nice` eps
    basename = os.path.basename(path)
    v = basename.split(".")[1]
    eps = float("0.{}".format(v))

    parser = SmtLibParser()
    script = parser.get_script(open(path, "r"))
    x = {}
    x_lower = {}
    x_upper = {}
    x_bounds = {}
    true_label = -1
    adv_labels: List[int] = []
    cmd: SmtLibCommand
    for cmd in script:
        """
        read all var decl. There should be only X_ var and Y_ var
        """
        if cmd.name == "declare-const":
            assert len(cmd.args) == 1
            var_name: str = cmd.args[0].symbol_name()
            if var_name.startswith("X"):
                var, index = var_name.split("_")
                index = int(index)
                x[index] = 0
        elif cmd.name == "assert":
            """
            read all constraints
            """
            root: Fnode = cmd.args[0]
            # pysmt only has less than + less than equal. Greater than are converted to less than
            if root.is_le():
                c1, c2 = root.args()
                if c1.is_constant():
                    # c <=X
                    x_lower[c2.symbol_name()] = float(c1.constant_value())
                else:
                    # X <=c
                    x_upper[c1.symbol_name()] = float(c2.constant_value())
            if root.is_or():
                for c in root.args():
                    assert c.is_le()
                    # var should be in the form Y_something
                    true_label = int(c.arg(0).symbol_name().split("_")[1])
                    adv_labels.append(
                        int(c.arg(1).symbol_name().split("_")[1]))
                assert true_label not in adv_labels
    # compute x_val, join upper and lower:
    x_bounds = [(float('-inf'), float('inf'))]*len(x)
    x_vals = [0]*len(x)
    assert len(x) == len(x_lower)
    assert len(x) == len(x_upper)
    for k in x_lower:
        assert k in x_upper
        idx = int(k.split("_")[1])
        if x_lower[k] == 0:
            x_vals[idx] = max(x_upper[k]-eps, 0)
        else:
            x_vals[idx] = x_lower[k]+eps

        x_bounds[idx] = (x_lower[k], x_upper[k])

    return torch.Tensor(x_vals), eps, true_label, adv_labels
