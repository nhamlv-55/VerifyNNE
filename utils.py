"""
Utility class to parse ACAS
"""
from selectors import EpollSelector
from typing import List, Tuple
import torch
import os
from pysmt.smtlib.parser import SmtLibParser, SmtLibCommand
from pysmt.fnode import FNode


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
