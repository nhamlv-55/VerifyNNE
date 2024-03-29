from BaseNet import BaseNet, get_activation
from torch import Tensor
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from datetime import datetime
import copy
from utils import CommaString
from typing import Callable, List, Sequence, Dict
import enum

class AcasOut(enum.IntEnum):
    CLEAR_OF_CONFLICT = 0
    WEAK_LEFT = 1
    WEAK_RIGHT = 2
    STRONG_LEFT = 3
    STRONG_RIGHT = 4
class AcasIn(enum.IntEnum):
    RHO = 0 #distance between ownship and intruder
    THETA = 1 #angle to intruder w.r.t ownship heading
    PSI = 2 #heading of intruder w.r.t ownship
    V_OWN = 3 #velocity of ownship
    V_INT = 4 #velocity of intruder
class AcasXu(BaseNet):
    def __init__(self, AcasPath: str):
        super().__init__()
        self.modelpath = AcasPath
        self.input_size: int = 0
        self.output_size: int = 0
        self.means: List[float] = []
        self.ranges: List[float] = []
        self.mins: List[float] = []
        self.maxs: List[float] = []
        self.layers = nn.ModuleList()
        self.read_acas(AcasPath)

    def read_acas(self, AcasPath: str)->None:
        """
        Borrow from https://github.com/XuankangLin/ART/blob/master/art/acas.py
        """
        # ===== Basic Initializations =====
        _num_layers = 0  # Number of layers in the network (excluding inputs, hidden + output = num_layers).
        _input_size = 0  # Number of inputs to the network.
        _output_size = 0  # Number of outputs to the network.
        _max_layer_size = 0  # Maximum size dimension of a layer in the network.

        _layer_sizes: List[int] = []  # Array of the dimensions of the layers in the network.

        _symmetric = False  # Network is symmetric or not. (was 1/0 for true/false)

        _mins: List[float] = []  # Minimum value of inputs.
        _maxs: List[float] = []  # Maximum value of inputs.

        _means: List[float] = []  # Array of the means used to scale the inputs and outputs.
        _ranges:List[float] = []  # Array of the ranges used to scale the inputs and outputs.

        _layer_weights: List[Tensor] = []  # holding concrete weights of each layer
        _layer_biases: List[Tensor] = []  # holding concrete biases of each layer


        with open(AcasPath, 'r') as f:
            line = f.readline()
            while line.startswith('//'):
                # ignore first several comment lines
                line = f.readline()

            # === Line 1: Basics ===
            data = CommaString(line)
            _num_layers = data.read_next_as_int()
            _input_size = data.read_next_as_int()
            _output_size = data.read_next_as_int()
            _max_layer_size = data.read_next_as_int()

            # === Line 2: Layer sizes ===
            data = CommaString(f.readline())
            for _ in range(_num_layers + 1):
                _layer_sizes.append(data.read_next_as_int())

            assert _layer_sizes[0] == _input_size
            assert _layer_sizes[-1] == _output_size
            assert all(size <= _max_layer_size for size in _layer_sizes)
            assert len(_layer_sizes) >= 2, f'Loaded layer sizes have {len(_layer_sizes)} (< 2) elements?! Too few.'

            # === Line 3: Symmetric ===
            data = CommaString(f.readline())
            _symmetric = data.read_next_as_bool()
            assert _symmetric is False, "We don't know what symmetric==True means."

            # It has to read by line, because in following lines, I noticed some files having more values than needed..

            # === Line 4: Mins of input ===
            data = CommaString(f.readline())
            for _ in range(_input_size):
                _mins.append(data.read_next_as_float())

            # === Line 5: Maxs of input ===
            data = CommaString(f.readline())
            for _ in range(_input_size):
                _maxs.append(data.read_next_as_float())

            # === Line 6: Means ===
            data = CommaString(f.readline())
            # the [-1] is storing the size for output normalization
            for _ in range(_input_size + 1):
                _means.append(data.read_next_as_float())

            # === Line 7: Ranges ===
            data = CommaString(f.readline())
            # the [-1] is storing the size for output normalization
            for _ in range(_input_size + 1):
                _ranges.append(data.read_next_as_float())

            # === The rest are layer weights/biases. ===
            for k in range(_num_layers):
                in_size: int = _layer_sizes[k]
                out_size: int = _layer_sizes[k + 1]

                # read "weights"
                tmp: List[List[float]] = []
                for i in range(out_size):
                    row: List[float] = []
                    data = CommaString(f.readline())
                    for j in range(in_size):
                        row.append(data.read_next_as_float())
                    tmp.append(row)
                    assert not data.has_next_comma()

                """ To fully comply with NNET in Reluplex, DoubleTensor is necessary.
                    Otherwise it may record 0.613717 as 0.6137170195579529.
                    But to make everything easy in PyTorch, I am just using FloatTensor.
                """
                _layer_weights.append(torch.tensor(tmp))

                # read "biases"
                tmp: List[float] = []
                for i in range(out_size):
                    # only 1 item for each
                    data = CommaString(f.readline())
                    tmp.append(data.read_next_as_float())
                    assert not data.has_next_comma()

                _layer_biases.append(torch.tensor(tmp))
                pass

            data = CommaString(f.read())
            assert not data.has_next_comma()  # should have no more data

        ##Reading is done. Setup the network
        try:
            self.input_size = _input_size
            self.output_size = _output_size
            self.means = _means
            self.ranges = _ranges
            self.mins = _mins
            self.maxs = _maxs
            for idx in range(len(_layer_sizes)-1):
                layer = nn.Linear(_layer_sizes[idx], _layer_sizes[idx+1])
                layer.weight.data = _layer_weights[idx]
                layer.bias.data = _layer_biases[idx]
                self.layers.append(layer)
        except Exception as e:
            print(e)

    def normalize_inputs(self, t: Tensor, mins: Sequence[float], maxs: Sequence[float]) -> Tensor:
        """ Normalize: ([min, max] - mean) / range """
        print(t.shape)
        slices: List[Tensor] = []
        for i in range(self.input_size):
            slice = t[:, i:i+1]
            slice = slice.clamp(mins[i], maxs[i])
            slice -= self.means[i]
            slice /= self.ranges[i]
            slices.append(slice)
        return torch.cat(slices, dim=-1)

    def denormalize_outputs(self, t: Tensor) -> Tensor:
        """ Denormalize: v * range + mean """
        # In NNET files, the mean/range of output is stored in [-1] of array.
        # All are with the same mean/range, so I don't need to slice.
        t *= self.ranges[-1]
        t += self.means[-1]
        return t
    def forward(self, x: Tensor)->Tensor:
        """ Normalization and Denomalization are called outside this method. """
        lin: torch.nn.Module
        for lid, lin in enumerate(self.layers[:-1]):
            x = lin(x)
            if(x.shape[0]==1): print(lid, x)
            x = F.relu(x)

        x = self.layers[-1](x)
        return x

    def register_log(self, detach:bool=True)->None:
        self.reset_hooks()
        # first layer should not make any difference?
        layer: torch.nn.Module
        for lidx, layer in enumerate(self.layers[:-1]):
            self.hooks.append(layer.register_forward_hook(get_activation('layer{}'.format(lidx),
                                self.tensor_log, detach=detach)))

    def dump_grad(self)->None:
        layer: torch.nn.Module
        for lidx, layer in enumerate(self.layers):
            print("----layer {}----".format(lidx))
            print(layer.weight.grad)
            print(layer.bias.grad)


    def compute_saliency_map(self, x: Tensor, label: int)->Tensor:
        """
            Given an input tensor of shape (n_inputs, ...), returns
            a saliency map tensor of the same size. 
        """
        n_inputs:int = x.shape[0]
        x = torch.autograd.Variable(x, requires_grad=True)
        saliency_map:List[Tensor] = []
        print(n_inputs)
        for idx in range(n_inputs):
            logits = self.forward(x)
            logits[idx][label].backward()
            saliency_map.append(x.grad[idx])
        return torch.stack(saliency_map)