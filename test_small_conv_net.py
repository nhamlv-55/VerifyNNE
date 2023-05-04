from io import FileIO, TextIOWrapper
from typing import Any, Dict, Optional, Tuple, List
from AcasXuNet import AcasXu
from BaseNet import BaseNet
from maraboupy import MarabouCore, Marabou, MarabouUtils
from maraboupy.MarabouNetwork import MarabouNetwork
import torch
from torch.utils.data import TensorDataset, DataLoader
import onnx
import numpy as np

from utils import set_default_bound, _write
torch.set_default_tensor_type(torch.DoubleTensor)

W1 = [[[[.101, .102],
        [.111, -.112]
       ]],
      [[[-.201, -.202],
        [.211, -.212]
      ]
      ]]
B1 = [0.01, -0.095]
W2 = [[.301, .302, -.303, .304, -.305, .306, .307, -.308],
      [.311, .312, .313, -.314, .315, .316, -.317, -.318]]
B2 = [0.007, -0.1]
W3 = [[.301, .302],
      [.311, .312]]


class ToyConvNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(
            in_channels=1, out_channels=2, kernel_size=2)
        self.ln1 = torch.nn.Linear(8, 2, bias=True)
        self.relu = torch.nn.ReLU()
        # fixed weight
        with torch.no_grad():
            self.conv1.weight = torch.nn.Parameter(torch.Tensor(W1))
            self.conv1.bias = torch.nn.Parameter(
                torch.Tensor(B1))
            self.ln1.weight = torch.nn.Parameter(torch.Tensor(W2))
            self.ln1.bias = torch.nn.Parameter(torch.Tensor(B2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.relu(x)
        x = x.reshape(-1, 8)
        x = self.ln1(x)

        return x


def build_saliency_mask_query(network: BaseNet, dummy_input: torch.Tensor, input: Optional[torch.Tensor], target: int) -> MarabouCore.InputQuery:
    marabou_net = network.build_marabou_net(dummy_input=dummy_input)
    ipq: MarabouCore.InputQuery = network.build_marabou_ipq()

    """
    Extracting just the forward query. For debugging only
    """
    set_default_bound(network.fw_ipq, range(marabou_net.numVars), -2, 2)
    # set input for the forward query
    print("input vars", marabou_net.inputVars[0])
    if input is not None:
        for rid, row in enumerate(marabou_net.inputVars[0][0][0]):
            print(row)
            for vid, v in enumerate(row.tolist()):
                print(input[0][0][rid])
                network.fw_ipq.setLowerBound(v, input[0][0][rid][vid].item())
                network.fw_ipq.setUpperBound(v, input[0][0][rid][vid].item())
    Marabou.saveQuery(network.fw_ipq, "forwardQuery_smallConv")

    """
    Setup full network (forward + backward)
    """
    set_default_bound(ipq, range(marabou_net.numVars), -2, 2)
    set_default_bound(ipq, range(marabou_net.numVars,
                      marabou_net.numVars*2), -2, 2)

    print("inputVars", marabou_net.inputVars[0])
    print("outputVars", marabou_net.outputVars[0])
    # set input
    if input is not None:
        for rid, row in enumerate(marabou_net.inputVars[0][0][0]):
            print(row)
            for vid, v in enumerate(row.tolist()):
                print(input[0][0][rid])
                ipq.setLowerBound(v, input[0][0][rid][vid].item())
                ipq.setUpperBound(v, input[0][0][rid][vid].item())

    # set grad
    for vid, v in enumerate(marabou_net.outputVars[0][0]):
        grad_v: int = v + marabou_net.numVars
        if vid == target:
            ipq.setLowerBound(grad_v, 1)
            ipq.setUpperBound(grad_v, 1)
        else:
            ipq.setLowerBound(grad_v, 0)
            ipq.setUpperBound(grad_v, 0)

    return ipq


def run_toy():
    toy = BaseNet(ToyConvNet())
    input = torch.autograd.Variable(torch.Tensor(
        [[[[0.11, 0.12, 0.13],
           [0.21, 0.22, 0.23],
           [0.31, 0.32, 0.33],
        ]]]), requires_grad=True)
    print("calculate using pytorch")
    print(input.shape)
    z1 = toy.pytorch_net.conv1(input)
    z1.retain_grad()
    h1 = toy.pytorch_net.relu(z1)
    h1 = h1.flatten()
    h1.retain_grad()
    print("h1 shape", h1.shape)
    z2 = toy.pytorch_net.ln1(h1)
    z2.retain_grad()
    out = z2
    out.retain_grad()
    # out = h1
    # out = z2
    print("*", toy.forward(input).squeeze(), out.squeeze())
    assert torch.allclose(toy.forward(input).squeeze(), 
                          out.squeeze()), "networks are not the same"

    loss = out[0]
    loss.backward(retain_graph=True)
    print(input, input.grad)
    with open("true_values.txt", "w") as f:
        _write(input, f)
        _write(input.grad, f)
        _write(z1, f)
        _write(z1.grad, f)
        _write(h1, f)
        _write(h1.grad, f)
        _write(z2, f)
        _write(z2.grad, f)

    toy.compute_jacobian_bounds(input, 0)


    ipq = build_saliency_mask_query(
        toy, dummy_input=torch.randn(1, 1, 3, 3), input=input, target=0)
    MarabouCore.saveQuery(ipq, "finalQuery_smallConv")

run_toy()
