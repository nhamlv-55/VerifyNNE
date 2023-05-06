from io import FileIO, TextIOWrapper
from typing import Any, Dict, Optional, Tuple, List
from AcasXuNet import AcasXu
from BaseNet import BaseNet
from maraboupy import MarabouCore, Marabou, MarabouUtils
from maraboupy.MarabouNetwork import MarabouNetwork
import torch
from onnx2pytorch import ConvertModel
import onnx
import numpy as np

from utils import set_default_bound, _write, load_vnnlib
torch.set_default_tensor_type(torch.DoubleTensor)

LAYER_MAP = {
    # "/input": "onnx::Gemm_8", 
             "/grad/input/params/1": "grad_onnx::Reshape_6"
             }


def build_saliency_mask_query(network: BaseNet, 
                              dummy_input: torch.Tensor, 
                              input: Optional[torch.Tensor], 
                              target: int) -> MarabouCore.InputQuery:
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


def check():
    toy: BaseNet = BaseNet(ConvertModel(onnx.load("datasets/onnxVNNCOMP2022/mnist-net_256x2.onnx")))
    input, eps, true_label, adv_labels = load_vnnlib("datasets/mnist_fc2022_specs/prop_0_0.03.vnnlib")

    input = torch.autograd.Variable(input.unsqueeze(0).to(torch.float), requires_grad = True)

    toy.build_marabou_net(input)
    modules = [m for m in toy.pytorch_net.modules() if not isinstance(m, ConvertModel)]
    #compute the output and grad using pytorch
    outs = []
    out = input
    for m in modules:
        print(m, type(m))
        out = m(out)
        out.retain_grad()
        outs.append(out)

    loss = out[0][true_label]
    loss.backward(retain_graph=True)

    with open("true_values.txt", "w") as f:
        for idx, out in enumerate(outs):
            _write(modules[idx], f)
            _write(out, f)
            _write("grad:", f)
            _write(out.grad, f)    

    print(toy.marabou_net.varMap)


    print("calculate using pytorch")
    print(input.shape)
    z1 = toy.pytorch_net.ln1(input.flatten())
    z1.retain_grad()
    h1 = toy.pytorch_net.relu(z1)
    h1 = h1.flatten()
    h1.retain_grad()
    print("h1 shape", h1.shape)
    z2 = toy.pytorch_net.ln2(h1)
    z2.retain_grad()
    out = z2
    out.retain_grad()
    # out = h1
    # out = z2
    print("*", toy.forward(input).squeeze(), out.squeeze())
    assert torch.allclose(toy.forward(input).squeeze(), 
                          out.squeeze()), "networks are not the same"

    loss = out[1]
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
    toy.build_marabou_net(input)
    toy.compute_jacobian_bounds(input, 0)

    for n in toy.immediate_bounds[0]:
        print(n, "\n", toy.immediate_bounds[0][n])
    
    toy.fusion(LAYER_MAP, 0)
    
    ipq = build_saliency_mask_query(
        toy, dummy_input=torch.randn(1, 1, 3, 3), input=input, target=1)
    MarabouCore.saveQuery(ipq, "finalQuery_smallConv")

check()
