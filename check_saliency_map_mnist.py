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
from torchviz import make_dot
from utils import set_default_bound, _write, load_vnnlib
import json
import pickle as pkl
torch.set_default_tensor_type(torch.DoubleTensor)

LAYER_MAP = {
    # "/input": "onnx::Gemm_8", 
             "/grad_norm": "grad_onnx::Flatten_0"
             }
PRECOMPUTED_BOUNDS = "pre_computed_bounds.pkl"

def build_saliency_mask_query(network: BaseNet, 
                              input: Optional[np.ndarray], 
                              target: int, lirpa_bounds: Dict[int, Tuple[float, float]]) -> MarabouCore.InputQuery:
    ipq: MarabouCore.InputQuery = network.build_marabou_ipq()
    marabou_net: MarabouNetwork = network.marabou_net

    assert marabou_net.inputVars[0].shape == input.shape

    """
    Extracting just the forward query. For debugging only
    """
    set_default_bound(network.fw_ipq, range(marabou_net.numVars), -2, 2)
    # set input for the forward query
    print("input vars", marabou_net.inputVars[0])
    if input is not None:
        for rid, row in enumerate(marabou_net.inputVars[0]):
            print(row)
            for vid, v in enumerate(row.tolist()):
                print(input[rid])
                network.fw_ipq.setLowerBound(v, input[0][rid][vid].item())
                network.fw_ipq.setUpperBound(v, input[0][rid][vid].item())
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

    # set other pre computed bounds:
    for v in lirpa_bounds:
        lower, upper = lirpa_bounds[v]
        logger.debug(f"setting bound for {v} : [{lower}, {upper}]")
        ipq.setLowerBound(v, lower)
        ipq.setUpperBound(v, upper)

    return ipq


def check():
    toy: BaseNet = BaseNet(ConvertModel(onnx.load("datasets/onnxVNNCOMP2022/mnist-net_256x2.onnx")))

    toy.pytorch_net.to(dtype=torch.double)
    input, eps, true_label, adv_labels = load_vnnlib("datasets/mnist_fc2022_specs/prop_0_0.03.vnnlib")
    input = input.unsqueeze(0)
    np_input = input.numpy()
    input = torch.autograd.Variable(input, requires_grad = True)

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
        _write(input.grad, f)
    print(toy.marabou_net.varMap)

    if PRECOMPUTED_BOUNDS is None:
        print("recomputing bounds...")
        toy.compute_jacobian_bounds(input, 0, true_label)

        with open("pre_computed_bounds.pkl", "wb") as f:
            pkl.dump({"forward_bounds": toy.forward_bounds,
                    "grad_bounds": toy.grad_bounds}, f)
    else:
        with open(PRECOMPUTED_BOUNDS, "rb") as f:
            toy.load_jacobian_bounds(pkl.load(f))

    lirpa_bounds = toy.fusion(LAYER_MAP, 0)
    
    ipq = build_saliency_mask_query(
        toy, input=np_input, target=(true_label+1)%10, lirpa_bounds=lirpa_bounds)
    MarabouCore.saveQuery(ipq, "finalQuery_smallConv")

check()
