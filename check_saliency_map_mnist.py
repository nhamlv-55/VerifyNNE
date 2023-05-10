from BanditNet import Net
from typing import Any, Dict, Optional, Tuple, List
from AcasXuNet import AcasXu
from BaseNet import BaseNet
from maraboupy import MarabouCore, Marabou, MarabouUtils
from maraboupy.MarabouNetwork import MarabouNetwork
import torch
from onnx2pytorch import ConvertModel
import onnx
import numpy as np
import numpy.typing as npt
from utils import set_default_bound, _write, load_vnnlib
import json
import pickle as pkl
torch.set_default_tensor_type(torch.DoubleTensor)



def build_saliency_mask_query(network: BaseNet, 
                              input: npt.ArrayLike,
                              adv_label: int,
                              true_label:int, 
                              input_eps: float,
                              grad_eps: float,
                              sanity_check: bool) -> MarabouCore.InputQuery:
    if sanity_check:
        assert input_eps == 0, print("When checking for sanity, we must set input_eps to 0")
        assert grad_eps == 0, print("When checking for sanity, we must set grad_eps to 0")
    ipq: MarabouCore.InputQuery = network.build_marabou_ipq()
    marabou_net: MarabouNetwork = network.marabou_net

    assert marabou_net.inputVars[0].shape == input.shape

    """
    Extracting just the forward query. For debugging only
    """
    set_default_bound(network.fw_ipq, range(marabou_net.numVars), -10, 10)
    # set input for the forward query
    input_vars = np.array(marabou_net.inputVars).squeeze().flatten()
    for v in input_vars:
        network.fw_ipq.setLowerBound(v, max(0, network.fused_bounds[v][0]-input_eps))
        network.fw_ipq.setUpperBound(v, min(1, network.fused_bounds[v][1]+input_eps))
    Marabou.saveQuery(network.fw_ipq, "forwardQuery_smallConv")

    """
    Setup full network (forward + backward)
    """
    # set_default_bound(ipq, range(marabou_net.numVars), -2, 2)
    # set_default_bound(ipq, range(marabou_net.numVars,
                    #   marabou_net.numVars*2), -2, 2)

    #compute the true saliency map for input
    _tmp_inp = torch.autograd.Variable(torch.Tensor(input), requires_grad = True)
    _out = network.pytorch_net(_tmp_inp)
    _loss = _out[0][true_label]
    _loss.backward()

    true_grad = _tmp_inp.grad.flatten().detach().numpy().tolist()
    print("True grad computed using Pytorch:", true_grad)


    input_vars: List[int] = np.array(marabou_net.inputVars).flatten().tolist()
    print(input_vars)
    output_vars: List[int] = np.array(marabou_net.outputVars).flatten().tolist()
    print(output_vars)

    input: List[float] = input.flatten().tolist()

    for v in input_vars:
        ipq.setLowerBound(v, max(0, input[v] - input_eps))
        ipq.setUpperBound(v, min(1, input[v] + input_eps))

    # set other pre computed bounds:
    for v in network.fused_bounds:
        if v in input_vars: continue # do not set bound of input
        if v - marabou_net.numVars in output_vars: continue # do not set bound of grad of output
        if v - marabou_net.numVars in input_vars: continue # do not set bound of grad of input node
        lower, upper = network.fused_bounds[v]
        print(f"setting bound for {v} : [{lower}, {upper}]")
        ipq.setLowerBound(v, lower)
        ipq.setUpperBound(v, upper)

    # set grad of the output layer
    for vid, v in enumerate(output_vars):
        grad_v: int = v + marabou_net.numVars
        print(f"setting bound for the grad node {grad_v} of the output layer")
        if vid == true_label:
            print(f"setting {vid} ({grad_v}) to 1")
            ipq.setLowerBound(grad_v, 1)
            ipq.setUpperBound(grad_v, 1)
        else:
            ipq.setLowerBound(grad_v, 0)
            ipq.setUpperBound(grad_v, 0)

    # set constraints for the grad of the input layer:
    print("bounding the saliency map...")
    for v in input_vars:
        grad_v = v + marabou_net.numVars
        ipq.setLowerBound(grad_v, true_grad[v] - grad_eps)
        ipq.setUpperBound(grad_v, true_grad[v] + grad_eps)
    # set output constraints:
    # can logits for target > logits for true_label?


    return ipq


def prepare_benchmarks(benchmark: int)-> Tuple[BaseNet, torch.Tensor, 
                                               float, int, List[int], 
                                               str|None, 
                                               Dict[str, List[int]]]:
    if benchmark == 1:
        toy: BaseNet = BaseNet(ConvertModel(onnx.load("datasets/onnxVNNCOMP2022/mnist-net_256x2.onnx")))

        toy.pytorch_net.to(dtype=torch.double)
        input, eps, true_label, adv_labels = load_vnnlib("datasets/mnist_fc2022_specs/prop_0_0.03.vnnlib")
        precomputed_bounds = "pre_computed_bounds.pkl"
        # PRECOMPUTED_BOUNDS = None
        layer_map = {"/grad_norm": list(range(1))}
        return toy, input, eps, true_label, adv_labels, precomputed_bounds, layer_map
    elif benchmark == 2:
        banditNet = Net()
        banditNet.load_state_dict(torch.load("BanditNet.pth"))
        toy: BaseNet = BaseNet(banditNet)
        toy.pytorch_net.to(dtype=torch.double)
        input = torch.Tensor([1, 1, 0, 1, 1])
        eps = 0.5
        true_label = 1
        adv_labels = [0]
        precomputed_bounds = None
        offset = 31
        layer_map = {
            "/0": list(range(0, 5)),
            "/7": list(range(7, 13)),
            "/8": list(range(13, 19)),
            "/9": list(range(19, 25)),
            "/10": list(range(25, 31)),
            "/11": list(range(5, 7)),
            "/grad/0": list(range(0 + offset, 5 + offset)),
            "/grad/7": list(range(7 + offset, 13 + offset)),
            "/grad/8": list(range(13 + offset, 19 + offset)),
            "/grad/9": list(range(19 + offset, 25 + offset)),
            "/grad/10": list(range(25 + offset, 31 + offset)),
            "/grad/11": list(range(5 + offset,7 + offset)),
             }

        return toy, input, eps, true_label, adv_labels, precomputed_bounds, layer_map

def check():
    toy, input, eps, true_label, adv_labels, precomputed_bounds, layer_map = prepare_benchmarks(2)

    input = input.unsqueeze(0)
    np_input = input.numpy()
    input = torch.autograd.Variable(input, requires_grad = True)

    toy.build_marabou_net(input)
    modules = [m for m in toy.pytorch_net.modules()][1:]
    #compute the output and grad using pytorch
    outs = []
    out = input
    print(out)
    for m in modules:
        print(m)
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

    if precomputed_bounds is None:
        print("recomputing bounds...")
        toy.compute_jacobian_bounds(input, 0, true_label)

        with open("pre_computed_bounds.pkl", "wb") as f:
            pkl.dump({"forward_bounds": toy.forward_bounds,
                    "grad_bounds": toy.grad_bounds,
                    }, f)
    else:
        with open(precomputed_bounds, "rb") as f:
            toy.load_jacobian_bounds(pkl.load(f))

    toy.fusion(layer_map, 0)
    for adv_label in adv_labels: 
        ipq = build_saliency_mask_query(
            toy, input=np_input, adv_label=adv_label, true_label=true_label,
            input_eps=0.5,
            grad_eps=0.1,
            sanity_check=False)
        MarabouCore.saveQuery(ipq, "finalQuery_smallConv")

check()
