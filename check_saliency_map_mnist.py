from BanditNet import Net
from typing import Any, Dict, Optional, Tuple, List
from AcasXuNet import AcasXu
from BaseNet import BaseNet
from maraboupy import MarabouCore, Marabou, MarabouUtils
from maraboupy.MarabouNetwork import MarabouNetwork, ZERO
import torch
from onnx2pytorch import ConvertModel
import onnx
import numpy as np
import numpy.typing as npt
from utils import set_default_bound, _write, load_vnnlib
from settings import MARABOU_BIN
import json
import pickle as pkl
import subprocess
import sys

NUM_THREADS = 30
TIMEOUT = 600
GRAD_EPS = 1.25
LIRPA_OPTION = {'optimize_bound_args': {'enable_beta_crown': True, 
                                        'fix_intermediate_layer_bounds': False, 
                                        'lr_alpha': 0.1, 
                                        'iteration': 200}}


# -----------------------------------------------------------------------------
def get_config():
    config_keys = [k for k,v in globals().items() if k.isupper()]
    # exec(open('configurator.py').read()) # overrides from command line or config file
    config = {k: globals()[k] for k in config_keys} # will be useful for logging
    return config
# -----------------------------------------------------------------------------

torch.set_default_tensor_type(torch.DoubleTensor)

def build_saliency_mask_query(network: BaseNet, 
                              input: npt.ArrayLike,
                              adv_label: int,
                              true_label:int, 
                              input_eps: float,
                              grad_eps: float,
                              sanity_check: bool,
                              top_k:int) -> MarabouCore.InputQuery:
    if sanity_check:
        assert input_eps == 0, print("When checking for sanity, we must set input_eps to 0")
        assert grad_eps == 0, print("When checking for sanity, we must set grad_eps to 0")

    network.marabou_net.buildBackwardConstraints()
    ipq = network.marabou_net.FB_ipq
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
        lower = true_grad[v] - grad_eps
        upper = true_grad[v] + grad_eps
        print(f"setting node {grad_v} to bound ({lower}, {upper})")
        ipq.setLowerBound(grad_v, lower)
        ipq.setUpperBound(grad_v, upper)
    # set output constraints:
    # can logits for target > logits for true_label?
    if sanity_check:
        constraint = MarabouUtils.Equation(MarabouCore.Equation.LE)
        constraint.setScalar(-ZERO)
    else:
        constraint = MarabouUtils.Equation(MarabouCore.Equation.GE)
        constraint.setScalar(ZERO)
    constraint.addAddend(1, adv_label + output_vars[0])
    constraint.addAddend(-1, true_label + output_vars[0])

    ipq.addEquation(constraint.toCoreEquation()) 

    to_be_abstracted = network.marabou_net.grad_ins[:]
    network.add_backward_query(to_be_abstracted, abs_domain='box', top_k = top_k)
    return ipq


def prepare_benchmarks(benchmark_set: str, cex:int)-> Tuple[BaseNet, torch.Tensor, 
                                               float, int, List[int], 
                                               str|None, 
                                               Dict[str, List[int]]]:
    if benchmark_set == "mnist_fc_4":
        net: BaseNet = BaseNet(ConvertModel(onnx.load("datasets/onnxVNNCOMP2022/mnist-net_256x4.onnx")))
        net.pytorch_net.to(dtype=torch.double)
        if cex==1:
            input, eps, true_label, adv_labels = load_vnnlib("datasets/mnist_fc2022_specs/prop_1_0.03.vnnlib")
            adv_labels = [7]
        elif cex==2:
            input, eps, true_label, adv_labels = load_vnnlib("datasets/mnist_fc2022_specs/prop_1_0.05.vnnlib")
            adv_labels = [9]
        elif cex==3:
            input, eps, true_label, adv_labels = load_vnnlib("datasets/mnist_fc2022_specs/prop_6_0.05.vnnlib")
            adv_labels = [5]

        precomputed_bounds = None
        
        offset = 784+10+256*8
        layer_map = {"/0": list(range(0, 784)),
                "/21": list(range(0, 784)),

                "/input": list(range(794, 1050)),
                "/23": list(range(1050, 1306)),

                "/input.3": list(range(1306, 1562)),
                "/25": list(range(1562, 1818)),

                "/input.7": list(range(1818, 2074)),
                "/27": list(range(2074, 2330)),

                "/input.11": list(range(2330, 2586)),
                "/29": list(range(2586, 2842)),

                "/30": list(range(784, 794))
                }
        
        for l in list(layer_map.keys()):
            layer_map[f"/grad{l}"] = [v+offset for v in layer_map[l]]


        return net, input, eps, true_label, adv_labels, precomputed_bounds, layer_map
    
    elif benchmark_set == "mnist_fc_6":
        net: BaseNet = BaseNet(ConvertModel(onnx.load("datasets/onnxVNNCOMP2022/mnist-net_256x6.onnx")))
        net.pytorch_net.to(dtype=torch.double)
        if cex==1:
            input, eps, true_label, adv_labels = load_vnnlib("datasets/mnist_fc2022_specs/prop_6_0.03.vnnlib")
            adv_labels = [5]
        elif cex==2:
            input, eps, true_label, adv_labels = load_vnnlib("datasets/mnist_fc2022_specs/prop_6_0.05.vnnlib")
            adv_labels = [8]


        precomputed_bounds = None
        
        offset = 784+10+256*12
        layer_map = {"/0": list(range(0, 784)),
                "/29": list(range(0, 784)),

                "/input": list(range(794, 1050)),
                "/31": list(range(1050, 1306)),

                "/input.3": list(range(1306, 1562)),
                "/33": list(range(1562, 1818)),

                "/input.7": list(range(1818, 2074)),
                "/35": list(range(2074, 2330)),

                "/input.11": list(range(2330, 2586)),
                "/37": list(range(2586, 2842)),

                "/input.15": list(range(2842, 3098)),
                "/39": list(range(3098, 3354)),

                "/input.19": list(range(3098, 3610)),
                "/41": list(range(3610, 3866)),


                "/42": list(range(784, 794))
                }
        
        for l in list(layer_map.keys()):
            layer_map[f"/grad{l}"] = [v+offset for v in layer_map[l]]

        return net, input, eps, true_label, adv_labels, precomputed_bounds, layer_map


    elif benchmark_set == "mnist_fc_2":
        toy: BaseNet = BaseNet(ConvertModel(onnx.load("datasets/onnxVNNCOMP2022/mnist-net_256x2.onnx")))

        toy.pytorch_net.to(dtype=torch.double)
        if cex ==1:
            input, eps, true_label, adv_labels = load_vnnlib("datasets/mnist_fc2022_specs/prop_1_0.03.vnnlib")
            #NHAM: from https://github.com/ChristopherBrix/vnncomp2022_results/blob/master/marabou/mnist_fc/mnist-net_256x2_prop_1_0.03.counterexample.gz
            # we know that one possible adv label is 9
            adv_labels = [9]
        elif cex ==2: 
            input, eps, true_label, adv_labels = load_vnnlib("datasets/mnist_fc2022_specs/prop_0_0.05.vnnlib")
            adv_labels = [8]
        elif cex ==3: 
            input, eps, true_label, adv_labels = load_vnnlib("datasets/mnist_fc2022_specs/prop_12_0.03.vnnlib")
            adv_labels = [8]
        elif cex ==4: 
            input, eps, true_label, adv_labels = load_vnnlib("datasets/mnist_fc2022_specs/prop_12_0.05.vnnlib")
            adv_labels = [8]
        elif cex ==5: 
            input, eps, true_label, adv_labels = load_vnnlib("datasets/mnist_fc2022_specs/prop_1_0.05.vnnlib")
            adv_labels = [8]
        elif cex ==6: 
            input, eps, true_label, adv_labels = load_vnnlib("datasets/mnist_fc2022_specs/prop_2_0.03.vnnlib")
            adv_labels = [8]
        elif cex ==7: 
            input, eps, true_label, adv_labels = load_vnnlib("datasets/mnist_fc2022_specs/prop_2_0.05.vnnlib")
            adv_labels = [8]
        elif cex ==8:
            input, eps, true_label, adv_labels = load_vnnlib("datasets/mnist_fc2022_specs/prop_4_0.05.vnnlib")
            adv_labels = [3]
        elif cex ==9:
            input, eps, true_label, adv_labels = load_vnnlib("datasets/mnist_fc2022_specs/prop_5_0.05.vnnlib")
            adv_labels = [8]                       
        elif cex ==9:
            input, eps, true_label, adv_labels = load_vnnlib("datasets/mnist_fc2022_specs/prop_6_0.05.vnnlib")
            adv_labels = [8]                       
        elif cex ==10:
            input, eps, true_label, adv_labels = load_vnnlib("datasets/mnist_fc2022_specs/prop_8_0.05.vnnlib")
            adv_labels = [8]                       
        # precomputed_bounds = "pre_computed_bounds.pkl"
        precomputed_bounds = None
        
        offset = 784+10+256*4
        layer_map = {"/0": list(range(0, 784)),
                     "/13": list(range(0, 784)),

                     "/input": list(range(794, 1050)),
                     "/15": list(range(1050, 1306)),

                     "/input.3": list(range(1306, 1562)),
                     "/17": list(range(1562, 1818)),

                     "/18": list(range(784, 794))}
        
        for l in list(layer_map.keys()):
            layer_map[f"/grad{l}"] = [v+offset for v in layer_map[l]]
        
        print("layer map")
        for l in layer_map:
            print(l)
            print(layer_map[l])

        return toy, input, eps, true_label, adv_labels, precomputed_bounds, layer_map
    elif benchmark_set == "bandit":
        banditNet = Net()
        banditNet.load_state_dict(torch.load("BanditNet.pth"))
        toy: BaseNet = BaseNet(banditNet)
        toy.pytorch_net.to(dtype=torch.double)
        input = torch.Tensor([1, 1, 0, 1, 1])
        # input = torch.Tensor([0.5, 0.000014, 0, 0.5, 0])
        eps = 1
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

    elif benchmark=="cifar2020":
        cifar10: BaseNet = BaseNet(ConvertModel(onnx.load('datasets/vnncomp2022_benchmarks/benchmarks/cifar2020/onnx/cifar10_2_255_simplified.onnx')))

        cifar10.pytorch_net.to(dtype=torch.double)

        if cex==1:
            input, eps, true_label, adv_labels = load_vnnlib("datasets/vnncomp2022_benchmarks/benchmarks/cifar2020/vnnlib/cifar10_spec_idx_0_eps_0.00784_n1.vnnlib")

            precomputed_bounds = None


        layer_map = {}

        input = input.reshape(3, 32, 32)

        return cifar10, input, eps, true_label, adv_labels, precomputed_bounds, layer_map

def check(benchmark_set: str, cex:int):
    toy, input, eps, true_label, adv_labels, precomputed_bounds, layer_map = prepare_benchmarks(benchmark_set, cex=cex)

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
        toy.compute_jacobian_bounds(input, eps, true_label, option=LIRPA_OPTION)

        with open("pre_computed_bounds.pkl", "wb") as f:
            pkl.dump({"forward_bounds": toy.forward_bounds,
                    "grad_bounds": toy.grad_bounds,
                    }, f)
    else:
        with open(precomputed_bounds, "rb") as f:
            toy.load_jacobian_bounds(pkl.load(f))

    toy.fusion(layer_map, true_label)
    for adv_label in adv_labels: 
        queryFile = "finalQuery"
        #build the query
        ipq = build_saliency_mask_query(
            toy, input=np_input, adv_label=adv_label, true_label=true_label,
            input_eps=eps,
            grad_eps=GRAD_EPS,
            sanity_check=False,
            top_k=0
            ) #NHAM: top_k = 0 is the same as doing no abstraction at all
        MarabouCore.saveQuery(ipq, queryFile)

        #solve the query using Marabou binary
        marabouRes = subprocess.run([MARABOU_BIN, 
                                    f"--input-query={queryFile}",
                                    "--snc",
                                    f"--num-workers={NUM_THREADS}",
                                    "--export-assignment",
                                    f"--timeout={TIMEOUT}"],
                                    capture_output=True,
                                    text=True, 
                                    timeout=TIMEOUT+10)

        #save results
        with open(f"benchmark_config.json", "w") as f:
            json.dump(get_config(), f, indent=2)

        with open(f"solving_stdout_{benchmark_set}_{cex}", "w") as f:
            f.write(marabouRes.stdout)

        with open(f"solving_stderr_{benchmark_set}_{cex}", "w") as f:
            f.write(marabouRes.stderr)

        

check(benchmark_set=sys.argv[1], cex= int(sys.argv[2]))
