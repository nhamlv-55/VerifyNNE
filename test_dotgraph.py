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
W1 = [[.1, .2],
      [-.111, .112],
      [-.121, .122]
      ]
B1 = [0.01, 0.04, -0.06]
W2 = [[.201, -.202, .203],
      [.211, -.212, .213]]
B2 = [-0.007, 0.1]
W3 = [[.301, .302],
      [.311, .312]]


class ToyNet(BaseNet):
    def __init__(self):
        super().__init__()
        #input: [x0, x1]
        #l1: [z0, z1, z2]
        #relu: []
        #output: [y0, y1]
        # self.model = torch.nn.Sequential()
        # self.model.add_module('W0', torch.nn.Linear(2, 3, bias = False))
        # self.model.add_module('relu', torch.nn.ReLU())
        # self.model.add_module('W1', torch.nn.Linear(3, 2, bias = False))

        self.ln1 = torch.nn.Linear(2, 3, bias=True)
        self.ln2 = torch.nn.Linear(3, 2, bias=True)
        self.ln3 = torch.nn.Linear(2, 2, bias=False)
        self.relu = torch.nn.ReLU()
        # fixed weight
        with torch.no_grad():
            self.ln1.weight = torch.nn.Parameter(torch.Tensor(W1))
            self.ln1.bias = torch.nn.Parameter(
                torch.Tensor(B1))
            self.ln2.weight = torch.nn.Parameter(torch.Tensor(W2))
            self.ln2.bias = torch.nn.Parameter(torch.Tensor(B2))
            self.ln3.weight = torch.nn.Parameter(torch.Tensor(W3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # return self.ln2(self.relu(self.ln1(x)))
        return self.ln3(self.relu(self.ln2(self.relu(self.ln1(x)))))
        # return self.relu(self.ln1(x))


class ToyOnnxNativeNet(BaseNet):
    def __init__(self):

        super().__init__()

        node = onnx.helper.make_node(
            'Gemm',
            inputs=['a', 'b', 'c'],
            outputs=['y'],
            alpha=0.25,
            beta=0.35,
            transA=1,
            transB=1
        )
        a = np.random.ranf([4, 3]).astype(np.float32)
        b = np.random.ranf([5, 4]).astype(np.float32)
        c = np.random.ranf([1, 5]).astype(np.float32)
        y = gemm_reference_implementation(
            a, b, c, transA=1, transB=1, alpha=0.25, beta=0.35)
        expect(node, inputs=[a, b, c], outputs=[y],
               name='test_gemm_all_attributes')

# helper function


def set_default_bound(ipq: MarabouCore.InputQuery, range: List[int], l: float, u: float):
    for v in range:
        ipq.setLowerBound(v, l)
        ipq.setUpperBound(v, u)


def build_saliency_mask_query(network: BaseNet, dummy_input: torch.Tensor, input: Optional[torch.Tensor], target: int) -> MarabouCore.InputQuery:
    marabou_net = network.build_marabou_net(dummy_input=dummy_input)
    ipq: MarabouCore.InputQuery = network.build_marabou_ipq()
    set_default_bound(ipq, range(marabou_net.numVars), -10, 10)
    set_default_bound(ipq, range(marabou_net.numVars,
                      marabou_net.numVars*2), -10, 10)
    # set input
    if input is not None:
        for vid, v in enumerate(marabou_net.inputVars[0]):
            ipq.setLowerBound(v, input[vid])
            ipq.setUpperBound(v, input[vid])

    # set grad
    for vid, v in enumerate(marabou_net.outputVars[0]):
        grad_v = v + marabou_net.numVars
        if vid == target:
            ipq.setLowerBound(grad_v, 1)
            ipq.setUpperBound(grad_v, 1)
        else:
            ipq.setLowerBound(grad_v, 0)
            ipq.setUpperBound(grad_v, 0)

    return ipq


def run_toy():
    def _write(o: Any, f: TextIOWrapper):
        print(o)
        if type(o) == torch.Tensor:
            f.write(np.array2string(o.detach().numpy())+"\n")
        else:
            f.write(o.str()+"\n")
    toy = ToyNet()
    input = torch.autograd.Variable(torch.Tensor(
        [-0.0005, 0.0008]), requires_grad=True)
    print("calculate using pytorch")
    z1 = toy.ln1(input)
    z1.retain_grad()
    h1 = toy.relu(z1)
    h1.retain_grad()
    z2 = toy.ln2(h1)
    z2.retain_grad()
    h2 = toy.relu(z2)
    h2.retain_grad()
    out = toy.ln3(h2)
    out.retain_grad()
    # out = h1
    # out = z2
    print("*", toy(input), out)
    assert torch.equal(toy(input), out), "networks are not the same"

    loss = out[0]
    loss.backward(retain_graph=True)
    print(input, input.grad)
    with open("true_values.txt", "w") as f:
        _write(z1, f)
        _write(z1.grad, f)
        _write(h1, f)
        _write(h1.grad, f)
        _write(z2, f)
        _write(z2.grad, f)
        _write(h2, f)
        _write(h2.grad, f)
        _write(out, f)
        _write(out.grad, f)
    ipq = build_saliency_mask_query(
        toy, dummy_input=torch.Tensor([0, 0]), input=input, target=0)
    MarabouCore.saveQuery(ipq, "finalQuery")


def run_acas():
    PATH = '/home/nle/workspace/VerifyNNE/datasets/ACAS/acas_nets/ACASXU_run2a_1_1_batch_2000.nnet'
    network = AcasXu(PATH)
    print(network.means)
    print("min", network.mins)
    print("max", network.maxs)
    dataset: Tuple[torch.Tensor] = torch.load(
        '/home/nle/workspace/VerifyNNE/datasets/ACAS/acas_nets/AcasNetID<1,1>-normed-train.pt')

    dataset_train = TensorDataset(dataset[0], dataset[1])
    print(dataset[0].shape, dataset[1].shape)
    print(dataset_train[0])

    sample_input, sample_output = dataset_train[0]

    print(network.forward(sample_input, verbose=True))
    print(network.global_bounds)
    saliency_map = network.compute_saliency_map(
        label=0, x=sample_input.unsqueeze(0))
    print("Saliency map:", saliency_map)
    ipq = build_saliency_mask_query(network, dummy_input=torch.Tensor(
        [0, 0, 0, 0, 0]), input=sample_input, target=sample_output.item())

    MarabouCore.saveQuery(ipq, "finalQuery")

    # fw_marabou_net.saveQuery("ToyNetForwardQuery")
run_toy()
