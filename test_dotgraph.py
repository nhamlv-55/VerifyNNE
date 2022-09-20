from typing import Dict, Optional
from BaseNet import BaseNet
from maraboupy import MarabouCore, Marabou, MarabouUtils
from maraboupy.MarabouNetwork import MarabouNetwork
import torch



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

        self.ln1 = torch.nn.Linear(2, 3, bias = False)
        self.ln2 = torch.nn.Linear(3, 2, bias = False)
        self.relu = torch.nn.ReLU()
        #fixed weight
        with torch.no_grad():
            self.ln1.weight = torch.nn.Parameter(torch.Tensor([[.1, .2], 
                                                                [-.111, .112], 
                                                                [-.121, .122]
                                                                ]))
            self.ln2.weight = torch.nn.Parameter(torch.Tensor([[.201, .202, .203],
                                                                [.211, .212, .213]]))
    def forward(self, x: torch.Tensor)->torch.Tensor:
        return self.ln2(self.relu(self.ln1(x)))
        # return self.relu(self.ln1(x))




toy = ToyNet()

print("calculate using pytorch")
input = torch.autograd.Variable(torch.Tensor([0.5, 0.8]), requires_grad=True)
z1 = toy.ln1(input); z1.retain_grad()
h1 = toy.relu(z1); h1.retain_grad()
out = toy.ln2(h1); out.retain_grad()
# out = h1
print("*", toy(input), out)
assert torch.equal(toy(input), out), "networks are not the same"

loss = out[0]
loss.backward(retain_graph = True)
print(input, input.grad)
print(z1, z1.grad)
print(h1, h1.grad)
print(out, out.grad)

#helper function
def set_default_bound(ipq: MarabouCore.InputQuery, n_vars: int):
    for v in range(n_vars):
        ipq.setLowerBound(v, -1.0)
        ipq.setUpperBound(v, 1.0)

def build_saliency_mask_query(network: BaseNet, input: Optional[torch.Tensor], target: int)->MarabouCore.InputQuery:
    marabou_net = network.build_marabou_net(dummy_input=torch.Tensor([1, -0.5]))
    ipq:MarabouCore.InputQuery = network.build_marabou_ipq(target=0)
    set_default_bound(ipq, marabou_net.numVars*2)
    #set input
    if input is not None:
        for vid, v in enumerate(marabou_net.inputVars[0]):
            ipq.setLowerBound(v, input[vid])
            ipq.setUpperBound(v, input[vid])

    #set grad
    for vid, v in enumerate(marabou_net.outputVars[0]):
        grad_v = v + marabou_net.numVars
        if vid==target:
            ipq.setLowerBound(grad_v, 1)
            ipq.setUpperBound(grad_v, 1)
        else:
            ipq.setLowerBound(grad_v, 0)
            ipq.setUpperBound(grad_v, 0)

    return ipq

ipq = build_saliency_mask_query(toy, input, 0)
MarabouCore.saveQuery(ipq, "finalQuery")

# fw_marabou_net.saveQuery("ToyNetForwardQuery")
