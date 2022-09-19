from typing import Dict
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

        self.ln1 = torch.nn.Linear(2, 1, bias = False)
        self.ln2 = torch.nn.Linear(3, 2, bias = False)
        self.relu = torch.nn.ReLU()
        #fixed weight
        with torch.no_grad():
            self.ln1.weight = torch.nn.Parameter(torch.Tensor([[.1, .2], 
                                                                # [-.111, .112], 
                                                                # [-.121, .122]
                                                                ]))
            # self.ln2.weight = torch.nn.Parameter(torch.Tensor([[.201, .202, .203],
                                                                # [.211, .212, .213]]))
    def forward(self, x: torch.Tensor)->torch.Tensor:
        # return self.ln2(self.relu(self.ln1(x)))
        return self.relu(self.ln1(x))




toy = ToyNet()

print("calculate using pytorch")
input = torch.autograd.Variable(torch.Tensor([0.5, -0.8]), requires_grad=True)
z1 = toy.ln1(input); z1.retain_grad()
h1 = toy.relu(z1); h1.retain_grad()
# out = toy.ln2(h1); out.retain_grad()
out = h1
print(toy(input), out)
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

marabou_net = toy.build_marabou_net(dummy_input=torch.Tensor([1, -0.5]))
ipq = marabou_net.getMarabouQuery()
# ipq.setLowerBound(0, 0.5)
# ipq.setLowerBound(1, 0.8)
# ipq.setUpperBound(0, 0.5)
# ipq.setUpperBound(1, 0.8)
# MarabouCore.saveQuery(ipq, "forwardQuery") 
# exitCode: str; vals:Dict[str, float]; stats: ... = MarabouCore.solve(ipq, Marabou.createOptions(verbosity = 2), "log")
# print(exitCode)
# print(vals)
# print(stats)

ipq = toy.build_marabou_ipq(target=0)
set_default_bound(ipq, 8)

ipq.setLowerBound(0, 0.5)
ipq.setLowerBound(1, 0.8)

# ipq.setLowerBound(17, 0.0)
# ipq.setLowerBound(18, 0.0)
# ipq.setLowerBound(19, 0.0)

ipq.setUpperBound(0, 0.5)
ipq.setUpperBound(1, 0.8)

ipq.setLowerBound(6, 1.0)
ipq.setUpperBound(6, 1.0)
# ipq.setUpperBound(19, 1.0)
# print(ipq.dump())

#grad constraints:
# c = MarabouUtils.Equation(MarabouCore.Equation.LE)
# c.addAddend(1, 5)
# c.addAddend(-1, 4)
# c.setScalar(0)
# ipq.addEquation(c.toCoreEquation())

MarabouCore.saveQuery(ipq, "finalQuery")
exitCode: str; vals:Dict[str, float]; stats: ... = MarabouCore.solve(ipq, Marabou.createOptions(verbosity = 2, snc=True), "log")
print(exitCode)
print(vals)
print(stats)

marabou_net.render_dot()
# fw_marabou_net.saveQuery("ToyNetForwardQuery")
