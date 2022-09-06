from typing import Dict
from BaseNet import BaseNet
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
            self.ln1.weight = torch.nn.Parameter(torch.Tensor([[.101, .102], 
                                                                [.111, .112], 
                                                                [.121, .122]]))
            self.ln2.weight = torch.nn.Parameter(torch.Tensor([[.201, .202, .203],
                                                                [.211, .212, .213]]))
    def forward(self, x: torch.Tensor)->torch.Tensor:
        return self.ln2(self.relu(self.ln1(x)))


toy = ToyNet()
fw_marabou_net = toy.build_marabou_forward_net(dummy_input=torch.Tensor([1, -0.5]))

# exitCode: str; vals:Dict[str, float]; stats: ... = marabou_net.solve()

fw_marabou_net.render_dot()
fw_marabou_net.saveQuery("ToyNetForwardQuery")

