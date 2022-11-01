from io import FileIO, TextIOWrapper
from typing import Any, Dict, Optional, Tuple, List

from matplotlib import pyplot as plt
from BaseNet import BaseNet
from maraboupy import MarabouCore, Marabou, MarabouUtils
from maraboupy.MarabouNetwork import MarabouNetwork
import torch
from torch.utils.data import TensorDataset, DataLoader
import torchvision.datasets as dset
import torchvision.transforms as transforms
import numpy as np
PATH = 'vnncomp2021/benchmarks/mnistfc/mnist-net_256x4.onnx'
root='datasets/MNIST/'
trans = transforms.Compose([transforms.ToTensor(), 
# transforms.Normalize((0.5,), (1.0,))
])

train_set = dset.MNIST(root=root, train=True, transform=trans, download=True)
test_set = dset.MNIST(root=root, train=False, transform=trans, download=True)

batch_size = 10000

train_loader:DataLoader = DataLoader(
                 dataset=train_set,
                 batch_size=batch_size,
                 shuffle=False)
test_loader:DataLoader = DataLoader(
                dataset=test_set,
                batch_size=batch_size,
                shuffle=False)
import onnx
from onnx2pytorch import ConvertModel

onnx_model = onnx.load(PATH)
pytorch_model = ConvertModel(onnx_model)
print(pytorch_model)


test_inputs, test_labels =  next(iter(test_loader))

# LABEL = 1
# all_image_of_label:List[torch.Tensor] = []
# for idx in range(batch_size):
    # if test_labels[idx]==LABEL:
        # all_image_of_label.append(test_inputs[idx])
# idx = np.random.randint(0, 10000)
idx = 8753
test_input = test_inputs[idx].squeeze()
test_label = test_labels[idx]
LABEL = test_label
print("test label", test_label, idx)
class MNISTNet(BaseNet):
    def __init__(self, network_path: str):
        super().__init__()
        self.network_path = network_path
        self.pytorch_net = ConvertModel(onnx.load(network_path))

    def build_marabou_net(self, dummy_input:torch.Tensor)->MarabouNetwork:
        print("Building Marabou network...")
        self.marabou_net = Marabou.read_onnx_plus(self.network_path)
        return self.marabou_net
    
    def forward(self, x:torch.Tensor)->torch.Tensor:
        return self.pytorch_net.forward(x)

    def compute_saliency_map(self, x: torch.Tensor, label: int)->torch.Tensor:
        """
            Given an input tensor of shape (n_inputs, ...), returns
            a saliency map tensor of the same size. 
        """
        print("label", label)
        n_inputs:int = x.shape[0]
        x = torch.autograd.Variable(x, requires_grad=True)
        saliency_map:List[torch.Tensor] = []
        assert n_inputs == 1
        for idx in range(n_inputs):
            logits = self.forward(x)
            logits[idx][label].backward()
            saliency_map.append(x.grad[idx])
        return torch.stack(saliency_map) 
#helper function
def set_default_bound(ipq: MarabouCore.InputQuery, range: List[int], l: float, u: float):
    for v in range:
        ipq.setLowerBound(v, l)
        ipq.setUpperBound(v, u)

def build_saliency_mask_query(network: BaseNet, dummy_input: torch.Tensor, input: Optional[torch.Tensor], target: int, high_intensity_nodes: List[int])->MarabouCore.InputQuery:
    marabou_net = network.build_marabou_net(dummy_input=dummy_input)
    ipq:MarabouCore.InputQuery = network.build_marabou_ipq()


    print(marabou_net.inputVars)
    set_default_bound(ipq, range(marabou_net.numVars), -10, 10)
    set_default_bound(ipq, range(marabou_net.numVars, marabou_net.numVars*2), -10, 10)
    #set input
    if input is not None:
        for vid, v in enumerate(marabou_net.inputVars[0][0]):
            ipq.setLowerBound(v[0], input[vid])
            ipq.setUpperBound(v[0], input[vid])

    #set grad
    for vid, v in enumerate(marabou_net.outputVars[0][0]):
        grad_v = v + marabou_net.numVars
        if vid==target:
            ipq.setLowerBound(grad_v, 1)
            ipq.setUpperBound(grad_v, 1)
        else:
            ipq.setLowerBound(grad_v, 0)
            ipq.setUpperBound(grad_v, 0)

    #add the saliency map constraints
    #the last node is used to extract the max
    add_grad_const = False
    if add_grad_const:
        max_node_idx = marabou_net.numVars*2
        ipq.setNumberOfVariables(marabou_net.numVars*2+1)
        set_default_bound(ipq, [max_node_idx], -10, 10)
        n_input_nodes = dummy_input.flatten().shape[0]
        low_intensity_nodes = set(range(n_input_nodes)) - set(high_intensity_nodes)
        #the last node is max of all low_intensity nodes
        MarabouCore.addMaxConstraint(ipq, low_intensity_nodes, max_node_idx)
        # ipq.addMaxConstraint(low_intensity_nodes, max_node_idx)
        #all the high intensity nodes must have values greater than that node
                # pos_condition = MarabouUtils.Equation(MarabouCore.Equation.LE)
                # pos_condition.addAddend(1, self.reluList[i][0]); 
                # pos_condition.setScalar(0)
        for v in high_intensity_nodes:
            constraint = MarabouUtils.Equation(MarabouCore.Equation.GE)
            constraint.addAddend(1, v)
            constraint.addAddend(-1, max_node_idx)
            constraint.setScalar(0)

            ipq.addEquation(constraint.toCoreEquation())

    return ipq

def normalize_sm(sm, k:int):
    """
    expect sm to be flatten already
    """
    top_k_idx = np.argpartition(sm, -k)[-k:].tolist()
    top_lowk_idx = np.argpartition(sm, k)[:k].tolist()
    print(top_k_idx)
    print(top_lowk_idx)
    new_sm = np.array([0]*28*28)
    for idx in top_k_idx:
        new_sm[idx]=1
    for idx in top_lowk_idx:
        new_sm[idx]=-1
    return new_sm, top_k_idx, top_lowk_idx

def main():
    PATH='/home/nle/workspace/VerifyNNE/datasets/MNIST/mnist-net_256x4.onnx'
    network = MNISTNet(PATH)
    fig, axs = plt.subplots(3)
    axs[0].imshow(test_input.squeeze().detach().numpy())
    saliency_map= network.compute_saliency_map(test_input.unsqueeze(dim=0), LABEL)
    final_sm, top_k_idx, top_lowk_idx = normalize_sm(saliency_map.reshape(28*28), k = 100)
    # axs[1].imshow(final_sm.reshape(28,28), cmap='gray')
    # axs[2].imshow(saliency_map.reshape(28,28), cmap='gray')
    # plt.show()

    ipq = build_saliency_mask_query(network=network, dummy_input=torch.Tensor([0]*28*28), input=test_input.reshape(28*28), target = test_label, high_intensity_nodes=top_k_idx)
    Marabou.saveQuery(ipq, "MNISTSaliencyQuery")

main()