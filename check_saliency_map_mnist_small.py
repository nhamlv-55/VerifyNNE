from onnx import numpy_helper
from onnx2pytorch import ConvertModel
import onnx
from typing import Any, Dict, Optional, Tuple, List
import time
from matplotlib import pyplot as plt
from BaseNet import BaseNet
from maraboupy import MarabouCore, Marabou, MarabouUtils
from maraboupy.MarabouNetwork import MarabouNetwork
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import torchvision.datasets as dset
import torchvision.transforms as transforms
import numpy as np
from utils import set_default_bound, _write, normalize_sm
root = 'datasets/MNIST/'
trans = transforms.Compose([transforms.ToTensor(),
                            # transforms.Normalize((0.5,), (1.0,))
                            ])
train_new_network: bool = False
load_vnn: bool = False
plot: bool = True


train_set = dset.MNIST(root=root, train=True, transform=trans, download=True)
test_set = dset.MNIST(root=root, train=False, transform=trans, download=True)

batch_size = 10000

train_loader: DataLoader = DataLoader(
    dataset=train_set,
    batch_size=batch_size,
    shuffle=False)
test_loader: DataLoader = DataLoader(
    dataset=test_set,
    batch_size=batch_size,
    shuffle=False)


test_inputs, test_labels = next(iter(test_loader))

# LABEL = 1
# all_image_of_label:List[torch.Tensor] = []
# for idx in range(batch_size):
# if test_labels[idx]==LABEL:
# all_image_of_label.append(test_inputs[idx])
# idx = np.random.randint(0, 10000)
idx = 8753
test_input = test_inputs[idx].squeeze()
test_label = test_labels[idx]
LABEL = 1
print("LABEL", LABEL, idx)

_model = onnx.load('datasets/MNIST/mnist-net_256x2.onnx')
INTIALIZERS = _model.graph.initializer
Weight = []
for initializer in INTIALIZERS:
    W = numpy_helper.to_array(initializer)
    Weight.append(W)
for w in Weight:
    print(w.shape)


class MNISTNet(BaseNet):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(28*28, 256)
        self.dropout = torch.nn.Dropout()
        self.fc2 = torch.nn.Linear(256, 256)
        self.fc3 = torch.nn.Linear(256, 10)
        self.relu = torch.nn.ReLU()

    def load_vnn(self) -> None:
        with torch.no_grad():
            assert (len(Weight) == 6)
            self.fc1.weight = torch.nn.Parameter(torch.Tensor(Weight[1]))
            self.fc1.bias = torch.nn.Parameter(torch.Tensor(Weight[0]))

            self.fc2.weight = torch.nn.Parameter(torch.Tensor(Weight[3]))
            self.fc2.bias = torch.nn.Parameter(torch.Tensor(Weight[2]))

            self.fc3.weight = torch.nn.Parameter(torch.Tensor(Weight[5]))
            self.fc3.bias = torch.nn.Parameter(torch.Tensor(Weight[4]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(-1, 28*28)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.fc3(x)
        return x

    def compute_saliency_map(self, x: torch.Tensor, label: int) -> torch.Tensor:
        """
            Given an input tensor of shape (n_inputs, ...), returns
            a saliency map tensor of the same size. 
        """
        self.eval()
        print("label", label)
        n_inputs: int = x.shape[0]
        x = torch.autograd.Variable(x, requires_grad=True)
        saliency_map: List[torch.Tensor] = []
        assert n_inputs == 1
        for idx in range(n_inputs):
            logits = self.forward(x)
            logits[idx][label].backward()
            saliency_map.append(x.grad[idx])
        return torch.stack(saliency_map)


"""
Init
"""
network = MNISTNet()
optimizer = torch.optim.Adam(network.parameters(), lr=0.001)
train_losses = []
train_counter = []
test_losses = []
log_interval = 10
n_epochs = 10
test_counter = [i*len(train_loader.dataset) for i in range(n_epochs + 1)]
"""
Testing
"""


def test():
    network.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            print("ldt", len(data))
            output = network(data)
            test_loss += F.cross_entropy(output, target).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


"""
Training
"""


def train(epoch: int):
    network.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = network(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        print(loss)
        optimizer.step()


def build_saliency_mask_query(network: BaseNet, dummy_input: torch.Tensor, input: Optional[torch.Tensor], 
                            target: int, high_intensity_nodes: List[int], low_intensity_nodes: List[int]) -> MarabouCore.InputQuery:

    marabou_net = network.build_marabou_net(dummy_input=dummy_input)
    ipq: MarabouCore.InputQuery = network.build_marabou_ipq()  # type: ignore

    """
    Extracting just the forward query. For debugging only
    """
    set_default_bound(network.fw_ipq, range(marabou_net.numVars), -5, 5)
    # set input for the forward query
    print("input vars", marabou_net.inputVars)
    if input is not None:
        for vid, v in enumerate(marabou_net.inputVars[0]):
            network.fw_ipq.setLowerBound(v, input[vid])
            network.fw_ipq.setUpperBound(v, input[vid])
    Marabou.saveQuery(network.fw_ipq, "MNISTSaliencyQueryForwardSmall")

    set_default_bound(ipq, range(marabou_net.numVars,
                      marabou_net.numVars*2), -10, 10)
    # set input
    if input is not None:
        for vid, v in enumerate(marabou_net.inputVars[0]):
            ipq.setLowerBound(v, input[vid])
            ipq.setUpperBound(v, input[vid])
    # set grad
    print("output vars", marabou_net.outputVars)
    for vid, v in enumerate(marabou_net.outputVars[0][0]):
        grad_v: int = v + marabou_net.numVars
        if vid == target:
            print("setting v {} to 1".format(grad_v))
            ipq.setLowerBound(grad_v, 1)
            ipq.setUpperBound(grad_v, 1)
        else:
            ipq.setLowerBound(grad_v, 0)
            ipq.setUpperBound(grad_v, 0)

    # add the saliency map constraints
    # the last node is used to extract the max
    add_grad_const = True
    if add_grad_const:
        max_node_idx = marabou_net.numVars*2
        ipq.setNumberOfVariables(marabou_net.numVars*2+1)
        set_default_bound(ipq, [max_node_idx], -1, 1)
        n_input_nodes = dummy_input.flatten().shape[0]
        if len(low_intensity_nodes)==0:
            low_intensity_nodes = set(range(n_input_nodes)) - set(high_intensity_nodes)
        print(low_intensity_nodes)

        # create a dummy node (the last node)
        # the last node is greater than all low_intensity_nodes
        # and the last node is smaller than all high_intensity_nodes
        for v in low_intensity_nodes:
            assert v <= n_input_nodes, "low intensity node is not in the set of input nodes"
            constraint = MarabouUtils.Equation(MarabouCore.Equation.LE)
            constraint.addAddend(1, v+marabou_net.numVars)
            constraint.addAddend(-1, max_node_idx)
            constraint.setScalar(0)

            print("low intensity node const:", constraint)
            ipq.addEquation(constraint.toCoreEquation())

        for v in high_intensity_nodes:
            assert v <= n_input_nodes, "high intensity is not in the set of input nodes"
            constraint = MarabouUtils.Equation(MarabouCore.Equation.GE)
            constraint.addAddend(1, v+marabou_net.numVars)
            constraint.addAddend(-1, max_node_idx)
            constraint.setScalar(0)

            print("high intensity node const:", constraint)
            ipq.addEquation(constraint.toCoreEquation())

    print(high_intensity_nodes)

    return ipq


def main():
    if train_new_network:
        for epoch in range(1, n_epochs + 1):
            train(epoch)
            test()

        torch.save(network.state_dict(), 'model{}.pth'.format(time.time()))
    else:
        if load_vnn:
            network.load_vnn()
        else:
            network.load_state_dict(torch.load("model1673967689.4435592.pth"))
            network.eval()
    zero_input: torch.autograd.Variable = torch.autograd.Variable(
        torch.Tensor([[0]*28*28]), requires_grad=True)

    INPUT = zero_input
    # input = torch.autograd.Variable(test_input.reshape(1, 28*28), requires_grad = True)

    print(network)
    out = network(INPUT)
    loss = out[0][LABEL]
    loss.backward(retain_graph=True)
    with open("MNIST_saliency_check_true_values_small.txt", "w") as f:
        _write(INPUT, f)
        _write(INPUT.grad, f)
        _write(out, f)

    saliency_map = network.compute_saliency_map(
        INPUT.unsqueeze(dim=0), LABEL)
    with open("saliency_map", "w") as f:
        _write(saliency_map, f)
    final_sm, top_k_idx, top_lowk_idx = normalize_sm(
        saliency_map.reshape(28*28), k=10)
    if plot:
        fig, axs = plt.subplots(3)
        axs[0].imshow(INPUT.detach().numpy().reshape(28, 28))
        axs[1].imshow(final_sm.reshape(28, 28), cmap='gray')
        axs[2].imshow(saliency_map.reshape(28, 28), cmap='gray')
        plt.savefig('plot.pdf')

    ipq = build_saliency_mask_query(network=network,
                                    dummy_input=torch.Tensor([0]*28*28),
                                    input=INPUT.reshape(28*28),
                                    target=LABEL,
                                    high_intensity_nodes=top_k_idx,
                                    low_intensity_nodes=top_lowk_idx)
    Marabou.saveQuery(ipq, "MNISTSaliencyQuerySmall")


main()
