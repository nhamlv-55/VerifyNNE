"""Examples of computing Jacobian bounds.
We use a small model with two convolutional layers and dense layers respectively.
The width of the model has been reduced for the demonstration here. And we use
data from CIFAR-10.
We show examples of:
- Computing Jacobian bounds
- Computing Linf local Lipschitz constants
- Computing JVP bounds
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm
from auto_LiRPA.utils import Flatten

torch.set_default_tensor_type(torch.DoubleTensor)

W1 = [[[[.101, .102],
        [.111, -.112]
       ]],
      [[[-.201, -.202],
        [.211, -.212]
      ]
      ]]
B1 = [0.01, -0.095]
W2 = [[.301, .302, -.303, .304, -.305, .306, .307, -.308],
      [.311, .312, .313, -.314, .315, .316, -.317, -.318]]
B2 = [0.007, -0.1]
W3 = [[.301, .302],
      [.311, .312]]
device = 'cpu'

class ToyConvNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(
            in_channels=1, out_channels=2, kernel_size=2)
        self.ln1 = torch.nn.Linear(8, 2, bias=True)
        self.relu = torch.nn.ReLU()
        # fixed weight
        with torch.no_grad():
            self.conv1.weight = torch.nn.Parameter(torch.Tensor(W1))
            self.conv1.bias = torch.nn.Parameter(
                torch.Tensor(B1))
            self.ln1.weight = torch.nn.Parameter(torch.Tensor(W2))
            self.ln1.bias = torch.nn.Parameter(torch.Tensor(B2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.relu(x)
        x = x.reshape(-1, 8)
        x = self.ln1(x)

        return x

toy = ToyConvNet()
input = torch.autograd.Variable(torch.Tensor(
    [[[[0.11, 0.12, 0.13],
        [0.21, 0.22, 0.23],
        [0.31, 0.32, 0.33],
    ]]]), requires_grad=True)
torch.manual_seed(0)

# Create a small model and load pre-trained parameters.
# model_ori = build_model(width=4, linear_size=32)
# model_ori.load_state_dict(torch.load('pretrained/cifar_2c2f.pth'))
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# model_ori = model_ori.to(device)
# print('Model:', model_ori)

# Prepare the dataset
test_data = datasets.CIFAR10('./data', train=False, download=True,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465], std=[0.2009, 0.2009, 0.2009])]))
x0 = test_data[0][0].unsqueeze(0).to(device)
print(x0.shape)

# Example 1: Convert the model for Jacobian bound computation
model = BoundedModule(toy, input, device=device)
model.augment_gradient_graph(input)

print(input.shape)
# Sanity check to ensure that the new graph matches the original gradient computation
y = toy(input)
ret_ori = torch.autograd.grad(y.sum(), input)[0].view(1, -1)
print("ret_ori", ret_ori)
# After running augment_gradient_graph, the model takes an additional input
# (the second input) which is a linear mapping applied on the output of the
# model before computing the gradient. It is the same as "grad_outputs" in
# torch.autograd.grad, which is "the 'vector' in the vector-Jacobian product".
# Here, setting torch.ones(1, 10) is equivalent to computing the gradients for
# y.sum() above.
print(input)
print(torch.ones(1,8).to(input))
ret_new = model(input, torch.ones(1, 2).to(input))
print("ret_new", ret_new)
assert torch.allclose(ret_ori, ret_new)

for eps in [0, 1./255, 4./255]:
    # The input region considered is an Linf ball with radius eps around x0.
    x = BoundedTensor(input, PerturbationLpNorm(norm=np.inf, eps=eps))
    # Compute the Linf locaal Lipscphitz constant
    lower, upper, imm_bounds = model.compute_jacobian_bounds(x, return_immediate_bounds=True)
    print("lower", lower)
    print("upper", upper)
    print("imm bounds", imm_bounds)
    print(f'Gap between upper and lower Jacobian bound for eps={eps:.5f}',
          (upper - lower).max())
    if eps == 0:
        assert torch.allclose(ret_new, lower.sum(dim=0, keepdim=True))
        assert torch.allclose(ret_new, upper.sum(dim=0, keepdim=True))