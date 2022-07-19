from typing import Tuple, List
from BaseNet import AcasXu
import torch
from torch import Tensor
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import classification_report
PATH = '/home/nle/workspace/VerifyNNE/datasets/ACAS/acas_nets/ACASXU_run2a_1_1_batch_2000.nnet'
network = AcasXu(PATH)
print(network.means)
print("min", network.mins)
print("max", network.maxs)
dataset: Tuple[Tensor] = torch.load('/home/nle/workspace/VerifyNNE/datasets/ACAS/acas_nets/AcasNetID<1,1>-normed-train.pt')

dataset_train = TensorDataset(dataset[0], dataset[1])
print(dataset[0].shape, dataset[1].shape)
print(dataset_train)

loader = DataLoader(dataset_train, batch_size=2000, shuffle=True)

all_preds: List[int] = []
true_labels: List[int] = []
for inputs, outputs in loader:
    raw_pred = network(inputs)
    pred: List[int] = torch.argmin(raw_pred, dim = 1).squeeze().tolist()
    all_preds.extend(pred)
    true_labels.extend(outputs.squeeze().tolist())

print(classification_report(true_labels, all_preds))