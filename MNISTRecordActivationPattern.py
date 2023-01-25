# %%
#cifar 10 dataset
import torch
import torchvision
import torchvision.transforms as transforms
from tensorflow.keras import datasets, layers, models
import json
import os
import shutil
from sklearn.metrics import confusion_matrix, accuracy_score
import sys

whichset = sys.argv[1]
idx = int(sys.argv[2])
"""
Load CIFAR10 dataset
"""
transform = transforms.Compose(
    [transforms.ToTensor()])

batch_size = 4

(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()



# %%
"""
Load a Marabou net for MNIST
"""
from maraboupy import Marabou, MarabouCore, MarabouUtils
import numpy as np

PATH='datasets/MNIST/mnist-net_256x6.onnx'
marabou_network = Marabou.read_onnx(PATH)
check_loaded_model = False
if check_loaded_model:
    import onnx
    from onnx2pytorch import ConvertModel
    onnx_model = onnx.load(PATH)
    pytorch_model = ConvertModel(onnx_model)
    print(pytorch_model)




# %%
"""
Check accuracy of the loaded model. Expected: 74.16%
"""
def check_model():
    true_labels = []
    pred_labels = []
    for idx in range(len(test_images)):
        input = (test_images[idx]/255).reshape(1, 784, 1)
        # print(input, input.shape)
        pred = pytorch_model(torch.Tensor(input))
        # print(pred, np.argmax(pred), labels[idx])
        true_labels.append(test_labels[idx])
        if pred is not None:
            pred_labels.append(torch.argmax(pred))
        else:
            pred_labels.append(-1)
    print(confusion_matrix(true_labels, pred_labels))
    print(accuracy_score(true_labels, pred_labels))
if check_loaded_model:
    check_model()
# %%
"""
Setup filenames
"""
model_name = os.path.basename(PATH)
train_set = (whichset == "trainset")
if train_set:
    imageset = train_images
    labelset = train_labels
    prefix = "trainset_{}".format(model_name)
else:
    imageset = test_images
    labelset = test_labels
    prefix = "testset_{}".format(model_name)

if os.path.exists(prefix):
    pass
    # shutil.rmtree(prefix)
    # os.mkdir(prefix)
else:
    os.mkdir(prefix)

IMAGE_PATH = "{}/input_{}.npt".format(prefix, idx)
RELUDICT_PATH = "{}/reluDict_{}.json".format(prefix, idx)
RELUBITMAP_PATH = "{}/reluBitmap_{}.txt".format(prefix, idx)

input = (imageset[idx]/255).reshape(1, 784, 1)
true_label = labelset[idx]
_, outputDict = marabou_network.evaluate(input, filename="{}.log".format(idx))
pred_label = np.argmax(_)
print(pred_label)
print(true_label)


# %%
"""write the relu values out"""
print(len(outputDict))
print(len(marabou_network.reluList))
relu_dict = {"pred_label": int(pred_label), "true_label": int(true_label), "relu_dict": {}}
print(relu_dict)
for i, o in marabou_network.reluList:
    relu_dict["relu_dict"][str(i)] = outputDict[i]
res = {"outputDict": outputDict, "reluList": marabou_network.reluList}
print(res)
np.save(IMAGE_PATH, input)
with open(RELUDICT_PATH, "w") as f:
    json.dump(relu_dict, f, indent=2)
with open(RELUBITMAP_PATH, "w") as f:
    res_str = ""
    for i, o in marabou_network.reluList:
        if outputDict[i] <=0: res_str+="0"
        else: res_str+="1"
    f.write("{} {}\n".format(pred_label, true_label))
    f.write(res_str)



# %%
""" Check if the written files are correct """
with open(RELUDICT_PATH, "r") as f:
    loaded_relu_dict = json.load(f)["relu_dict"]
with open(RELUBITMAP_PATH, "r") as f:
    relu_map = f.readlines()[1].strip()

assert len(relu_map) == len(loaded_relu_dict.keys()), print(len(relu_map), "!=", len(loaded_relu_dict.keys()))
relu_keys = sorted([int(k) for k in loaded_relu_dict.keys()])
print(relu_keys)
for ridx, relu in enumerate(relu_keys):
    relu = str(relu)
    assert int(loaded_relu_dict[relu] >0) == int(relu_map[ridx]), print(relu, loaded_relu_dict[relu]>0, ridx, relu_map[ridx])


