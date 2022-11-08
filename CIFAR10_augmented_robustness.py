import torch
import torchvision
import torchvision.transforms as transforms
from tensorflow.keras import datasets, layers, models
import json
import os
import shutil
from sklearn.metrics import confusion_matrix, accuracy_score
import sys
from maraboupy import Marabou, MarabouCore, MarabouUtils
import numpy as np

MAX_TIME = 600
M_OPTIONS: MarabouCore.Options = Marabou.createOptions(verbosity=0,
                                                       initialSplits=2,
                                                       timeoutInSeconds=MAX_TIME,
                                                       snc=True,
                                                       numWorkers=6,
                                                       )


#cifar 10 dataset
whichset = sys.argv[1]
idx = int(sys.argv[2])
"""
Load CIFAR10 dataset
"""
transform = transforms.Compose(
    [transforms.ToTensor()])

batch_size = 4
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# %%
"""
Load a Marabou net for CIFAR 10
"""
PATH='datasets/CIFAR10/marabou-cifar10/nets/cifar10_small.onnx'
marabou_network = Marabou.read_onnx(PATH)

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

IMAGE_PATH = "{}/input_{}.npt.npy".format(prefix, idx)
RELUBITMAP_PATH = "{}/reluBitmap_{}.txt".format(prefix, idx)
with open(RELUBITMAP_PATH, "r") as f:
    raw = f.readlines()
bitmap = raw[1].strip()
assert len(marabou_network.reluList) == len(bitmap)

input = np.load(IMAGE_PATH).flatten()
print(input[:10])
output = marabou_network.outputVars[0][0]
print(output)
#set range for input
for i in range(len(input)):
    marabou_network.setLowerBound(i, max(0, input[i]-0.012))
    marabou_network.setUpperBound(i, min(1, input[i]+0.012))
#add output constraint
constraint = MarabouUtils.Equation(MarabouCore.Equation.GE)
constraint.addAddend(1, output[1]) #adv label
constraint.addAddend(-1, output[0]) #true label
constraint.setScalar(0)
marabou_network.addEquation(constraint)
#set relu constraints
#for any relu that is fixed, we need to remove it from the reluList.
new_relu_list = []
for idx, relu_inout in enumerate(marabou_network.reluList):
    relu_in, relu_out = relu_inout
    if bitmap[idx]=="1": #positive relu
        #input of relu must be >=0
        marabou_network.setLowerBound(relu_in, 0)
        #output of relu must be relu_in
        positive_phase_con = MarabouUtils.Equation(MarabouCore.Equation.EQ)
        positive_phase_con.addAddend(1, relu_in)
        positive_phase_con.addAddend(-1, relu_out)
        positive_phase_con.setScalar(0)
        marabou_network.addEquation(positive_phase_con)

    elif bitmap[idx]=="0": #negative relu
        #output of relu must be 0
        marabou_network.setLowerBound(relu_out, 0)
        marabou_network.setUpperBound(relu_out, 0)

        #input of Relu must be <=0
        marabou_network.setUpperBound(relu_in, 0)
    else:
        new_relu_list.append([relu_in, relu_out])
#update the relu list to be the one that is not fixed
marabou_network.reluList = new_relu_list
#hmmm
marabou_network.saveQuery("testQuery")
#start solving
print("start solving")
exit_code, vals, stats = marabou_network.solve(options=M_OPTIONS)
print(exit_code)
print(json.dumps(vals, indent=2))
exit(0)
