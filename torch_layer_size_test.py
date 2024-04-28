import torch.nn as nn
import torch
import numpy as np
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

network = nn.Sequential(
    # 4 frame stack so that is the first number

    layer_init(nn.Conv2d(1, 256, 8, stride=2)),
    nn.MaxPool2d(kernel_size=4, stride=2),
    nn.LeakyReLU(),
    layer_init(nn.Conv2d(256, 128, 4, stride=2)),
    nn.LeakyReLU(),
    layer_init(nn.Conv2d(128, 128, 2, stride=1)),
    nn.Flatten(),



)
height = 84
width = 84
channels = 1
print(network(torch.zeros((1, channels, height, width))).shape)