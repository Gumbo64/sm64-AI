import torch.nn as nn
import torch
import numpy as np
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

network = nn.Sequential(
            # 4 frame stack so that is the first number
    layer_init(nn.Conv2d(4, 128, 8, stride=2)),
    nn.MaxPool2d(kernel_size=4, stride=2),
    nn.LeakyReLU(),
    layer_init(nn.Conv2d(128, 64, 4, stride=2)),
    nn.LeakyReLU(),
    nn.Flatten(),

)
height = 128
width = 72
channels = 4
print(network(torch.zeros((1, channels, height, width))).shape)