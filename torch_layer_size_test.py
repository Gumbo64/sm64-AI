import torch.nn as nn
import torch
import numpy as np
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

network = nn.Sequential(
    # frame stack is the first number
    layer_init(nn.Linear(7680 + 512, 4096)),
    nn.LeakyReLU(),
    layer_init(nn.Linear(4096, 4096)),
    nn.LeakyReLU(),
    layer_init(nn.Linear(4096, 4096)),
    nn.LeakyReLU(),
    layer_init(nn.Linear(4096, 4096)),
    nn.LeakyReLU(),
    layer_init(nn.Linear(4096, 2048)),
    nn.LeakyReLU(),
)
height = 72
width = 128
channels = 1
print(network(torch.zeros((1, channels, height, width))).shape)