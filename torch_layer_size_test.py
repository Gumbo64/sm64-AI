import torch.nn as nn
import torch
import numpy as np
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

conv_net = nn.Sequential(
    # frame stack is the first number
    layer_init(nn.Conv2d(1, 256, 8, stride=2)),
    nn.MaxPool2d(kernel_size=4, stride=2),
    nn.LeakyReLU(),
    layer_init(nn.Conv2d(256, 128, 4, stride=2)),
    nn.LeakyReLU(),
    layer_init(nn.Conv2d(128, 64, 2, stride=1)),
    nn.LeakyReLU(),
    layer_init(nn.Conv2d(64, 32, 2, stride=1)),
    nn.LeakyReLU(),
    layer_init(nn.Conv2d(32, 16, 2, stride=1)),
    nn.LeakyReLU(),
    nn.Flatten(),
)




height = 72
width = 128
channels = 1

print(conv_net(torch.zeros((1, channels, height, width))).shape)

