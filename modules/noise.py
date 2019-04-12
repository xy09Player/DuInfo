# coding = utf-9
# author = xy


import torch
from torch import nn


class GaussianNoise(nn.Module):
    def __init__(self, stddev=0.1):
        super(GaussianNoise, self).__init__()

        self.stddev = stddev

    def forward(self, x):
        noise = torch.empty_like(x)
        noise.normal_(0, self.stddev)
        return x + noise
