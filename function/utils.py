import torch
import torch.utils.data
from torch.utils.data import TensorDataset
from einops import rearrange
from torch import nn, einsum
import torch.nn.functional as F
from math import ceil


class PatchMerging(nn.Module):
    def __init__(self, in_channels=1, out_channels=64, downscaling_factor=4):
        super(PatchMerging, self).__init__()
        self.downscaling_factor = downscaling_factor
        self.linear = nn.Linear(in_channels * downscaling_factor, out_channels)

    def forward(self, x):
        x = x.transpose(2, 1)
        x = rearrange(x, 'b (p f) c -> b f (p c)', p=self.downscaling_factor)
        # x = self.linear(x)
        x = x.transpose(2, 1)
        return x




