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


def series_comp_fuc(data,scale):
    if data.size(-1) % scale:
        pad_size = (scale - data.size(-1) % scale) / 2
        data = F.pad(data, (int(pad_size), ceil(pad_size)), mode='constant', value=0.0)
    else:
        data = data
    return data



class series_comp_cla(nn.Module):
    def __init__(self, scale):
        super(series_comp_cla, self).__init__()
        self.scale = scale

    def forward(self,x):
        if x.size(-1) % self.scale:
            pad_size = (self.scale - x.size(-1) % self.scale) / 2
            x = F.pad(x, (int(pad_size), ceil(pad_size)), mode='constant', value=0.0)
        else:
            x = x
        return x