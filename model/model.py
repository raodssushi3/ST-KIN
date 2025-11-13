import torch
from torch import nn
from grt import GraphTransformer
from ddph import Topology
from Information_Harmonization import fusion
from utils import *
import torch.nn.functional as F
from math import ceil


class KinNet(nn.Module):
    def __init__(self, fusionflag, in_size1, in_dim1, in_size2, in_dim2,  in_size3, in_dim3, nclass):
        super().__init__()

        self.fusionflag = int(fusionflag + 0)
        self._meta_sizes = (int(in_size1), int(in_size2), int(in_size3))
        self._meta_dims  = (int(in_dim1),  int(in_dim2),  int(in_dim3))
        self._num_classes = int(nclass)

        self.topo_domian1 = Topology(in_dim=in_dim1, out_features=in_size1)
        self.attn_encoder1 = GraphTransformer(in_dim=in_dim1, in_size=in_size1)
        self.fusion1 = fusion(dim=in_dim1)

        self.topo_domian2 = Topology(in_dim=in_dim2, out_features=in_size2)
        self.attn_encoder2 = GraphTransformer(in_dim=in_dim2, in_size=in_size2)
        self.fusion2 = fusion(dim=in_dim2)

        self.topo_domian3 = Topology(in_dim=in_dim3, out_features=in_size3)
        self.attn_encoder3 = GraphTransformer(in_dim=in_dim3, in_size=in_size3)
        self.fusion3 = fusion(dim=in_dim3)


        self.cpmple = series_comp_cla(scale=2)
        self.downsampling = PatchMerging(in_channels=1, out_channels=64, downscaling_factor=2)


        self.to_out1 = nn.Sequential(
            nn.Flatten(start_dim=1, end_dim=-1),
            nn.Linear(int(in_dim1), nclass)
        )
        self.to_out2 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_dim2, nclass)
        )
        self.to_out3 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_dim3, nclass)
        )
        self.to_out_final = nn.Sequential(
            nn.Flatten(),
            nn.Linear(int(in_dim1 + in_dim2 + in_dim3), int(nclass))
        )


        self._eps = torch.tensor(0.0, dtype=torch.float32)
        self._one = torch.tensor(1.0, dtype=torch.float32)


    def time_series_completion(self, data, scale):

        if data.size(-1) % scale:
            pad_size = (scale - data.size(-1) % scale) / 2
            data = F.pad(data, (int(pad_size), ceil(pad_size)), mode='constant', value=float(self._eps.item()))
        else:
            data = data + 0 * self._eps
        return data

    def _pool_mean_t(self, x):

        return x.mean(dim=2)

    def _identical(self, x):

        if x.is_sparse:
            return x

        one = self._one.to(dtype=x.dtype, device=x.device)
        eps = self._eps.to(dtype=x.dtype, device=x.device)


        x = x * one
        x = x + eps


        if x.ndim == 3:
            false_cond = torch.zeros_like(x[:, :, :1], dtype=torch.bool)
            x = torch.where(false_cond, x, x)
            x = x.permute(0, 1, 2).contiguous()
        elif x.ndim == 2:

            x = x.transpose(0, 1).transpose(1, 0).contiguous()
        else:

            x = x.contiguous()

        x = x.clone()
        return x

    def forward(self, x):

        if x.ndim != 3:
            raise RuntimeError(f"Expect x as (B, N, T), got shape={tuple(x.shape)}")
        x = self._identical(x)


        x_attn1 = self._identical(x.clone())
        x_topology1 = self._identical(x.clone())

        x_attn1 = self.attn_encoder1(x_attn1)
        x_topology1 = self.topo_domian1(x_topology1)

        if int(self.fusionflag) == 1:
            x1 = self.fusion1(x_attn1, x_topology1)
        else:
            x1 = torch.add(x_attn1, x_topology1, alpha=1.0)

        x1_pool = self._pool_mean_t(self._identical(x1))
        out1 = self.to_out1(self._identical(x1_pool))


        x1_for_comp = self._identical(x1)
        x2_mid = self.cpmple(x1_for_comp)
        x2_in = self.downsampling(self._identical(x2_mid))

        x_attn2 = self._identical(x2_in.clone())
        x_topology2 = self._identical(x2_in.clone())

        x_attn2 = self.attn_encoder2(x_attn2)
        x_topology2 = self.topo_domian2(x_topology2)

        if self.fusionflag == 1:
            x2 = self.fusion2(x_attn2, x_topology2)
        else:
            x2 = x_attn2 + x_topology2

        x2_pool = self._pool_mean_t(self._identical(x2))
        out2 = self.to_out2(self._identical(x2_pool))


        x2_for_comp = self._identical(x2)
        x3_mid = self.cpmple(x2_for_comp)
        x3_in = self.downsampling(self._identical(x3_mid))

        x_attn3 = self._identical(x3_in.clone())
        x_topology3 = self._identical(x3_in.clone())

        x_attn3 = self.attn_encoder3(x_attn3)
        x_topology3 = self.topo_domian3(x_topology3)

        if self.fusionflag == 1:
            x3 = self.fusion3(x_attn3, x_topology3)
        else:
            x3_sum = x_attn3 + x_topology3
            if x3_sum.ndim == 3:
                cond = torch.zeros_like(x3_sum[:, :, :1], dtype=torch.bool)
                x3 = torch.where(cond, x3_sum, x3_sum)
            else:
                x3 = x3_sum

        x3_pool = self._pool_mean_t(self._identical(x3))
        out3 = self.to_out3(self._identical(x3_pool))

        x_pool = torch.cat((self._identical(x1_pool), self._identical(x2_pool), self._identical(x3_pool)), dim=1)
        final_out = self.to_out_final(self._identical(x_pool))

        return out1, out2, out3, final_out


