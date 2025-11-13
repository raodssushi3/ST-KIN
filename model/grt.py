# -*- coding: utf-8 -*-
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange, repeat
from mixprop_cy import mixprop_forward
# ---- env / misc ----
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
__all__ = ["GraphTransformer"]


def _dev():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _with_eye(mat):
    n = mat.size(0)
    return mat + torch.eye(n, device=mat.device)


def _row_norm(mat):
    s = mat.sum(1)
    return mat / s.view(-1, 1)


def _einsum_gate(x, a, sig="cwl,vw->cvl"):
    return torch.einsum(sig, (x, a)).contiguous()

def _seed_like(x):
    return torch.rand_like(x, device=x.device) * 1e-2

def _xavier_(m: nn.Module):
    for mod in m.modules():
        if isinstance(mod, nn.Linear):
            nn.init.xavier_normal_(mod.weight)
        elif isinstance(mod, nn.BatchNorm1d):
            nn.init.constant_(mod.weight, 1.0)
            nn.init.constant_(mod.bias, 0.0)


class _Pointwise(nn.Module):
    # 1x1 conv
    def __init__(self, c_in, c_out):
        super().__init__()
        self.proj = nn.Conv1d(c_in, c_out, kernel_size=1, bias=True)

    def forward(self, x):
        return self.proj(x)

class _NConv(nn.Module):
    def forward(self, x, a):
        return _einsum_gate(x, a)



class _MixProp(nn.Module):
    def __init__(self, c_in, c_out, gdep, alpha):
        super().__init__()
        self._conv = _NConv()
        self._head = _Pointwise(c_in, c_out)
        self._steps = int(gdep)
        self._alpha = float(alpha)

    def forward(self, x, adj):
        #
        adj = _with_eye(adj)
        P = _row_norm(adj)
        base = x
        carry = [base]
        for _ in range(self._steps):
            base = self._alpha * x + (1.0 - self._alpha) * self._conv(base, P)
            carry.append(base)
        z = torch.cat(carry, dim=1)

        return self._head(z)


class _AdjacencyLearner(nn.Module):
    def __init__(self, n, k):
        super().__init__()
        self.n = int(n)
        self.k = int(k)
        self._e1 = nn.Embedding(n, n)
        self._e2 = nn.Embedding(n, n)
        self._l1 = nn.Linear(n, n, bias=True)
        self._l2 = nn.Linear(n, n, bias=True)

    @torch.no_grad()
    def _topk_mask(self, A):
        M = torch.zeros(self.n, self.n, device=A.device)

        s, t = (A + _seed_like(A)).topk(self.k, dim=1)
        M.scatter_(1, t, torch.ones_like(s))
        return M

    def forward(self, x):
        dev = x.device
        idx = torch.arange(x.size(1), device=dev)
        v1 = torch.tanh(self._l1(self._e1(idx)))
        v2 = torch.tanh(self._l2(self._e2(idx)))
        A = torch.sigmoid(v1 @ v2.T)
        M = self._topk_mask(A)
        return A * M


class _GraphStage(nn.Module):
    def __init__(self, n_nodes, k_ratio, gdep, alpha):
        super().__init__()
        k = max(1, int(k_ratio * n_nodes))
        self._A = _AdjacencyLearner(n_nodes, k)
        self._P = _MixProp(n_nodes * (gdep + 1), n_nodes, gdep, alpha)

    def forward(self, x):

        A = self._A(x)
        z = self._P(x, A)
        return z, A


class _MultiHead(nn.Module):
    def __init__(self, d_model, dim_head, heads, dropout):
        super().__init__()
        inner = heads * dim_head
        self.h = heads
        self.sc = dim_head ** -0.5

        self._q = nn.Linear(d_model, inner, bias=True)
        self._k = nn.Linear(d_model, inner, bias=True)
        self._v = nn.Linear(d_model, inner, bias=True)

        self.proj = nn.Sequential(
            nn.Linear(inner, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, q, k, v):
        q, k, v = self._q(q), self._k(k), self._v(v)
        spl = lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.h)
        q, k, v = map(spl, (q, k, v))

        scores = (q @ k.transpose(-1, -2)) * self.sc
        raw = scores.clone()
        attn = F.softmax(scores, dim=-1)
        out = attn @ v
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.proj(out), raw


class _HybridBlock(nn.Module):
    def __init__(self, in_dim, factor, d_model, dim_head, heads):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.route = nn.Parameter(torch.randn(factor, d_model))

        self.down = nn.Linear(d_model, in_dim)
        self.up = nn.Linear(in_dim, d_model)

        self.send = _MultiHead(d_model, dim_head, heads, dropout=0.2)
        self.recv = _MultiHead(d_model, dim_head, heads, dropout=0.2)

        self.struct = _GraphStage(
            n_nodes=in_dim,
            k_ratio=0.7,
            gdep=2,
            alpha=0.05
        )

    def forward(self, x):
        B = x.size(0)
        R = repeat(self.route, "f d -> b f d", b=B)

        buf, score_send = self.send(R, x, x)

        buf = self.down(buf).transpose(-1, -2)
        buf, adj = self.struct(buf)
        buf = self.up(buf.transpose(-1, -2))

        y, score_recv = self.recv(x, buf, buf)
        y = self.norm(x + y)
        return y


class _Embedding(nn.Module):
    def __init__(self, d_feature, d_timestep, d_model, wise="timestep"):
        super().__init__()
        if wise not in ("timestep", "feature"):
            raise ValueError("Embedding wise error!")
        self._wise = wise
        self._proj = nn.Linear(d_feature if wise == "timestep" else d_timestep, d_model)

    def forward(self, x):
        if self._wise == "feature":
            x = x.transpose(-1, -2)
        return self._proj(x)


class _Positional(nn.Module):

    def __init__(self, in_size, d_model):
        super().__init__()
        self.pe = nn.Embedding(in_size, d_model)

    def forward(self, x):
        b = x.size(0)
        return self.pe.weight.unsqueeze(0).repeat(b, 1, 1)


class GraphTransformer(nn.Module):
    def __init__(self, in_dim, in_size, d_model=32, depth=2, dropout=0.5):
        super().__init__()

        self._embed_t = _Embedding(
            d_feature=in_dim,
            d_timestep=in_size,
            d_model=d_model,
            wise="timestep"
        )
        self._pos = _Positional(in_size, d_model)

        self._layers = nn.ModuleList()
        self._layers.append(
            _HybridBlock(
                in_dim=in_dim,
                factor=40,
                d_model=d_model,
                dim_head=d_model,
                heads=8
            )
        )

        self._head = nn.Sequential(
            nn.Linear(d_model, in_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )

        self.alpha = nn.Parameter(torch.ones(1))
        _xavier_(self)

    def forward(self, x):
        #
        z = x.transpose(1, 2)
        z = self._embed_t(z)
        z = z + self._pos(z)

        z = self._layers[0](z)

        z = self._head(z)
        z = z.transpose(1, 2)
        return z



