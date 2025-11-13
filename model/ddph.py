
import torch
import torch.nn as nn
import numpy as np
import gt_betti_ext
from gtda.time_series import TakensEmbedding
from gtda.homology import VietorisRipsPersistence
from gtda.diagrams import BettiCurve


class Topology(nn.Module):
    def __init__(self, in_dim, out_features, time_delay=1, dimension=5,
                 homology_dims=(0, 1), n_bins=100, hidden_ratio=0.5):
        super().__init__()
        self.TE = TakensEmbedding(time_delay=time_delay, dimension=dimension, flatten=True)
        self.VR = VietorisRipsPersistence(homology_dimensions=homology_dims, collapse_edges=True)
        self.BC = BettiCurve(n_bins=n_bins)

        self.var_scorer = None
        self.proj = None

        self.in_dim = in_dim
        self.out_features = out_features
        self.n_bins = n_bins
        self.H_dim = len(homology_dims)
        self.hidden_ratio = hidden_ratio

    def _build_heads_if_needed(self, feat_dim, device):
        if self.var_scorer is None:
            hidden = max(8, int(feat_dim * self.hidden_ratio))
            self.var_scorer = nn.Sequential(
                nn.Linear(feat_dim, hidden),
                nn.ReLU(inplace=True),
                nn.Linear(hidden, 1)
            ).to(device)
        if self.proj is None:
            self.proj = nn.Linear(feat_dim, self.out_features).to(device)

    @torch.no_grad()
    def _extract_betti_per_var(self, x_np):

        features_np = gt_betti_ext.extract_betti_per_var(x_np, self.TE, self.VR, self.BC)
        return features_np

    def forward(self, x):
        device = x.device
        x_np = x.detach().cpu().numpy()

        features_np = self._extract_betti_per_var(x_np)
        features = torch.from_numpy(features_np).float().to(device)
        B, N, feat_dim = features.shape

        self._build_heads_if_needed(feat_dim, device)

        scores = self.var_scorer(features)
        var_weights = torch.softmax(scores.squeeze(-1), dim=1)

        fused_topo = torch.sum(features * var_weights.unsqueeze(-1), dim=1)

        y = self.proj(fused_topo)
        y = var_weights.unsqueeze(-1) * y.unsqueeze(1)
        return y



