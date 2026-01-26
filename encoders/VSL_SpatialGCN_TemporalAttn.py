import os
import numpy as np
from torch.utils.data import Dataset, Sampler
import torch
import torch.nn as nn
import torch.nn.functional as F


def build_adjacency(num_nodes, self_loop=True):
    """
    Simple fully-connected adjacency (baseline).
    Có thể thay bằng skeleton MediaPipe sau.
    """
    A = torch.ones(num_nodes, num_nodes)
    if self_loop:
        A = A + torch.eye(num_nodes)
    A = A / A.sum(dim=1, keepdim=True)
    return A


class SpatialGCN(nn.Module):
    def __init__(self, in_channels, hidden_dim, A):
        super().__init__()
        self.register_buffer("A", A)
        self.fc1 = nn.Linear(in_channels, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        """
        x: (B, T, V, C)
        """
        B, T, V, C = x.shape
        x = x.view(B * T, V, C)

        # Graph propagation
        x = torch.einsum("vw,bwc->bvc", self.A, x)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        x = x.view(B, T, V, -1)
        return x


class TemporalEncoder(nn.Module):
    def __init__(self, dim, num_heads=4, num_layers=2, dropout=0.1):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers)

    def forward(self, x):
        """
        x: (B, T, D)
        """
        return self.encoder(x)


class AttnPooling(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attn = nn.Linear(dim, 1)

    def forward(self, x):
        """
        x: (B, T, D)
        """
        w = self.attn(x).squeeze(-1)   # (B, T)
        w = torch.softmax(w, dim=1)
        x = (x * w.unsqueeze(-1)).sum(dim=1)
        return x


class VSL_SpatialGCN_TemporalAttn(nn.Module):
    def __init__(
        self,
        num_keypoints=67,
        in_channels=3,
        hidden_dim=256,
        emb_dim=256,
        num_heads=4,
        num_layers=2
    ):
        super().__init__()

        A = build_adjacency(num_keypoints)

        self.spatial = SpatialGCN(
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            A=A
        )

        self.temporal = TemporalEncoder(
            dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers
        )

        self.pool = AttnPooling(hidden_dim)
        self.proj = nn.Linear(hidden_dim, emb_dim)

    def forward(self, x):
        """
        x: (B, T, V*C)  -> (B, T, V, C)
        """
        B, T, D = x.shape
        x = x.view(B, T, -1, 3)

        # Spatial GCN
        x = self.spatial(x)           # (B, T, V, hidden)
        x = x.mean(dim=2)             # (B, T, hidden)

        # Temporal modeling
        x = self.temporal(x)          # (B, T, hidden)

        # Attention pooling
        x = self.pool(x)              # (B, hidden)

        # Projection + normalize
        x = self.proj(x)
        x = F.normalize(x, dim=1)

        return x