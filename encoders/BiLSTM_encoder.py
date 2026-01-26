import os
import numpy as np
from torch.utils.data import Dataset, Sampler
import torch
import torch.nn as nn
import torch.nn.functional as F


class VSL_BiLSTM_Encoder(nn.Module):
    def __init__(
        self,
        input_dim=201,
        hidden_dim=256,
        num_layers=2,
        emb_dim=256,
        dropout=0.2
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        self.proj = nn.Linear(hidden_dim * 2, emb_dim)

    def forward(self, x):
        """
        x: (B, T, 201)
        """
        # LSTM
        out, _ = self.lstm(x)     # (B, T, 2*hidden)

        # Temporal pooling (mean)
        out = out.mean(dim=1)     # (B, 2*hidden)

        # Projection
        emb = self.proj(out)      # (B, emb_dim)
        emb = F.normalize(emb, dim=1)

        return emb