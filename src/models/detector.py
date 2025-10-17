"""Gesture detector scaffolding."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .components import ResidualTCN, mlp


@dataclass
class DetectorConfig:
    feature_dim: int = 32
    sequence_dim: int = 3
    hidden_dim: int = 128
    tcn_layers: int = 3


class FeatureBranch(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int):
        super().__init__()
        self.net = mlp(feature_dim, hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SequenceBranch(nn.Module):
    def __init__(self, sequence_dim: int, hidden_dim: int, tcn_layers: int):
        super().__init__()
        self.encoder = ResidualTCN(sequence_dim, hidden_dim, num_layers=tcn_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        x = sequence.permute(0, 2, 1)
        encoded = self.encoder(x)
        pooled = self.pool(encoded).squeeze(-1)
        return pooled


class GestureDetector(nn.Module):
    def __init__(self, config: DetectorConfig):
        super().__init__()
        self.config = config
        self.feature_branch = FeatureBranch(config.feature_dim, config.hidden_dim)
        self.sequence_branch = SequenceBranch(config.sequence_dim, config.hidden_dim, config.tcn_layers)
        self.head = mlp(config.hidden_dim * 2, config.hidden_dim, config.hidden_dim // 2, 1)

    def forward(self, sequence: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        feat_embed = self.feature_branch(features)
        seq_embed = self.sequence_branch(sequence)
        fused = torch.cat([feat_embed, seq_embed], dim=-1)
        logits = self.head(fused)
        return logits.squeeze(-1)

