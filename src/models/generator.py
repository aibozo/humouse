"""Conditional gesture generator scaffolding."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn

from .components import mlp, ResidualTCN


@dataclass
class GeneratorConfig:
    latent_dim: int = 64
    condition_dim: int = 16
    hidden_dim: int = 128
    target_len: int = 64
    num_layers: int = 3


class ConditionalGenerator(nn.Module):
    def __init__(self, config: GeneratorConfig):
        super().__init__()
        self.config = config
        input_dim = config.latent_dim + config.condition_dim
        self.fc = mlp(input_dim, config.hidden_dim * config.target_len)
        self.decoder = ResidualTCN(config.hidden_dim, config.hidden_dim, num_layers=config.num_layers)
        self.proj = nn.Conv1d(config.hidden_dim, 3, kernel_size=1)
        self.activation_xy = nn.Tanh()
        self.activation_dt = nn.Softplus()

    def forward(self, z: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        x = torch.cat([z, cond], dim=-1)
        batch_size = x.shape[0]
        hidden = self.fc(x).view(batch_size, self.config.target_len, self.config.hidden_dim)
        hidden = hidden.permute(0, 2, 1)  # (B, C, L)
        hidden = self.decoder(hidden)
        out = self.proj(hidden).permute(0, 2, 1)
        out_xy = self.activation_xy(out[..., :2])
        out_dt = self.activation_dt(out[..., 2:])
        return torch.cat([out_xy, out_dt], dim=-1)

    def sample(self, cond: torch.Tensor, num_samples: Optional[int] = None) -> torch.Tensor:
        if num_samples is None:
            num_samples = cond.size(0)
        z = torch.randn(num_samples, self.config.latent_dim, device=cond.device, dtype=cond.dtype)
        cond = cond[:num_samples]
        return self.forward(z, cond)

