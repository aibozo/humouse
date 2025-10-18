"""Discriminator / critic scaffolding."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn.utils import spectral_norm

from .components import ResidualTCN, mlp


@dataclass
class DiscriminatorConfig:
    input_dim: int = 3
    condition_dim: int = 16
    hidden_dim: int = 128
    num_layers: int = 4
    use_spectral_norm: bool = False


class GestureDiscriminator(nn.Module):
    def __init__(self, config: DiscriminatorConfig):
        super().__init__()
        self.config = config
        self.condition_mlp = mlp(
            config.condition_dim,
            config.hidden_dim,
            use_spectral_norm=config.use_spectral_norm,
        )
        self.encoder = ResidualTCN(
            config.input_dim,
            config.hidden_dim,
            num_layers=config.num_layers,
            use_spectral_norm=config.use_spectral_norm,
        )
        self.critic_head = mlp(
            config.hidden_dim,
            config.hidden_dim,
            1,
            use_spectral_norm=config.use_spectral_norm,
        )
        self.aux_head = mlp(
            config.hidden_dim,
            config.hidden_dim,
            4,
            use_spectral_norm=config.use_spectral_norm,
        )

    def forward(self, sequence: torch.Tensor, cond: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # sequence: (B, L, C)
        x = sequence.permute(0, 2, 1)
        encoded = self.encoder(x)
        pooled = encoded.mean(dim=-1)
        cond_embed = self.condition_mlp(cond)
        fused = pooled + cond_embed
        critic = self.critic_head(fused)
        aux = self.aux_head(fused)
        return critic.squeeze(-1), aux
