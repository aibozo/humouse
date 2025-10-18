"""Conditional gesture generator scaffolding."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn

from .components import mlp, ResidualTCN


def _build_activation(name: str | None) -> nn.Module:
    if name is None:
        return nn.Identity()
    lowered = name.lower()
    if lowered in {"none", "identity", "linear"}:
        return nn.Identity()
    if lowered == "tanh":
        return nn.Tanh()
    if lowered == "sigmoid":
        return nn.Sigmoid()
    if lowered == "softplus":
        return nn.Softplus()
    raise ValueError(f"Unsupported activation '{name}' for generator output")


@dataclass
class GeneratorConfig:
    latent_dim: int = 64
    condition_dim: int = 16
    hidden_dim: int = 128
    target_len: int = 64
    num_layers: int = 3
    use_spectral_norm: bool = False
    output_dim: int = 3
    activation_xy: str = "tanh"
    activation_dt: str = "softplus"


class ConditionalGenerator(nn.Module):
    def __init__(self, config: GeneratorConfig):
        super().__init__()
        self.config = config
        self.xy_dims = min(2, config.output_dim)
        self.has_dt = config.output_dim > self.xy_dims
        input_dim = config.latent_dim + config.condition_dim
        self.fc = mlp(
            input_dim,
            config.hidden_dim * config.target_len,
            use_spectral_norm=config.use_spectral_norm,
        )
        self.decoder = ResidualTCN(
            config.hidden_dim,
            config.hidden_dim,
            num_layers=config.num_layers,
            use_spectral_norm=config.use_spectral_norm,
        )
        proj_layer = nn.Conv1d(config.hidden_dim, config.output_dim, kernel_size=1)
        if config.use_spectral_norm:
            proj_layer = nn.utils.spectral_norm(proj_layer)
        self.proj = proj_layer
        self.activation_xy = _build_activation(config.activation_xy)
        self.activation_dt = _build_activation(config.activation_dt) if self.has_dt else None

    def forward(self, z: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        x = torch.cat([z, cond], dim=-1)
        batch_size = x.shape[0]
        hidden = self.fc(x).view(batch_size, self.config.target_len, self.config.hidden_dim)
        hidden = hidden.permute(0, 2, 1)  # (B, C, L)
        hidden = self.decoder(hidden)
        out = self.proj(hidden).permute(0, 2, 1)
        xy = self.activation_xy(out[..., : self.xy_dims])
        if self.has_dt:
            dt_raw = out[..., self.xy_dims :]
            dt = self.activation_dt(dt_raw) if self.activation_dt is not None else dt_raw
            return torch.cat([xy, dt], dim=-1)
        return xy

    def sample(self, cond: torch.Tensor, num_samples: Optional[int] = None) -> torch.Tensor:
        if num_samples is None:
            num_samples = cond.size(0)
        z = torch.randn(num_samples, self.config.latent_dim, device=cond.device, dtype=cond.dtype)
        cond = cond[:num_samples]
        return self.forward(z, cond)
