"""LSTM-based generator and discriminator for mouse trajectory GAN."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class LSTMConfig:
    hidden_dim: int
    layers: int


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
    raise ValueError(f"Unsupported activation '{name}' for LSTM generator output")


class LSTMGenerator(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.xy_dims = min(2, config.output_dim)
        self.has_dt = config.output_dim > self.xy_dims
        input_dim = config.latent_dim + config.condition_dim
        hidden_dim = config.hidden_dim
        self.pre = nn.Linear(input_dim, hidden_dim)
        lstm_layers = []
        for _ in range(max(1, config.num_layers)):
            lstm_layers.append(nn.LSTM(hidden_dim, hidden_dim, batch_first=True))
        self.lstm_layers = nn.ModuleList(lstm_layers)
        self.output = nn.Linear(hidden_dim, config.output_dim)
        self.activation_xy = _build_activation(config.activation_xy)
        self.activation_dt = _build_activation(config.activation_dt) if self.has_dt else None

    def forward(self, z: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        x = torch.cat([z, cond], dim=-1)
        base = torch.tanh(self.pre(x))
        repeated = base.unsqueeze(1).repeat(1, self.config.target_len, 1)
        out = repeated
        for lstm in self.lstm_layers:
            out, _ = lstm(out)
        out = self.output(out)
        xy = self.activation_xy(out[..., : self.xy_dims])
        if self.has_dt:
            dt_raw = out[..., self.xy_dims :]
            dt = self.activation_dt(dt_raw) if self.activation_dt is not None else dt_raw
            return torch.cat([xy, dt], dim=-1)
        return xy

    def sample(self, cond: torch.Tensor, num_samples: int | None = None) -> torch.Tensor:
        if num_samples is None:
            num_samples = cond.size(0)
        z = torch.randn(num_samples, self.config.latent_dim, device=cond.device, dtype=cond.dtype)
        cond = cond[:num_samples]
        return self.forward(z, cond)


class LSTMDiscriminator(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        hidden_dim = config.hidden_dim
        lstm_layers = []
        input_dim = config.input_dim
        for idx in range(max(1, config.num_layers)):
            lstm_layers.append(nn.LSTM(input_dim if idx == 0 else hidden_dim, hidden_dim, batch_first=True))
        self.lstm_layers = nn.ModuleList(lstm_layers)
        self.condition_proj = nn.Linear(config.condition_dim, hidden_dim) if config.condition_dim > 0 else None
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, sequence: torch.Tensor, cond: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = sequence
        for lstm in self.lstm_layers:
            out, _ = lstm(out)
        pooled = out.mean(dim=1)
        if self.condition_proj is not None and cond.shape[-1] > 0:
            cond_embed = torch.tanh(self.condition_proj(cond))
            fused = pooled + cond_embed
        else:
            fused = pooled
        logits = self.classifier(fused)
        return logits.squeeze(-1), torch.zeros(sequence.size(0), device=sequence.device, dtype=sequence.dtype)


__all__ = ["LSTMGenerator", "LSTMDiscriminator"]
