"""Timing model predicting gesture durations and Δt profiles."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class TimingModelConfig:
    feature_dim: int = 32
    hidden_dim: int = 128
    num_layers: int = 2
    dropout: float = 0.1
    sequence_length: int = 64
    positional_dim: int = 8
    profile_mlp_dim: int = 128
    min_concentration: float = 5.0
    max_concentration: float = 30.0
    log_sigma_base: float = 0.5
    log_sigma_range: float = 0.5


class TimingModel(nn.Module):
    def __init__(self, config: TimingModelConfig):
        super().__init__()
        self.config = config
        layers = []
        in_dim = config.feature_dim
        for _ in range(config.num_layers):
            layers.append(nn.Linear(in_dim, config.hidden_dim))
            layers.append(nn.SiLU())
            layers.append(nn.Dropout(config.dropout))
            in_dim = config.hidden_dim
        self.backbone = nn.Sequential(*layers) if layers else nn.Identity()
        self.backbone_out_dim = in_dim
        pos_dim = max(0, config.positional_dim)
        self.profile_input_dim = self.backbone_out_dim + pos_dim
        self.duration_head = nn.Linear(self.backbone_out_dim, 2)
        mlp_dim = max(32, config.profile_mlp_dim)
        self.profile_head = nn.Sequential(
            nn.Linear(self.profile_input_dim, mlp_dim),
            nn.SiLU(),
            nn.Linear(mlp_dim, 1),
        )
        self.concentration_head = nn.Linear(self.backbone_out_dim, 1)
        self.register_buffer(
            "positional_features",
            self._build_positional_features(config.sequence_length, pos_dim),
            persistent=False,
        )

    def forward(self, features: torch.Tensor, mask: torch.Tensor | None = None) -> dict:
        h = self.backbone(features)
        duration_params = self.duration_head(h)
        mu_log = duration_params[:, 0]
        raw_sigma = duration_params[:, 1]
        log_sigma = self.config.log_sigma_base + self.config.log_sigma_range * torch.tanh(raw_sigma)

        batch = features.size(0)
        seq_len = self.config.sequence_length
        h_expand = h.unsqueeze(1).expand(-1, seq_len, -1)
        if self.positional_features is not None:
            pos = self.positional_features.to(h_expand.dtype)
            pos = pos.unsqueeze(0).expand(batch, -1, -1)
            profile_input = torch.cat([h_expand, pos], dim=-1)
        else:
            profile_input = h_expand
        logits = self.profile_head(profile_input).squeeze(-1)
        template = _masked_softmax(logits, mask, dim=-1)

        raw_concentration = self.concentration_head(h).squeeze(-1)
        sigmoid = torch.sigmoid(raw_concentration)
        max_conc = max(self.config.max_concentration, self.config.min_concentration + 1e-6)
        concentration = self.config.min_concentration + (max_conc - self.config.min_concentration) * sigmoid

        alpha = (concentration.unsqueeze(-1) * template).clamp_min(1e-6)
        return {
            "mu_log_duration": mu_log,
            "log_sigma_log_duration": log_sigma,
            "alpha": alpha,
            "template": template,
            "concentration": concentration,
        }

    @staticmethod
    def _build_positional_features(seq_len: int, dim: int) -> torch.Tensor | None:
        if dim <= 0:
            return None
        steps = torch.linspace(0.0, 1.0, seq_len).unsqueeze(1)
        half = dim // 2
        features = [steps]
        if half == 0:
            return steps
        freqs = torch.arange(1, half + 1).float()
        angles = 2 * torch.pi * steps * freqs
        features.extend([torch.sin(angles), torch.cos(angles)])
        feat = torch.cat(features, dim=1)
        if feat.size(1) > dim:
            feat = feat[:, :dim]
        return feat


def _masked_softmax(logits: torch.Tensor, mask: torch.Tensor | None, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    if mask is None:
        probs = torch.softmax(logits, dim=dim)
        return probs.clamp_min(eps)

    mask_bool = mask.to(dtype=torch.bool)
    valid_counts = mask_bool.sum(dim=dim, keepdim=True)
    fallback = valid_counts == 0
    if fallback.any():
        expand_fallback = fallback.expand_as(mask_bool)
        mask_bool = mask_bool | expand_fallback
    mask_float = mask_bool.to(logits.dtype)
    safe_logits = logits.masked_fill(~mask_bool, float("-inf"))
    if fallback.any():
        safe_logits = torch.where(fallback.expand_as(logits), logits, safe_logits)
    probs = torch.softmax(safe_logits, dim=dim)
    probs = probs * mask_float
    denom = probs.sum(dim=dim, keepdim=True).clamp_min(eps)
    probs = probs / denom
    probs = probs * mask_float
    return probs.clamp_min(eps)
