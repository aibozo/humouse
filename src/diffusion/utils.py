"""Utility helpers for diffusion training and evaluation."""

from __future__ import annotations

import copy
from typing import Optional

import torch


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, *, weights: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Compute mean squared error on masked sequences."""
    if pred.shape != target.shape:
        raise ValueError("Pred and target must have the same shape.")
    if mask.shape != pred.shape[:2]:
        raise ValueError("Mask must match batch and time dimensions.")
    if pred.ndim != 3:
        raise ValueError("Pred/target tensors must be 3D [B, T, C].")
    mask = mask.to(dtype=pred.dtype)
    errors = (pred - target) ** 2
    if weights is not None:
        while weights.ndim < errors.ndim:
            weights = weights.unsqueeze(-1)
        errors = errors * weights
    errors = errors * mask.unsqueeze(-1)
    denom = (mask.sum() * pred.shape[-1]).clamp_min(1.0)
    return errors.sum() / denom


def infer_mask_from_deltas(deltas: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Infer a validity mask from delta sequences assuming padded zeros."""
    if deltas.ndim != 3:
        raise ValueError("Delta tensor must have shape [B, T, C].")
    return (deltas.abs().sum(dim=-1) > eps).to(dtype=deltas.dtype)


def cumulative_positions(deltas: torch.Tensor) -> torch.Tensor:
    """Compute cumulative positions from delta sequences."""
    return torch.cumsum(deltas, dim=-2)


def match_time_channel(target: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Copy Δt channel from reference into target for overlapping batch/time extent."""
    if target.ndim != 3 or reference.ndim != 3:
        raise ValueError("match_time_channel expects tensors shaped [B, T, C].")
    if target.size(-1) < 3 or reference.size(-1) < 3:
        return target
    batch = min(target.size(0), reference.size(0))
    steps = min(target.size(1), reference.size(1))
    if batch <= 0 or steps <= 0:
        return target
    out = target.clone()
    out[:batch, :steps, 2] = reference[:batch, :steps, 2].to(out.device, dtype=out.dtype)
    return out


class EMAModel:
    """Exponential moving average tracker for model parameters."""

    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.decay = float(decay)
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, source: torch.nn.Module) -> None:
        for shadow_param, source_param in zip(self.shadow.parameters(), source.parameters()):
            shadow_param.data.lerp_(source_param.data, 1.0 - self.decay)
        for shadow_buf, source_buf in zip(self.shadow.buffers(), source.buffers()):
            shadow_buf.data.copy_(source_buf.data)

    def state_dict(self) -> dict:
        return {
            "decay": self.decay,
            "shadow": self.shadow.state_dict(),
        }

    def load_state_dict(self, state: dict) -> None:
        self.decay = float(state.get("decay", self.decay))
        self.shadow.load_state_dict(state["shadow"])

    def to(self, device: torch.device | str) -> "EMAModel":
        self.shadow.to(device)
        return self
