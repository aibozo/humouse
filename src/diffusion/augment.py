"""Augmentation helpers for diffusion training."""

from __future__ import annotations

from typing import Optional

import torch


def apply_default_augmentations(
    sequences: torch.Tensor,
    *,
    mask: Optional[torch.Tensor] = None,
    time_stretch: float = 0.0,
    jitter_std: float = 0.0,
    mirror_prob: float = 0.0,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Apply lightweight geometric/time augmentations to gesture deltas."""
    if time_stretch <= 0 and jitter_std <= 0 and mirror_prob <= 0:
        return sequences

    device = sequences.device
    dtype = sequences.dtype
    seq = sequences.clone()
    if mask is None:
        mask_tensor = torch.ones(seq.shape[0], seq.shape[1], device=device, dtype=dtype)
    else:
        mask_tensor = mask.to(device=device, dtype=dtype)

    if mirror_prob > 0 and seq.size(-1) >= 2:
        probs = torch.rand(seq.size(0), device=device, generator=generator)
        flip_mask = probs < mirror_prob
        if flip_mask.any():
            seq[flip_mask, :, 1] = seq[flip_mask, :, 1] * -1.0

    if jitter_std > 0:
        noise = torch.randn_like(seq[..., :2], generator=generator) * jitter_std
        seq[..., :2] = seq[..., :2] + noise * mask_tensor.unsqueeze(-1)

    if time_stretch > 0 and seq.size(-1) >= 3:
        stretch = 1.0 + (torch.rand(seq.size(0), device=device, generator=generator) * 2.0 - 1.0) * time_stretch
        dt = seq[..., 2]
        dt = dt * mask_tensor
        dt = dt * stretch.unsqueeze(1)
        dt_sum = dt.sum(dim=1, keepdim=True).clamp_min(1e-6)
        dt = dt / dt_sum
        seq[..., 2] = dt

    return seq
