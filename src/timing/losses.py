"""Loss helpers for timing models."""

from __future__ import annotations

import torch


def lognormal_nll(log_duration: torch.Tensor, mu: torch.Tensor, log_sigma: torch.Tensor) -> torch.Tensor:
    sigma = torch.exp(log_sigma)
    var = sigma**2
    return 0.5 * torch.log(2 * torch.pi * var) + (log_duration - mu) ** 2 / (2 * var)


def dirichlet_nll(
    target_profile: torch.Tensor,
    template: torch.Tensor,
    concentration: torch.Tensor,
    mask: torch.Tensor,
    *,
    smoothing: float = 1e-3,
    reduction: str = "mean",
    eps: float = 1e-8,
) -> torch.Tensor:
    """Compute masked Dirichlet NLL between target profile and template/concentration.

    Args:
        target_profile: [B, T] target Δt proportions (can contain zeros outside mask).
        template: [B, T] predicted template probabilities (already masked softmax output).
        concentration: [B] concentration scalars.
        mask: [B, T] boolean mask indicating valid time steps.
        smoothing: Dirichlet label smoothing coefficient applied per sequence.
        reduction: "mean", "sum", or "none".
        eps: numerical stability constant.
    """
    losses = torch.zeros(target_profile.size(0), device=target_profile.device, dtype=target_profile.dtype)
    mask_bool = mask.to(dtype=torch.bool)

    for idx, (target, tmpl, kappa, m) in enumerate(zip(target_profile, template, concentration, mask_bool)):
        valid_idx = m.nonzero(as_tuple=True)[0]
        if valid_idx.numel() < 2:
            continue

        target_valid = target[valid_idx].clamp_min(eps)
        target_valid = target_valid / target_valid.sum().clamp_min(eps)
        if smoothing > 0.0:
            smooth = smoothing / float(valid_idx.numel())
            target_valid = (1.0 - smoothing) * target_valid + smooth
            target_valid = target_valid / target_valid.sum().clamp_min(eps)

        tmpl_valid = tmpl[valid_idx].clamp_min(eps)
        tmpl_valid = tmpl_valid / tmpl_valid.sum().clamp_min(eps)
        alpha = (kappa * tmpl_valid).clamp_min(eps)

        log_norm = torch.lgamma(alpha.sum()) - torch.sum(torch.lgamma(alpha))
        log_prob = torch.sum((alpha - 1.0) * torch.log(target_valid))
        losses[idx] = -(log_norm + log_prob)

    if reduction == "none":
        return losses
    if reduction == "sum":
        return losses.sum()
    valid_sequences = (mask_bool.sum(dim=-1) >= 2).to(dtype=losses.dtype)
    denom = valid_sequences.sum().clamp_min(1.0)
    weighted = losses * valid_sequences
    return weighted.sum() / denom


def progress_cdf_loss(
    template: torch.Tensor,
    target_profile: torch.Tensor,
    mask: torch.Tensor,
    *,
    reduction: str = "mean",
    eps: float = 1e-8,
) -> torch.Tensor:
    """Compare cumulative timing profiles under a mask."""
    mask_float = mask.to(dtype=template.dtype)
    template = template * mask_float
    target_profile = target_profile * mask_float
    template = template / template.sum(dim=-1, keepdim=True).clamp_min(eps)
    target_profile = target_profile / target_profile.sum(dim=-1, keepdim=True).clamp_min(eps)

    template_cdf = torch.cumsum(template, dim=-1)
    target_cdf = torch.cumsum(target_profile, dim=-1)
    diff = (template_cdf - target_cdf) ** 2 * mask_float
    per_sequence = diff.sum(dim=-1) / mask_float.sum(dim=-1).clamp_min(1.0)

    if reduction == "none":
        return per_sequence
    if reduction == "sum":
        return per_sequence.sum()
    return per_sequence.mean()
