"""Diffusion noise schedules and helper utilities."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch


@dataclass
class DiffusionScheduleConfig:
    timesteps: int = 1000
    schedule: str = "cosine"  # or "linear"
    linear_beta_start: float = 1e-4
    linear_beta_end: float = 0.02
    cosine_s: float = 0.008


class DiffusionSchedule:
    """Precomputed diffusion schedule providing alpha/sigma coefficients."""

    def __init__(self, alpha_bar: torch.Tensor):
        if alpha_bar.ndim != 1:
            raise ValueError("alpha_bar must be a 1D tensor.")
        if alpha_bar.numel() == 0:
            raise ValueError("alpha_bar must contain at least one timestep.")
        self.alpha_bar = alpha_bar.float().clamp(min=1e-5, max=0.99999)
        self.timesteps = int(self.alpha_bar.numel())
        self.alpha = torch.sqrt(self.alpha_bar)
        self.sigma = torch.sqrt(1.0 - self.alpha_bar)
        eps = 1e-12
        self.log_snr = torch.log(self.alpha_bar + eps) - torch.log1p(-self.alpha_bar + eps)

    def to(self, device: torch.device | str) -> DiffusionSchedule:
        device = torch.device(device)
        copied = DiffusionSchedule(self.alpha_bar.to(device))
        return copied

    def coefficients(self, t: torch.Tensor, *, device: Optional[torch.device] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return alpha and sigma for integer timesteps `t` with shape broadcastable to x."""
        if device is None:
            device = self.alpha.device
        t = t.to(device=device, dtype=torch.long)
        alpha = _gather(self.alpha, t)
        sigma = _gather(self.sigma, t)
        return alpha, sigma

    def log_snr_at(self, t: torch.Tensor, *, device: Optional[torch.device] = None) -> torch.Tensor:
        if device is None:
            device = self.log_snr.device
        t = t.to(device=device, dtype=torch.long)
        return _gather(self.log_snr, t)


def build_schedule(cfg: DiffusionScheduleConfig, *, device: Optional[torch.device] = None) -> DiffusionSchedule:
    """Create a diffusion schedule with the requested noise profile."""
    if cfg.timesteps <= 0:
        raise ValueError("Timesteps must be positive.")
    device = torch.device(device) if device is not None else None
    if cfg.schedule.lower() == "cosine":
        alpha_bar = _cosine_alpha_bar(cfg.timesteps, cfg.cosine_s, device=device)
    elif cfg.schedule.lower() == "linear":
        alpha_bar = _linear_alpha_bar(cfg.timesteps, cfg.linear_beta_start, cfg.linear_beta_end, device=device)
    else:
        raise ValueError(f"Unsupported schedule '{cfg.schedule}'.")
    return DiffusionSchedule(alpha_bar)


def q_sample(
    schedule: DiffusionSchedule,
    x0: torch.Tensor,
    t: torch.Tensor,
    noise: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Diffuse clean sample x0 to timestep t returning (x_t, noise_used)."""
    if noise is None:
        noise = torch.randn_like(x0)
    alpha, sigma = schedule.coefficients(t, device=x0.device)
    while alpha.ndim < x0.ndim:
        alpha = alpha.unsqueeze(-1)
        sigma = sigma.unsqueeze(-1)
    xt = alpha * x0 + sigma * noise
    return xt, noise


def compute_v(x0: torch.Tensor, noise: torch.Tensor, alpha: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """Compute velocity target v = alpha * noise - sigma * x0."""
    while alpha.ndim < x0.ndim:
        alpha = alpha.unsqueeze(-1)
        sigma = sigma.unsqueeze(-1)
    return alpha * noise - sigma * x0


def x0_from_v(xt: torch.Tensor, v: torch.Tensor, alpha: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """Recover x0 from velocity prediction."""
    while alpha.ndim < xt.ndim:
        alpha = alpha.unsqueeze(-1)
        sigma = sigma.unsqueeze(-1)
    return (alpha * xt - sigma * v) / (alpha**2 + sigma**2).clamp_min(1e-12)


def x0_from_eps(xt: torch.Tensor, eps: torch.Tensor, alpha: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """Recover x0 from epsilon prediction."""
    while alpha.ndim < xt.ndim:
        alpha = alpha.unsqueeze(-1)
        sigma = sigma.unsqueeze(-1)
    return (xt - sigma * eps) / alpha.clamp_min(1e-12)


def eps_from_v(v: torch.Tensor, x0: torch.Tensor, alpha: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """Recover epsilon noise from velocity prediction."""
    while alpha.ndim < v.ndim:
        alpha = alpha.unsqueeze(-1)
        sigma = sigma.unsqueeze(-1)
    return (v + sigma * x0) / alpha.clamp_min(1e-12)


def v_from_eps(x0: torch.Tensor, eps: torch.Tensor, alpha: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """Compute velocity target from epsilon."""
    while alpha.ndim < eps.ndim:
        alpha = alpha.unsqueeze(-1)
        sigma = sigma.unsqueeze(-1)
    return alpha * eps - sigma * x0


def _gather(values: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    gathered = values.index_select(0, t)
    return gathered.view(t.shape)


def _cosine_alpha_bar(timesteps: int, s: float, *, device: Optional[torch.device]) -> torch.Tensor:
    steps = torch.arange(timesteps + 1, dtype=torch.float64, device=device)
    t = steps / timesteps
    alpha_bar = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
    alpha_bar = alpha_bar / alpha_bar[0]
    alpha_bar = torch.clamp(alpha_bar, min=1e-5, max=0.99999)
    return alpha_bar[1:].to(dtype=torch.float32)


def _linear_alpha_bar(timesteps: int, beta_start: float, beta_end: float, *, device: Optional[torch.device]) -> torch.Tensor:
    betas = torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64, device=device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_cumprod = torch.clamp(alphas_cumprod, min=1e-5, max=0.99999)
    return alphas_cumprod.to(dtype=torch.float32)
    timesteps = cfg.timesteps
    if timesteps <= 0:
        raise ValueError("Timesteps must be positive.")
    beta = torch.linspace(1e-4, 0.02, timesteps + 1)
    alpha = 1.0 - beta
    alpha_bar = torch.cumprod(alpha, dim=0)
    sigma = torch.sqrt(1.0 - alpha_bar)
    return alpha_bar, sigma
