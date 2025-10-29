"""Utilities for sampling Δt using trained timing models."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
from torch.distributions import Dirichlet
import torch.nn.functional as F

from timing.data import TimingDataset
from timing.model import TimingModel, TimingModelConfig

try:
    torch.serialization.add_safe_globals([TimingModelConfig])
except AttributeError:
    pass


class TimingSampler:
    def __init__(
        self,
        model: TimingModel,
        cache: TimingDataset,
        *,
        device: torch.device,
        temperature: float = 1.0,
        clip_quantile: float = 0.94,
        clip_multiplier: float = 1.002,
        max_duration: Optional[float] = None,
        profile_mix: float = 0.0,
        duration_mix: float = 0.0,
        min_profile_value: float = 1e-4,
    ) -> None:
        self.model = model.to(device).eval()
        self.device = device
        self.temperature = float(temperature)
        self.features = cache.features.clone()
        self.masks = cache.masks.clone()
        self.profiles = cache.profiles.clone()
        self.durations = cache.durations.clone()
        self.sequence_length = cache.profiles.size(1)
        if max_duration is not None:
            self.duration_clip = float(max_duration)
        else:
            durations = cache.durations
            q = float(torch.quantile(durations, torch.tensor(clip_quantile)))
            self.duration_clip = q * clip_multiplier
        self.clip_quantile = clip_quantile
        self.clip_multiplier = clip_multiplier
        self.profile_mix = float(max(0.0, min(1.0, profile_mix)))
        self.duration_mix = float(max(0.0, min(1.0, duration_mix)))
        self.min_profile_value = float(max(0.0, min_profile_value))

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        cache_path: str | Path,
        *,
        device: torch.device,
        temperature: float = 1.0,
        clip_quantile: float = 0.94,
        clip_multiplier: float = 1.002,
        max_duration: Optional[float] = None,
        profile_mix: float = 0.0,
        duration_mix: float = 0.0,
        min_profile_value: float = 1e-4,
    ) -> "TimingSampler":
        checkpoint = torch.load(Path(checkpoint_path), map_location=device)
        model_cfg = checkpoint.get("config")
        if isinstance(model_cfg, dict):
            model_cfg = TimingModelConfig(**model_cfg)
        elif model_cfg is None:
            raise ValueError("Timing checkpoint missing config entry")
        model = TimingModel(model_cfg)
        model.load_state_dict(checkpoint["model"])
        cache = TimingDataset(cache_path)
        return cls(
            model=model,
            cache=cache,
            device=device,
            temperature=temperature,
            clip_quantile=clip_quantile,
            clip_multiplier=clip_multiplier,
            max_duration=max_duration,
            profile_mix=profile_mix,
            duration_mix=duration_mix,
            min_profile_value=min_profile_value,
        )

    @torch.no_grad()
    def sample(self, batch_size: int) -> dict[str, torch.Tensor]:
        idx = torch.randint(0, self.features.size(0), (batch_size,))
        feat = self.features[idx].to(self.device)
        mask = self.masks[idx].to(self.device)
        mask_bool = mask.bool()
        mask_float = mask_bool.float()
        cache_profile = self.profiles[idx].to(self.device)
        cache_duration = self.durations[idx].to(self.device)
        outputs = self.model(feat, mask=mask_bool)
        mu = outputs["mu_log_duration"]
        log_sigma = outputs["log_sigma_log_duration"]
        sigma = log_sigma.exp()
        log_duration = mu + sigma * torch.randn_like(mu)
        duration = log_duration.exp().clamp_min(1e-4)

        alpha = outputs["template"] * outputs["concentration"].unsqueeze(-1)
        alpha = (alpha * mask_float + (~mask_bool).float() * self.min_profile_value).clamp_min(self.min_profile_value)
        alpha = alpha * self.temperature
        dist = Dirichlet(alpha)
        profile = dist.sample()
        profile = profile * mask_float

        cache_profile = cache_profile * mask_float
        cache_profile = cache_profile / cache_profile.sum(dim=1, keepdim=True).clamp_min(1e-6)
        eps = self.min_profile_value
        if eps > 0.0:
            cache_profile = torch.where(mask_bool, cache_profile.clamp_min(eps), torch.zeros_like(cache_profile))
            cache_profile = cache_profile / cache_profile.sum(dim=1, keepdim=True).clamp_min(1e-6)
            cache_profile = cache_profile * mask_float
        if self.profile_mix <= 0.0:
            profile = cache_profile
        elif self.profile_mix < 1.0:
            profile = torch.lerp(cache_profile, profile, self.profile_mix)
        eps = self.min_profile_value
        if eps > 0.0:
            profile = torch.where(mask_bool, profile.clamp_min(eps), torch.zeros_like(profile))
        eps = self.min_profile_value
        profile = profile / profile.sum(dim=1, keepdim=True).clamp_min(1e-6)
        profile = profile * mask_float

        if self.duration_mix <= 0.0:
            duration = cache_duration
        elif self.duration_mix < 1.0:
            duration = cache_duration + (duration - cache_duration) * self.duration_mix

        duration = duration.clamp_max(self.duration_clip)
        delta_t = profile * duration.unsqueeze(-1)
        return {
            "duration": duration,
            "profile": profile,
            "mask": mask_bool,
            "delta_t": delta_t,
        }

    @torch.no_grad()
    def assign(self, sequences: torch.Tensor) -> torch.Tensor:
        if sequences.size(1) != self.sequence_length:
            sequences = self._resize_time(sequences, self.sequence_length)
        batch = sequences.size(0)
        sample = self.sample(batch)
        dt = sample["delta_t"].to(sequences.device)
        mask = sample["mask"].to(sequences.device).unsqueeze(-1)
        if sequences.size(-1) >= 3:
            out = sequences.clone()
            if sequences.size(-1) >= 2:
                out[..., :2] = out[..., :2] * mask
            out[..., 2] = dt
            return out
        xy = sequences * mask[..., :1] if sequences.size(-1) >= 2 else sequences
        return torch.cat([xy, dt.unsqueeze(-1)], dim=-1)

    @staticmethod
    def _resize_time(sequences: torch.Tensor, target_len: int) -> torch.Tensor:
        if target_len <= 0:
            raise ValueError("target_len must be positive")
        curr_len = sequences.size(1)
        if curr_len == target_len:
            return sequences
        seq = sequences.permute(0, 2, 1)
        mode = "linear" if seq.size(1) <= 4 else "area"
        resized = F.interpolate(seq, size=target_len, mode=mode, align_corners=False)
        return resized.permute(0, 2, 1)
