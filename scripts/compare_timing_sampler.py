#!/usr/bin/env python3
"""Compare timing sampler outputs against cached real Δt distributions."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from timing.data import TimingDataset
from timing.sampler import TimingSampler


def _quantiles(tensor: torch.Tensor, q: tuple[float, ...] = (0.1, 0.25, 0.5, 0.75, 0.9)) -> list[float]:
    tensor = tensor.flatten().to(torch.float32)
    return [float(torch.quantile(tensor, qq).item()) for qq in q]


def _masked_profile_mean(profiles: torch.Tensor, masks: torch.Tensor | None) -> torch.Tensor:
    if masks is None:
        return profiles.mean(dim=0)
    masks_float = masks.to(dtype=profiles.dtype)
    weighted = (profiles * masks_float).sum(dim=0)
    counts = masks_float.sum(dim=0).clamp_min(1e-6)
    return weighted / counts


def summarize(label: str, durations: torch.Tensor, profiles: torch.Tensor, masks: torch.Tensor | None = None) -> None:
    mean = float(durations.mean().item())
    std = float(durations.std(unbiased=False).item())
    quants = _quantiles(durations)
    profile_mean = _masked_profile_mean(profiles, masks)
    profile_first = float(profile_mean[0].item())
    profile_mid = float(profile_mean[len(profile_mean) // 2].item())
    profile_last = float(profile_mean[-1].item())
    print(f"[{label}] duration μ={mean:.4f} σ={std:.4f} quantiles={quants}")
    print(
        f"[{label}] profile mean (mask-aware) first/mid/last = {profile_first:.5f} / {profile_mid:.5f} / {profile_last:.5f}"
    )
    if masks is not None:
        lengths = masks.sum(dim=1).to(torch.float32)
        len_quants = _quantiles(lengths)
        print(f"[{label}] valid length quantiles (steps) = {len_quants}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Path to timing model checkpoint.")
    parser.add_argument("--cache", required=True, help="Path to timing cache (train_timing.pt).")
    parser.add_argument("--samples", type=int, default=2048, help="Number of samples to draw from the sampler.")
    parser.add_argument("--temperature", type=float, default=1.0, help="Dirichlet temperature scaling.")
    parser.add_argument("--device", default="auto", help="Device for inference (cuda|cpu|auto).")
    parser.add_argument("--clip-quantile", type=float, default=0.94, help="Quantile of real durations used to set clip.")
    parser.add_argument("--clip-multiplier", type=float, default=1.002, help="Multiplier applied to quantile for clipping.")
    parser.add_argument("--max-duration", type=float, default=None, help="Explicit max duration; overrides quantile-based clip.")
    parser.add_argument("--no-clip", action="store_true", help="Disable duration clipping entirely.")
    parser.add_argument("--profile-mix", type=float, default=0.0, help="Blend factor between cache profile (0) and model sample (1).")
    parser.add_argument("--duration-mix", type=float, default=0.0, help="Blend factor between cache duration (0) and model prediction (1).")
    args = parser.parse_args()

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device)
    cache = TimingDataset(Path(args.cache))
    clip_args = {
        "clip_quantile": args.clip_quantile,
        "clip_multiplier": args.clip_multiplier,
        "max_duration": args.max_duration,
        "profile_mix": args.profile_mix,
        "duration_mix": args.duration_mix,
    }
    if args.no_clip:
        clip_args.update({"clip_quantile": 1.0, "clip_multiplier": 1.0, "max_duration": None})
    sampler = TimingSampler.from_checkpoint(
        checkpoint_path=args.checkpoint,
        cache_path=args.cache,
        device=device,
        temperature=args.temperature,
        **clip_args,
    )

    real_durations = cache.durations
    real_profiles = cache.profiles
    real_masks = cache.masks
    summarize("Real", real_durations, real_profiles, real_masks)

    batches = []
    remaining = args.samples
    while remaining > 0:
        batch = min(remaining, 1024)
        sample = sampler.sample(batch)
        batches.append(sample)
        remaining -= batch
    fake_durations = torch.cat([b["duration"].cpu() for b in batches], dim=0)
    fake_profiles = torch.cat([b["profile"].cpu() for b in batches], dim=0)
    fake_masks = torch.cat([b["mask"].cpu().float() for b in batches], dim=0)
    summarize("Sampled", fake_durations, fake_profiles, fake_masks)
    clip_info = f"(clip={sampler.duration_clip:.3f} from q={sampler.clip_quantile}×{sampler.clip_multiplier})"
    print(f"[Sampler] duration clip {clip_info}")

    n = min(fake_durations.size(0), real_durations.size(0))
    real_durations_n = real_durations[:n]
    fake_durations_n = fake_durations[:n]
    duration_mse = torch.mean((fake_durations_n - real_durations_n) ** 2).item()

    real_profiles_n = real_profiles[:n]
    real_masks_n = real_masks[:n]
    fake_profiles_n = fake_profiles[:n]
    fake_masks_n = fake_masks[:n] > 0.5
    common_mask = real_masks_n.bool() & fake_masks_n
    common_mask_float = common_mask.to(dtype=real_profiles_n.dtype)
    diff = (fake_profiles_n - real_profiles_n) ** 2 * common_mask_float
    denom = common_mask_float.sum().clamp_min(1.0)
    profile_mse = float(diff.sum().item() / denom.item())

    print(f"Duration MSE (vs first {n} reals): {duration_mse:.6f}")
    print(f"Profile MSE (mask-aware, vs first {n} reals): {profile_mse:.6f}")


if __name__ == "__main__":
    main()
