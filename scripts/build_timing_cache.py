#!/usr/bin/env python3
"""Build timing caches (durations + normalized Δt profiles) for gesture datasets."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List

import torch

from diffusion.data import DiffusionDataConfig, build_diffusion_dataset
from diffusion.utils import infer_mask_from_deltas


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-id", default="balabit", help="Dataset identifier registered in GestureDataset.")
    parser.add_argument("--sequence-length", type=int, default=64, help="Sequence length to load from the dataset cache.")
    parser.add_argument("--splits", default="train,val", help="Comma-separated list of splits to process (e.g., train,val,test).")
    parser.add_argument("--max-gestures", type=int, default=0, help="Optional cap per split; 0 disables the cap.")
    parser.add_argument("--sampling-rate", default="auto", help="Sampling rate override passed to the dataset (float or 'auto').")
    parser.add_argument("--canonicalize-path", action="store_true", help="Enable unit-path canonicalization before stats.")
    parser.add_argument("--canonicalize-duration", action="store_true", help="Enable unit-duration canonicalization before stats.")
    parser.add_argument("--no-normalize", action="store_true", help="Disable sequence normalization when loading the dataset.")
    parser.add_argument("--output-dir", default="datasets", help="Base directory to write timing cache files into.")
    parser.add_argument("--feature-mode", default="neuromotor", help="Feature mode for GestureDataset (unused but recorded).")
    parser.add_argument("--include-goal-geometry", action="store_true", help="Preserve goal geometry features in dataset samples.")
    parser.add_argument("--summary", action="store_true", help="Print JSON summary after finishing all splits.")
    return parser.parse_args()


def _build_data_config(args: argparse.Namespace) -> DiffusionDataConfig:
    max_gestures = None if args.max_gestures <= 0 else args.max_gestures
    sampling_rate: float | str | None
    if args.sampling_rate.lower() == "auto":
        sampling_rate = "auto"
    else:
        try:
            sampling_rate = float(args.sampling_rate)
        except ValueError:  # invalid literal, fallback to None
            sampling_rate = None
    return DiffusionDataConfig(
        dataset_id=args.dataset_id,
        sequence_length=args.sequence_length,
        batch_size=1,
        num_workers=0,
        canonicalize_path=args.canonicalize_path,
        canonicalize_duration=args.canonicalize_duration,
        normalize_sequences=not args.no_normalize,
        normalize_features=not args.no_normalize,
        include_goal_geometry=args.include_goal_geometry,
        feature_mode=args.feature_mode,
        sampling_rate=sampling_rate,
        max_train_gestures=max_gestures,
        max_val_gestures=max_gestures,
    )


def _compute_split_payload(dataset, seq_len: int) -> Dict[str, torch.Tensor | Dict[str, float]]:
    durations: List[float] = []
    profiles: List[torch.Tensor] = []
    masks: List[torch.Tensor] = []
    features: List[torch.Tensor] = []
    for seq, feat, label in dataset.samples:
        if float(label.item()) < 0.5:
            continue
        if seq.size(-1) < 3:
            continue
        # Estimate validity after denormalisation so padded steps remain masked out.
        seq_batch = seq.unsqueeze(0)
        denorm = dataset.denormalize_sequences(seq_batch)[0]
        mask = infer_mask_from_deltas(denorm.unsqueeze(0)).squeeze(0).bool()
        if not mask.any():
            continue
        dt = denorm[:, 2]
        valid_dt = dt[mask]
        total = float(valid_dt.sum().item())
        if total <= 1e-6:
            continue
        durations.append(total)
        profile = torch.zeros(seq_len, dtype=torch.float32)
        profile[mask] = (valid_dt / total).to(torch.float32)
        profiles.append(profile)
        masks.append(mask.to(torch.float32))
        features_tensor = feat.to(torch.float32).clone()
        features.append(features_tensor)

    if not durations:
        raise RuntimeError("No positive samples with valid Δt found for this split.")

    durations_tensor = torch.tensor(durations, dtype=torch.float32)
    log_durations = torch.log(durations_tensor)
    profiles_tensor = torch.stack(profiles)
    masks_tensor = torch.stack(masks)
    features_tensor = torch.stack(features) if features else torch.empty((0, 0))

    valid_counts = masks_tensor.sum(dim=0)
    profile_sum = (profiles_tensor * masks_tensor).sum(dim=0)
    masked_mean = profile_sum / valid_counts.clamp_min(1e-6)
    profile_sq_sum = ((profiles_tensor ** 2) * masks_tensor).sum(dim=0)
    masked_var = profile_sq_sum / valid_counts.clamp_min(1e-6) - masked_mean**2
    masked_var = torch.clamp(masked_var, min=0.0)
    valid_lengths = masks_tensor.sum(dim=1)
    valid_lengths_float = valid_lengths.to(torch.float32)
    quantiles = {
        "p10": float(valid_lengths_float.quantile(0.10).item()),
        "p25": float(valid_lengths_float.quantile(0.25).item()),
        "p50": float(valid_lengths_float.quantile(0.50).item()),
        "p75": float(valid_lengths_float.quantile(0.75).item()),
        "p90": float(valid_lengths_float.quantile(0.90).item()),
    }

    stats = {
        "count": int(durations_tensor.numel()),
        "duration_mean": float(durations_tensor.mean().item()),
        "duration_std": float(durations_tensor.std(unbiased=False).item()),
        "duration_log_mean": float(log_durations.mean().item()),
        "duration_log_std": float(log_durations.std(unbiased=False).item()),
        "duration_min": float(durations_tensor.min().item()),
        "duration_max": float(durations_tensor.max().item()),
        "profile_mean": profiles_tensor.mean(dim=0).tolist(),
        "profile_var": profiles_tensor.var(dim=0, unbiased=False).tolist(),
        "profile_mean_masked": masked_mean.tolist(),
        "profile_var_masked": masked_var.tolist(),
        "profile_valid_counts": valid_counts.tolist(),
        "valid_length_mean": float(valid_lengths_float.mean().item()),
        "valid_length_median": float(valid_lengths_float.median().item()),
        "valid_length_quantiles": quantiles,
        "feature_dim": int(features_tensor.shape[1]) if features_tensor.ndim == 2 else 0,
    }
    return {
        "durations": durations_tensor,
        "profiles": profiles_tensor,
        "masks": masks_tensor,
        "stats": stats,
        "features": features_tensor,
    }


def main() -> None:
    args = _parse_args()
    data_cfg = _build_data_config(args)
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    if not splits:
        raise ValueError("No dataset splits specified.")

    base_output = Path(args.output_dir) / args.dataset_id
    base_output.mkdir(parents=True, exist_ok=True)

    summary: Dict[str, dict] = {}
    for split in splits:
        dataset = build_diffusion_dataset(
            data_cfg,
            split=split,
            max_gestures=data_cfg.max_train_gestures if split == data_cfg.train_split else data_cfg.max_val_gestures,
        )
        payload = _compute_split_payload(dataset, data_cfg.sequence_length)
        payload["config"] = asdict(data_cfg)
        payload["split"] = split
        output_path = base_output / f"{split}_timing.pt"
        torch.save(payload, output_path)
        summary[split] = payload["stats"]
        print(f"Saved timing cache for split '{split}' to {output_path} (count={payload['stats']['count']})")

    if args.summary:
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
