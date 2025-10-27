#!/usr/bin/env python3
"""Inspect x̂₀ reconstructions from a diffusion checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import torch

from diffusion.data import DiffusionDataConfig, create_dataloader
from diffusion.models import UNet1D, UNet1DConfig
from diffusion.noise import DiffusionScheduleConfig, build_schedule, q_sample, x0_from_eps, x0_from_v
from diffusion.utils import infer_mask_from_deltas, masked_mse


def _prepare_sequences(sequences: torch.Tensor, target_channels: int) -> torch.Tensor:
    if sequences.size(-1) == target_channels:
        return sequences
    if sequences.size(-1) > target_channels:
        return sequences[..., :target_channels]
    pad = target_channels - sequences.size(-1)
    return torch.nn.functional.pad(sequences, (0, pad))


def _load_configs(config_payload: dict) -> tuple[DiffusionDataConfig, DiffusionScheduleConfig, UNet1DConfig, str]:
    data_cfg = DiffusionDataConfig(**config_payload.get("data", {}))
    diffusion_cfg = DiffusionScheduleConfig(**config_payload.get("diffusion", {}))
    model_cfg = UNet1DConfig(**config_payload.get("model", {}))
    training_cfg = config_payload.get("training", {})
    objective = str(training_cfg.get("objective", "v")).lower()
    return data_cfg, diffusion_cfg, model_cfg, objective


def run_diagnostics(
    checkpoint_path: Path,
    *,
    max_batches: int,
    batch_size: Optional[int],
    device: str,
) -> dict:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if "config" not in checkpoint:
        raise ValueError("Checkpoint is missing embedded config metadata.")

    data_cfg, diffusion_cfg, model_cfg, objective = _load_configs(checkpoint["config"])
    if batch_size is not None:
        data_cfg.batch_size = batch_size

    val_split = data_cfg.val_split or "val"
    dataloader = create_dataloader(
        data_cfg,
        split=val_split,
        max_gestures=data_cfg.max_val_gestures,
        shuffle=False,
    )

    device_obj = torch.device(device)
    schedule = build_schedule(diffusion_cfg, device=device_obj)
    model = UNet1D(model_cfg).to(device_obj)
    ema_state = checkpoint.get("ema")
    if ema_state and "shadow" in ema_state:
        model.load_state_dict(ema_state["shadow"])
    else:
        model.load_state_dict(checkpoint["model"])
    model.eval()

    total_loss = 0.0
    total_mae = 0.0
    batch_count = 0
    seq_std_accum = torch.zeros(model_cfg.in_channels)
    xhat_std_accum = torch.zeros(model_cfg.in_channels)

    with torch.no_grad():
        for batch in dataloader:
            sequences = batch["sequences"].to(device_obj)
            sequences = _prepare_sequences(sequences, model_cfg.in_channels)
            mask = infer_mask_from_deltas(sequences)

            timesteps = torch.randint(0, schedule.timesteps, (sequences.size(0),), device=device_obj)
            noise = torch.randn_like(sequences)
            xt, eps = q_sample(schedule, sequences, timesteps, noise=noise)
            alpha, sigma = schedule.coefficients(timesteps, device=device_obj)

            preds = model(xt.permute(0, 2, 1), timesteps, mask=mask)
            preds = preds.permute(0, 2, 1)

            if objective == "epsilon":
                x0_hat = x0_from_eps(xt, preds, alpha, sigma)
            else:
                x0_hat = x0_from_v(xt, preds, alpha, sigma)

            loss = masked_mse(x0_hat, sequences, mask)
            mae = torch.mean(torch.abs(x0_hat - sequences) * mask.unsqueeze(-1))

            total_loss += float(loss.item())
            total_mae += float(mae.item())
            seq_std_accum += sequences.detach().cpu().std(dim=(0, 1))
            xhat_std_accum += x0_hat.detach().cpu().std(dim=(0, 1))
            batch_count += 1

            if 0 < max_batches <= batch_count:
                break

    if batch_count == 0:
        raise RuntimeError("Validation dataloader produced zero batches; check config overrides.")

    seq_std = (seq_std_accum / batch_count).tolist()
    xhat_std = (xhat_std_accum / batch_count).tolist()
    ratio = [float(h / max(s, 1e-6)) for h, s in zip(xhat_std, seq_std)]

    return {
        "checkpoint": str(checkpoint_path),
        "objective": objective,
        "batches": batch_count,
        "avg_mse": total_loss / batch_count,
        "avg_mae": total_mae / batch_count,
        "sequence_std": seq_std,
        "xhat_std": xhat_std,
        "std_ratio": ratio,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run x̂₀ variance diagnostics on a diffusion checkpoint.")
    parser.add_argument("checkpoint", type=Path, help="Path to diffusion checkpoint (.pt).")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-batches", type=int, default=10, help="Number of validation batches to inspect (0 = all).")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size used for diagnostics.")
    parser.add_argument("--output", type=Path, default=None, help="Optional path to write JSON summary.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint_path = args.checkpoint.resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)
    summary = run_diagnostics(
        checkpoint_path,
        max_batches=args.max_batches,
        batch_size=args.batch_size,
        device=args.device,
    )
    print(json.dumps(summary, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
