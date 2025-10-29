"""Hydra entry point for training diffusion models on mouse trajectories."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict, fields
from pathlib import Path
from typing import Optional

import hydra
import math
import torch
import torch.nn.functional as F
from hydra.core.config_store import ConfigStore
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from torch import amp, nn
from torch.nn.utils import clip_grad_norm_
import numpy as np

from diffusion.data import DiffusionDataConfig, create_dataloader
from diffusion.models import UNet1D, UNet1DConfig
from diffusion.noise import DiffusionScheduleConfig, build_schedule, compute_v, q_sample, x0_from_eps, x0_from_v
from diffusion.utils import EMAModel, infer_mask_from_deltas, masked_mse, match_time_channel
from timing.sampler import TimingSampler
from diffusion.augment import apply_default_augmentations
from diffusion.sample import DiffusionSampler
from features import FEATURE_SPECS, compute_features_from_sequence
from utils.logging import CSVMetricLogger

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

logger = logging.getLogger(__name__)


@dataclass
class DiffusionTimingEvalConfig:
    enabled: bool = False
    checkpoint_path: Optional[str] = None
    cache_dir: str = "datasets"
    dataset_id: Optional[str] = None
    split: str = "train"
    temperature: float = 1.0
    profile_mix: float = 0.0
    duration_mix: float = 0.0
    clip_quantile: float = 1.0
    clip_multiplier: float = 1.0
    max_duration: Optional[float] = None
    min_profile_value: float = 1e-4


@dataclass
class DiffusionEvalPrepConfig:
    """Controls how sequences are prepared for evaluation metrics."""

    force_deltas_when_missing_dt: bool = True
    clamp_delta_t: bool = True
    clamp_percentile: float = 0.5
    clamp_value: Optional[float] = None
    log_batch_stats: bool = True
    match_length_distribution: bool = False

    def __post_init__(self) -> None:
        self.clamp_percentile = float(np.clip(self.clamp_percentile, 0.0, 1.0))


@dataclass
class DiffusionTrainingConfig:
    """Training hyperparameters."""

    epochs: int = 50
    lr: float = 2e-4
    betas: tuple[float, float] = (0.9, 0.999)
    weight_decay: float = 1e-3
    ema_decay: float = 0.9995
    grad_clip: float = 1.0
    amp: bool = True
    log_interval: int = 50
    eval_interval: int = 1
    checkpoint_dir: str = "checkpoints/diffusion"
    checkpoint_interval: int = 5
    resume_from: Optional[str] = None
    eval_classifier: bool = True
    classifier_samples: int = 2048
    classifier_steps: int = 50
    min_snr_gamma: Optional[float] = None
    objective: str = "v"  # options: "v", "epsilon"
    summary_path: Optional[str] = None
    sample_eval_count: int = 256
    sample_eval_steps: int = 50
    scale_reg_weight: float = 0.0
    scale_reg_channels: Optional[list[int]] = None
    path_length_reg_weight: float = 0.0
    balance_time_steps: bool = True
    time_weight_max: Optional[float] = 4.0
    eval_path_scale_min: float = 0.05
    eval_path_scale_max: float = 4.0
    direction_change_reg_weight: float = 0.0
    curvature_reg_weight: float = 0.0
    direction_change_reg_warmup_epochs: int = 0
    curvature_reg_warmup_epochs: int = 0
    loss_skip_threshold: Optional[float] = None
    log_loss_details: bool = False
    eval_prep: DiffusionEvalPrepConfig = field(default_factory=DiffusionEvalPrepConfig)
    timing_eval: DiffusionTimingEvalConfig = field(default_factory=DiffusionTimingEvalConfig)

    def __post_init__(self) -> None:
        if not isinstance(self.betas, tuple):
            self.betas = tuple(self.betas)
        if not isinstance(self.eval_prep, DiffusionEvalPrepConfig):
            self.eval_prep = DiffusionEvalPrepConfig(**self.eval_prep)
        if not isinstance(self.timing_eval, DiffusionTimingEvalConfig):
            self.timing_eval = DiffusionTimingEvalConfig(**self.timing_eval)
        self.eval_path_scale_min = float(max(1e-3, self.eval_path_scale_min))
        self.eval_path_scale_max = float(max(self.eval_path_scale_min, self.eval_path_scale_max))


@dataclass
class DiffusionExperimentConfig:
    experiment_name: str = "diffusion_mouse_v1"
    seed: int = 1337
    data: DiffusionDataConfig = field(default_factory=DiffusionDataConfig)
    model: UNet1DConfig = field(default_factory=UNet1DConfig)
    diffusion: DiffusionScheduleConfig = field(default_factory=DiffusionScheduleConfig)
    training: DiffusionTrainingConfig = field(default_factory=DiffusionTrainingConfig)


def _filter_kwargs(datatype, data: dict) -> dict:
    valid = {f.name for f in fields(datatype)}
    return {k: v for k, v in (data or {}).items() if k in valid}


def _analyze_sequence_masks(
    dataset,
    *,
    eps: float = 1e-6,
    clamp: Optional[float] = None,
) -> tuple[torch.Tensor, np.ndarray]:
    """Compute per-time valid counts and a length distribution for the dataset."""
    if len(dataset) == 0:
        seq_len = dataset.config.sequence_length
        weights = torch.ones(seq_len, dtype=torch.float32)
        lengths = np.full((1,), seq_len, dtype=np.int64)
        return weights, lengths

    first_seq = dataset.samples[0][0]
    seq_len = first_seq.size(0)
    counts = torch.zeros(seq_len, dtype=torch.float64)
    lengths: list[int] = []
    for seq, _, _ in dataset.samples:
        mask = (seq.abs().sum(dim=-1) > eps)
        counts += mask.to(dtype=torch.float64)
        lengths.append(int(mask.sum().item()))

    weights = torch.ones_like(counts, dtype=torch.float32)
    positive = counts > 0
    if positive.any():
        max_count = counts[positive].max()
        raw = (max_count / counts[positive]).to(dtype=torch.float32)
        if clamp is not None and clamp > 0:
            raw = raw.clamp(max=float(clamp))
        mean = raw.mean().clamp_min(1e-6)
        raw = raw / mean
        weights[positive] = raw
        weights[~positive] = 0.0

    lengths_np = np.asarray(lengths, dtype=np.int64)
    return weights, lengths_np


def _masked_channel_std(tensor: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    valid = mask.to(dtype=tensor.dtype).unsqueeze(-1)
    count = valid.sum(dim=(0, 1)).clamp_min(1.0)
    mean = (tensor * valid).sum(dim=(0, 1)) / count
    centered = tensor - mean.view(1, 1, -1)
    var = ((centered**2) * valid).sum(dim=(0, 1)) / count
    return torch.sqrt(var + 1e-6)


def _sequence_path_lengths(sequences: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if sequences.ndim != 3 or sequences.size(-1) < 2:
        return torch.zeros(sequences.size(0), device=sequences.device, dtype=sequences.dtype)
    valid = mask.to(dtype=sequences.dtype).unsqueeze(-1)
    dxdy = sequences[..., :2] * valid
    step_lengths = torch.linalg.norm(dxdy, dim=-1)
    return step_lengths.sum(dim=-1)


def _angular_deltas(sequences: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    lengths = _sequence_path_lengths(sequences, mask).clamp_min(1e-6)
    xy = sequences[..., :2] / lengths.view(-1, 1, 1)
    angles = torch.atan2(xy[..., 1], xy[..., 0])
    mask_float = mask.to(dtype=angles.dtype)
    angles = angles * mask_float
    deltas = angles[:, 1:] - angles[:, :-1]
    deltas = torch.atan2(torch.sin(deltas), torch.cos(deltas))
    pair_mask = mask[:, 1:] & mask[:, :-1]
    deltas = deltas * pair_mask.to(dtype=deltas.dtype)
    return deltas, pair_mask


def _curvature_deltas(angle_deltas: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    curv = angle_deltas[:, 1:] - angle_deltas[:, :-1]
    curv = torch.atan2(torch.sin(curv), torch.cos(curv))
    curv_mask = mask[:, 1:] & mask[:, :-1]
    curv = curv * curv_mask.to(dtype=curv.dtype)
    return curv, curv_mask


def _direction_change_penalty(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    pred_delta, pred_mask = _angular_deltas(pred, mask)
    target_delta, target_mask = _angular_deltas(target, mask)
    shared_mask = pred_mask & target_mask
    if not bool(shared_mask.any()):
        return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
    diff = pred_delta - target_delta
    penalty = (diff.pow(2) * shared_mask.to(diff.dtype)).sum() / shared_mask.sum().clamp_min(1.0)
    return penalty


def _curvature_penalty(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    pred_delta, pred_mask = _angular_deltas(pred, mask)
    target_delta, target_mask = _angular_deltas(target, mask)
    pred_curv, pred_curv_mask = _curvature_deltas(pred_delta, pred_mask)
    target_curv, target_curv_mask = _curvature_deltas(target_delta, target_mask)
    shared_mask = pred_curv_mask & target_curv_mask
    if not bool(shared_mask.any()):
        return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
    diff = pred_curv - target_curv
    penalty = (diff.pow(2) * shared_mask.to(diff.dtype)).sum() / shared_mask.sum().clamp_min(1.0)
    return penalty


def _rescale_sequences_to_length(
    sequences: torch.Tensor,
    mask: torch.Tensor,
    target_lengths: torch.Tensor,
    *,
    min_scale: float = 0.1,
    max_scale: float = 10.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    if sequences.ndim != 3 or sequences.size(-1) < 2:
        return sequences, torch.ones(sequences.size(0), device=sequences.device, dtype=sequences.dtype)
    device = sequences.device
    dtype = sequences.dtype
    target = target_lengths.to(device=device, dtype=dtype)
    valid = mask.to(dtype=dtype).unsqueeze(-1)
    dxdy = sequences[..., :2]
    fake_lengths = torch.linalg.norm(dxdy * valid, dim=-1).sum(dim=-1).clamp_min(1e-6)
    scale = (target / fake_lengths).clamp(min=min_scale, max=max_scale)
    scaled = sequences.clone()
    scaled[..., :2] = dxdy * scale.view(-1, 1, 1)
    return scaled, scale


def _load_experiment_config(cfg: DictConfig) -> DiffusionExperimentConfig:
    container = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(container, dict):
        raise TypeError("Expected dict container for diffusion config")
    if "diffusion" in container and isinstance(container["diffusion"], dict):
        container = container["diffusion"]
    return DiffusionExperimentConfig(
        experiment_name=container.get("experiment_name", "diffusion"),
        seed=int(container.get("seed", 0)),
        data=DiffusionDataConfig(**_filter_kwargs(DiffusionDataConfig, container.get("data", {}))),
        model=UNet1DConfig(**_filter_kwargs(UNet1DConfig, container.get("model", {}))),
        diffusion=DiffusionScheduleConfig(**_filter_kwargs(DiffusionScheduleConfig, container.get("diffusion", {}))),
        training=DiffusionTrainingConfig(**_filter_kwargs(DiffusionTrainingConfig, container.get("training", {}))),
    )


def _prepare_sequences(
    sequences: torch.Tensor,
    *,
    target_channels: int,
) -> torch.Tensor:
    if sequences.shape[-1] == target_channels:
        return sequences
    if sequences.shape[-1] > target_channels:
        return sequences[..., :target_channels]
    pad = target_channels - sequences.shape[-1]
    return F.pad(sequences, (0, pad))


def _prepare_features(features: torch.Tensor, cond_dim: int) -> Optional[torch.Tensor]:
    if cond_dim == 0:
        return None
    if features.shape[1] == cond_dim:
        return features
    if features.shape[1] > cond_dim:
        return features[:, :cond_dim]
    pad = cond_dim - features.shape[1]
    return F.pad(features, (0, pad))


def _maybe_build_timing_sampler(
    timing_cfg: DiffusionTimingEvalConfig,
    data_cfg: DiffusionDataConfig,
    device: torch.device,
) -> Optional[TimingSampler]:
    if not timing_cfg.enabled or not timing_cfg.checkpoint_path:
        return None
    dataset_id = timing_cfg.dataset_id or data_cfg.dataset_id
    cache_path = Path(timing_cfg.cache_dir) / dataset_id / f"{timing_cfg.split}_timing.pt"
    if not cache_path.exists():
        logger.warning("Timing cache not found at %s; disabling timing eval", cache_path)
        return None
    checkpoint = Path(timing_cfg.checkpoint_path)
    if not checkpoint.exists():
        logger.warning("Timing checkpoint not found at %s; disabling timing eval", checkpoint)
        return None
    try:
        sampler = TimingSampler.from_checkpoint(
            checkpoint,
            cache_path,
            device=device,
            temperature=timing_cfg.temperature,
            profile_mix=timing_cfg.profile_mix,
            duration_mix=timing_cfg.duration_mix,
            clip_quantile=timing_cfg.clip_quantile,
            clip_multiplier=timing_cfg.clip_multiplier,
            max_duration=timing_cfg.max_duration,
            min_profile_value=timing_cfg.min_profile_value,
        )
        logger.info("Loaded timing sampler from %s", checkpoint)
        return sampler
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to load timing sampler: %s", exc)
        return None


def _positions_to_deltas_xy(sequences: torch.Tensor) -> torch.Tensor:
    if sequences.ndim != 3:
        raise ValueError("Expected [B, T, C] tensor")
    if sequences.size(-1) < 2:
        return sequences
    pos = sequences[..., :2]
    first = pos[:, :1, :]
    diffs = pos[:, 1:, :] - pos[:, :-1, :]
    deltas = torch.cat([first, diffs], dim=1)
    if sequences.size(-1) > 2:
        remainder = sequences[..., 2:]
        return torch.cat([deltas, remainder], dim=-1)
    return deltas


def _ensure_dt_channel(sequences: torch.Tensor, default_dt: float) -> torch.Tensor:
    if sequences.ndim != 3:
        raise ValueError("Expected [B, T, C] tensor")
    if sequences.size(-1) >= 3:
        return sequences
    fill = torch.full(
        (sequences.size(0), sequences.size(1), 1),
        float(default_dt),
        device=sequences.device,
        dtype=sequences.dtype,
    )
    return torch.cat([sequences, fill], dim=-1)


def _estimate_dt_value(sequences: torch.Tensor, percentile: float) -> float:
    if sequences.ndim != 3 or sequences.size(-1) < 3:
        return 1.0 / max(sequences.size(1), 1)
    dt = sequences[..., 2].reshape(-1)
    valid = dt[dt > 1e-8]
    values = valid if valid.numel() > 0 else dt
    if values.numel() == 0:
        return 1.0 / max(sequences.size(1), 1)
    if values.numel() == 1:
        return float(values.item())
    q = torch.quantile(values, float(percentile))
    return float(q.item())


def _clamp_dt_channel(sequences: torch.Tensor, value: float) -> torch.Tensor:
    if sequences.ndim != 3 or sequences.size(-1) < 3:
        return sequences
    out = sequences.clone()
    out[..., 2] = float(value)
    return out


def _describe_sequence_batch(tag: str, sequences: torch.Tensor, mask: Optional[torch.Tensor] = None) -> None:
    if sequences.ndim != 3 or sequences.size(-1) < 2:
        return
    if mask is None:
        mask = infer_mask_from_deltas(sequences).bool()
    else:
        mask = mask.to(dtype=torch.bool)
    if not mask.any():
        logger.info("%s: no valid timesteps", tag)
        return
    dxdy = sequences[..., :2][mask]
    dxdy_std = float(dxdy.std().item()) if dxdy.numel() else 0.0
    if sequences.size(-1) >= 3:
        dt = sequences[..., 2][mask]
    else:
        dt = torch.ones_like(dxdy[..., 0])
    dt_median = float(dt.median().item())
    dt_min = float(dt.min().item())
    dt_max = float(dt.max().item())
    speed = torch.linalg.norm(dxdy, dim=-1) / dt.clamp_min(1e-6)
    speed = torch.nan_to_num(speed, nan=0.0, posinf=0.0, neginf=0.0)
    speed_median = float(speed.median().item())
    speed_p95 = float(torch.quantile(speed, 0.95).item())
    logger.info(
        "%s: Δxy std=%.4g | Δt median=%.4g min=%.4g max=%.4g | |vel| median=%.4g p95=%.4g",
        tag,
        dxdy_std,
        dt_median,
        dt_min,
        dt_max,
        speed_median,
        speed_p95,
    )


def _prepare_eval_batch(
    sequences: torch.Tensor,
    *,
    prep_cfg: DiffusionEvalPrepConfig,
    tag: str,
    reference_dt: Optional[float] = None,
) -> tuple[torch.Tensor, Optional[float]]:
    seq = sequences.clone()
    if prep_cfg.force_deltas_when_missing_dt and seq.size(-1) < 3:
        seq = _positions_to_deltas_xy(seq)
    seq = _ensure_dt_channel(seq, default_dt=1.0 / max(seq.size(1), 1))
    dt_value = reference_dt
    if prep_cfg.clamp_delta_t:
        dt_value = dt_value if dt_value is not None else prep_cfg.clamp_value
        if dt_value is None:
            dt_value = _estimate_dt_value(seq, prep_cfg.clamp_percentile)
        seq = _clamp_dt_channel(seq, dt_value)
    if prep_cfg.log_batch_stats:
        _describe_sequence_batch(tag, seq)
    return seq, dt_value


@torch.no_grad()
def evaluate(
    model: nn.Module,
    data_loader,
    schedule,
    *,
    device: torch.device,
    in_channels: int,
    cond_dim: int,
    objective: str,
    stats_dataset=None,
    time_weights: Optional[torch.Tensor] = None,
) -> float:
    model.eval()
    total_loss = 0.0
    total_weight = 0.0
    mask_total = 0.0
    target_std_total = 0.0
    batches = 0
    obj = objective.lower().strip()
    for batch in data_loader:
        raw_sequences = batch["sequences"].to(device)
        raw_features = batch["features"].to(device)
        source_dataset = getattr(data_loader, "dataset", None)
        if stats_dataset is not None and source_dataset is not None and source_dataset is not stats_dataset:
            denorm_seq = source_dataset.denormalize_sequences(raw_sequences)
            norm_seq = stats_dataset.normalize_sequences(denorm_seq)
            raw_sequences = norm_seq
            denorm_feat = source_dataset.denormalize_features(raw_features)
            norm_feat = stats_dataset.normalize_features(denorm_feat)
            raw_features = norm_feat
        sequences = _prepare_sequences(raw_sequences, target_channels=in_channels)
        features = raw_features
        mask = batch["mask"].to(device)
        cond = _prepare_features(features, cond_dim)
        timesteps = torch.randint(0, schedule.timesteps, (sequences.size(0),), device=device)
        noise = torch.randn_like(sequences)
        xt, noise = q_sample(schedule, sequences, timesteps, noise=noise)
        alpha, sigma = schedule.coefficients(timesteps, device=device)
        if obj == "epsilon":
            target = noise
        else:
            target = compute_v(sequences, noise, alpha, sigma)

        preds = model(xt.permute(0, 2, 1), timesteps, cond=cond, mask=mask)
        preds = preds.permute(0, 2, 1)
        loss_weights = None
        if time_weights is not None:
            loss_weights = time_weights.to(device=preds.device, dtype=preds.dtype)
        loss = masked_mse(preds, target, mask, weights=loss_weights)
        weight = float(mask.sum().item() * sequences.shape[-1])
        total_loss += float(loss.item()) * max(weight, 1.0)
        total_weight += max(weight, 1.0)
        mask_total += mask.sum().item()
        target_std_total += float(target.std().item())
        batches += 1
    if total_weight == 0.0:
        return 0.0
    avg_mask = mask_total / batches if batches else 0.0
    avg_target_std = target_std_total / batches if batches else 0.0
    logger.info(
        "Validation stats: avg_mask_sum=%.1f avg_target_std=%.4f",
        avg_mask,
        avg_target_std,
    )
    return total_loss / total_weight


def _sample_positive_sequences(source_dataset, limit: int, rng: Optional[np.random.Generator] = None) -> list[torch.Tensor]:
    if limit <= 0:
        return []
    reservoir: list[torch.Tensor] = []
    seen = 0
    for seq_tensor, _, label in source_dataset.samples:
        if label.item() < 0.5:
            continue
        seq_cpu = seq_tensor.detach().cpu()
        if rng is None:
            reservoir.append(seq_cpu)
            if len(reservoir) >= limit:
                break
            continue
        seen += 1
        if len(reservoir) < limit:
            reservoir.append(seq_cpu)
        else:
            idx = int(rng.integers(0, seen))
            if idx < limit:
                reservoir[idx] = seq_cpu
    return reservoir


def _diffusion_classifier_metrics(
    dataset,
    sampler: DiffusionSampler,
    *,
    samples: int,
    seq_len: int,
    steps: int,
    seed: int,
    real_dataset=None,
    real_label: str = "val",
    prep_cfg: DiffusionEvalPrepConfig,
    timing_sampler: Optional[TimingSampler] = None,
    length_values: Optional[np.ndarray] = None,
    match_length: bool = False,
    path_scale_min: float = 0.05,
    path_scale_max: float = 4.0,
) -> dict[str, float]:
    return _diffusion_classifier_metrics_with_val(
        dataset,
        real_dataset=real_dataset or dataset,
        sampler=sampler,
        samples=samples,
        seq_len=seq_len,
        steps=steps,
        seed=seed,
        real_label=real_label,
        prep_cfg=prep_cfg,
        timing_sampler=timing_sampler,
        length_values=length_values,
        match_length=match_length,
        path_scale_min=path_scale_min,
        path_scale_max=path_scale_max,
    )


def _diffusion_classifier_metrics_with_val(
    dataset,
    *,
    real_dataset,
    sampler: DiffusionSampler,
    samples: int,
    seq_len: int,
    steps: int,
    seed: int,
    real_label: str,
    prep_cfg: DiffusionEvalPrepConfig,
    timing_sampler: Optional[TimingSampler],
    length_values: Optional[np.ndarray] = None,
    match_length: bool = False,
    path_scale_min: float = 0.05,
    path_scale_max: float = 4.0,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    match_length = bool(match_length) and length_values is not None and np.size(length_values) > 0
    length_values_np = (
        np.asarray(length_values, dtype=np.int64) if match_length else None
    )

    real_sequences = _sample_positive_sequences(real_dataset, samples, rng=rng)
    if len(real_sequences) < 40:
        return {}
    num_samples = min(samples, len(real_sequences))

    real_batch = torch.stack(real_sequences[:num_samples])
    real_denorm = real_dataset.denormalize_sequences(real_batch)
    real_prep_cfg = prep_cfg
    if timing_sampler is not None and prep_cfg.clamp_delta_t:
        real_prep_cfg = DiffusionEvalPrepConfig(
            force_deltas_when_missing_dt=prep_cfg.force_deltas_when_missing_dt,
            clamp_delta_t=False,
            clamp_percentile=prep_cfg.clamp_percentile,
            clamp_value=prep_cfg.clamp_value,
            log_batch_stats=prep_cfg.log_batch_stats,
        )
    real_ready, dt_value = _prepare_eval_batch(
        real_denorm,
        prep_cfg=real_prep_cfg,
        tag=f"REAL ({real_label})",
    )

    steps = max(1, min(steps, sampler.schedule.timesteps))
    fake = sampler.sample(num_samples, seq_len, steps=steps)
    target_dim = dataset.samples[0][0].size(-1) if dataset.samples else fake.size(-1)
    if fake.size(-1) < target_dim:
        pad_dim = target_dim - fake.size(-1)
        dt_value = 1.0 / max(seq_len, 1)
        pad = torch.zeros(fake.size(0), fake.size(1), pad_dim, device=fake.device, dtype=fake.dtype)
        pad[..., 0] = dt_value
        fake = torch.cat([fake, pad], dim=-1)
    elif fake.size(-1) > target_dim:
        fake = fake[..., :target_dim]
    fake_denorm = dataset.denormalize_sequences(fake.cpu())
    if timing_sampler is not None:
        fake_denorm = timing_sampler.assign(fake_denorm)
        prep_no_clamp = DiffusionEvalPrepConfig(
            force_deltas_when_missing_dt=prep_cfg.force_deltas_when_missing_dt,
            clamp_delta_t=False,
            clamp_percentile=prep_cfg.clamp_percentile,
            clamp_value=None,
            log_batch_stats=prep_cfg.log_batch_stats,
        )
        fake_ready, _ = _prepare_eval_batch(
            fake_denorm,
            prep_cfg=prep_no_clamp,
            tag="FAKE",
        )
    else:
        if not match_length:
            fake_denorm = match_time_channel(fake_denorm, real_ready)
        clamp_target = dt_value if prep_cfg.clamp_delta_t else None
        fake_ready, _ = _prepare_eval_batch(
            fake_denorm,
            prep_cfg=prep_cfg,
            tag="FAKE",
            reference_dt=clamp_target,
        )

    real_masks = infer_mask_from_deltas(real_ready).bool()
    fake_masks = infer_mask_from_deltas(fake_ready).bool()
    eval_real_masks = real_masks.clone()
    eval_fake_masks = fake_masks.clone()

    if match_length and length_values_np is not None and length_values_np.size > 0:
        seq_len_total = real_ready.size(1)
        sampled_idx = rng.integers(0, length_values_np.size, size=num_samples)
        sampled_lengths = length_values_np[sampled_idx]
        eval_real_masks.zero_()
        eval_fake_masks.zero_()
        for i, target_len in enumerate(sampled_lengths):
            target = int(np.clip(target_len, 1, seq_len_total))
            real_valid = int(real_masks[i].sum().item())
            fake_valid = int(fake_masks[i].sum().item())
            use_real = max(1, min(target, real_valid if real_valid > 0 else seq_len_total))
            use_fake = max(1, min(target, fake_valid if fake_valid > 0 else seq_len_total))
            eval_real_masks[i, :use_real] = True
            eval_fake_masks[i, :use_fake] = True

    real_lengths = _sequence_path_lengths(real_ready, eval_real_masks)
    fake_ready, scales = _rescale_sequences_to_length(
        fake_ready,
        eval_fake_masks,
        real_lengths,
        min_scale=path_scale_min,
        max_scale=path_scale_max,
    )

    if prep_cfg.log_batch_stats:
        _describe_sequence_batch(f"REAL ({real_label}) [eval]", real_ready, eval_real_masks)
        _describe_sequence_batch("FAKE [eval]", fake_ready, eval_fake_masks)
        if scales is not None:
            scale_cpu = scales.detach().cpu()
            logger.info(
                "FAKE path-length scales: median=%.3f p95=%.3f",
                float(torch.quantile(scale_cpu, torch.tensor(0.5))),
                float(torch.quantile(scale_cpu, torch.tensor(0.95))),
            )

    real_features: list[torch.Tensor] = []
    for seq, mask_tensor in zip(real_ready, eval_real_masks):
        if mask_tensor.any():
            feat = compute_features_from_sequence(seq, mask=mask_tensor)
            feat = dataset.normalize_features(feat)
            real_features.append(feat)
    if not real_features:
        return {}
    real_np = torch.stack(real_features).numpy()

    fake_features: list[torch.Tensor] = []
    for seq, mask_tensor in zip(fake_ready, eval_fake_masks):
        if mask_tensor.any():
            feat = compute_features_from_sequence(seq, mask=mask_tensor)
            feat = dataset.normalize_features(feat)
            fake_features.append(feat)
    if not fake_features:
        return {}
    fake_np = torch.stack(fake_features).numpy()

    delta = fake_np.mean(axis=0) - real_np.mean(axis=0)
    delta_abs = np.abs(delta)
    feature_delta_mean = float(delta_abs.mean())
    feature_delta_max = float(delta_abs.max())

    _log_feature_stats(real_np, fake_np, real_label)

    feature_dim = min(real_np.shape[1], fake_np.shape[1])
    real_np = real_np[:num_samples, :feature_dim]
    fake_np = fake_np[:num_samples, :feature_dim]

    if np.allclose(real_np, fake_np, atol=1e-6):
        return {
            "c2st_accuracy": 0.5,
            "c2st_auc": 0.5,
            "feature_delta_mean": feature_delta_mean,
            "feature_delta_max": feature_delta_max,
        }

    rng = np.random.default_rng(seed)
    min_count = min(real_np.shape[0], fake_np.shape[0])
    if min_count < 40:
        return {}
    if real_np.shape[0] > min_count:
        idx = rng.choice(real_np.shape[0], size=min_count, replace=False)
        real_np = real_np[idx]
    if fake_np.shape[0] > min_count:
        fake_np = fake_np[:min_count]

    X = np.concatenate([real_np, fake_np], axis=0)
    y = np.concatenate([np.zeros(real_np.shape[0]), np.ones(fake_np.shape[0])], axis=0)
    if X.shape[0] < 80:
        return {}

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        stratify=y,
        random_state=seed,
    )
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    clf.fit(X_train, y_train)
    accuracy = float(clf.score(X_test, y_test))
    proba = clf.predict_proba(X_test)[:, 1]
    auc = float(roc_auc_score(y_test, proba))
    return {
        "c2st_accuracy": accuracy,
        "c2st_auc": auc,
        "feature_delta_mean": feature_delta_mean,
        "feature_delta_max": feature_delta_max,
    }


def _log_feature_stats(real_np: np.ndarray, fake_np: np.ndarray, real_label: str) -> None:
    if real_np.size == 0 or fake_np.size == 0:
        return
    real_mean = real_np.mean(axis=0)
    fake_mean = fake_np.mean(axis=0)
    real_std = real_np.std(axis=0)
    fake_std = fake_np.std(axis=0)
    delta = fake_mean - real_mean
    feature_names = [spec.name for spec in FEATURE_SPECS]
    topk = np.argsort(np.abs(delta))[-5:][::-1]
    parts = []
    for idx in topk:
        name = feature_names[idx] if idx < len(feature_names) else f"feat_{idx}"
        parts.append(
            f"{name}: μ_real={real_mean[idx]:+.3f}, μ_fake={fake_mean[idx]:+.3f}, Δμ={delta[idx]:+.3f}, σ_real={real_std[idx]:+.3f}, σ_fake={fake_std[idx]:+.3f}"
        )
    logger.info("C2ST (%s real) top |Δμ| features: %s", real_label, "; ".join(parts))


def _compute_sample_stats(
    sampler: DiffusionSampler,
    dataset,
    *,
    sample_count: int,
    seq_len: int,
    steps: int,
    timing_sampler: Optional[TimingSampler] = None,
    path_scale_min: float = 0.05,
    path_scale_max: float = 4.0,
) -> dict[str, object]:
    sample_count = min(sample_count, len(dataset))
    if sample_count <= 0:
        return {}
    real_sequences = _sample_positive_sequences(dataset, sample_count)
    if not real_sequences:
        return {}
    num = min(sample_count, len(real_sequences))
    real_batch = torch.stack(real_sequences[:num])
    real_norm_std = real_batch.std(dim=(0, 1))
    real_denorm = dataset.denormalize_sequences(real_batch)
    real_denorm_std = real_denorm.std(dim=(0, 1))

    fake = sampler.sample(num, seq_len, steps=steps).detach().cpu()
    target_dim = dataset.samples[0][0].size(-1) if dataset.samples else fake.size(-1)
    if fake.size(-1) < target_dim:
        pad_dim = target_dim - fake.size(-1)
        pad = torch.zeros(fake.size(0), fake.size(1), pad_dim)
        pad[..., 0] = 1.0 / max(seq_len, 1)
        fake = torch.cat([fake, pad], dim=-1)
    fake_norm = fake
    if timing_sampler is not None:
        fake_denorm = dataset.denormalize_sequences(fake_norm)
        fake_denorm = timing_sampler.assign(fake_denorm)
    else:
        fake_norm = match_time_channel(fake_norm, real_batch)
        fake_denorm = dataset.denormalize_sequences(fake_norm)
    real_masks = infer_mask_from_deltas(real_denorm).bool()
    fake_masks = infer_mask_from_deltas(fake_denorm).bool()
    real_lengths = _sequence_path_lengths(real_denorm, real_masks)
    fake_denorm, _ = _rescale_sequences_to_length(
        fake_denorm,
        fake_masks,
        real_lengths,
        min_scale=path_scale_min,
        max_scale=path_scale_max,
    )
    fake = dataset.normalize_sequences(fake_denorm)
    fake_norm_std = fake.std(dim=(0, 1))
    fake_denorm_std = fake_denorm.std(dim=(0, 1))

    def _tensor_to_list(t: torch.Tensor) -> list[float]:
        return [float(x) for x in t.tolist()]

    return {
        "sample_norm_std": _tensor_to_list(fake_norm_std),
        "sample_norm_std_mean": float(fake_norm_std.mean().item()),
        "sample_denorm_std": _tensor_to_list(fake_denorm_std),
        "sample_denorm_std_mean": float(fake_denorm_std.mean().item()),
        "real_norm_std": _tensor_to_list(real_norm_std),
        "real_norm_std_mean": float(real_norm_std.mean().item()),
        "real_denorm_std": _tensor_to_list(real_denorm_std),
        "real_denorm_std_mean": float(real_denorm_std.mean().item()),
    }


def _write_summary(summary_path: Path, payload: dict) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _min_snr_weight(log_snr: torch.Tensor, gamma: float) -> torch.Tensor:
    gamma = float(gamma)
    if gamma <= 0:
        return torch.ones_like(log_snr)
    snr = log_snr.exp()
    weight = (gamma + snr) / (snr + 1)
    return weight


def save_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    ema: EMAModel,
    optimizer: torch.optim.Optimizer,
    scaler: amp.GradScaler,
    epoch: int,
    global_step: int,
    config: Optional[dict] = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "global_step": global_step,
            "model": model.state_dict(),
            "ema": ema.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "config": config or {},
        },
        path,
    )
    logger.info("Saved checkpoint to %s", path)


@hydra.main(version_base=None, config_path="../../conf", config_name="diffusion/base")
def main(cfg: DictConfig) -> None:
    OmegaConf.set_struct(cfg, False)
    experiment_cfg = _load_experiment_config(cfg)
    torch.manual_seed(experiment_cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(experiment_cfg.seed)
        torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Running diffusion training on device %s", device)

    data_cfg = experiment_cfg.data
    model_cfg = experiment_cfg.model
    diffusion_cfg = experiment_cfg.diffusion
    training_cfg = experiment_cfg.training

    train_loader = create_dataloader(
        data_cfg,
        split=data_cfg.train_split,
        max_gestures=data_cfg.max_train_gestures,
        shuffle=True,
    )
    if len(train_loader.dataset) == 0:
        logger.error("Training dataset is empty. Check dataset configuration.")
        return

    val_loader = None
    if data_cfg.val_split:
        try:
            val_loader = create_dataloader(
                data_cfg,
                split=data_cfg.val_split,
                max_gestures=data_cfg.max_val_gestures,
                shuffle=False,
            )
        except ValueError:
            val_loader = None

    train_length_values: Optional[np.ndarray] = None
    val_length_values: Optional[np.ndarray] = None
    time_weight_map: Optional[torch.Tensor] = None
    time_weight_cache: dict[torch.dtype, torch.Tensor] = {}

    analyze_masks = (
        training_cfg.balance_time_steps
        or training_cfg.eval_classifier
        or training_cfg.eval_prep.match_length_distribution
    )
    if analyze_masks:
        mask_weights, lengths_np = _analyze_sequence_masks(
            train_loader.dataset,
            clamp=training_cfg.time_weight_max,
        )
        train_length_values = lengths_np
        if training_cfg.balance_time_steps:
            time_weight_map = mask_weights.to(device=device, dtype=torch.float32).view(1, -1, 1)
    if val_loader is not None and (
        training_cfg.eval_classifier or training_cfg.eval_prep.match_length_distribution
    ):
        _, lengths_np = _analyze_sequence_masks(
            val_loader.dataset,
            clamp=training_cfg.time_weight_max,
        )
        val_length_values = lengths_np

    seq_target_std = train_loader.dataset.get_sequence_stats()[1]
    if data_cfg.normalize_sequences:
        scale_target_std = torch.ones_like(seq_target_std)
    else:
        scale_target_std = seq_target_std
    if scale_target_std.numel() > model_cfg.in_channels:
        scale_target_std = scale_target_std[: model_cfg.in_channels]
    elif scale_target_std.numel() < model_cfg.in_channels:
        pad = model_cfg.in_channels - scale_target_std.numel()
        scale_target_std = torch.cat([scale_target_std, scale_target_std.new_ones(pad)])
    if training_cfg.scale_reg_channels:
        mask = torch.zeros_like(scale_target_std, dtype=torch.bool)
        for idx in training_cfg.scale_reg_channels:
            if 0 <= idx < mask.numel():
                mask[idx] = True
        scale_target_std = torch.where(mask, scale_target_std, torch.ones_like(scale_target_std))
    scale_target_std = scale_target_std.to(device)

    model = UNet1D(model_cfg).to(device)
    schedule = build_schedule(diffusion_cfg, device=device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_cfg.lr,
        betas=training_cfg.betas,
        weight_decay=training_cfg.weight_decay,
    )
    grad_scaler_device = "cuda" if device.type == "cuda" else "cpu"
    autocast_device_type = grad_scaler_device
    autocast_enabled = bool(training_cfg.amp and device.type == "cuda")
    scaler = amp.GradScaler(device=grad_scaler_device, enabled=autocast_enabled)
    ema = EMAModel(model, decay=training_cfg.ema_decay).to(device)
    ema_sampler = DiffusionSampler(
        ema.shadow,
        schedule,
        device=device,
        cond_dim=model_cfg.cond_dim,
        in_channels=model_cfg.in_channels,
        self_condition=model_cfg.self_condition,
        objective=training_cfg.objective,
    )
    timing_sampler = _maybe_build_timing_sampler(training_cfg.timing_eval, data_cfg, device)
    classifier_logger = None
    if training_cfg.eval_classifier:
        clf_path = Path(to_absolute_path(training_cfg.checkpoint_dir)) / "diffusion_classifier_metrics.csv"
        classifier_logger = CSVMetricLogger(
            clf_path,
            fieldnames=[
                "epoch",
                "c2st_val_accuracy",
                "c2st_val_auc",
                "c2st_train_accuracy",
                "c2st_train_auc",
            ],
        )

    best_val_loss = float("inf")
    last_val_loss: Optional[float] = None
    val_metrics: dict[str, float] = {}
    train_metrics: dict[str, float] = {}

    objective = training_cfg.objective.lower()
    use_min_snr = training_cfg.min_snr_gamma not in (None, 0) and objective == "epsilon"
    if training_cfg.min_snr_gamma not in (None, 0) and not use_min_snr:
        logger.warning(
            "Min-SNR weighting (gamma=%.3f) disabled for objective '%s'.",
            float(training_cfg.min_snr_gamma),
            objective,
        )

    global_step = 0
    for epoch in range(1, training_cfg.epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_weight = 0.0
        dir_weight = float(training_cfg.direction_change_reg_weight)
        if dir_weight > 0 and training_cfg.direction_change_reg_warmup_epochs > 0:
            warm = max(1, training_cfg.direction_change_reg_warmup_epochs)
            dir_weight *= min(epoch / warm, 1.0)
        curv_weight = float(training_cfg.curvature_reg_weight)
        if curv_weight > 0 and training_cfg.curvature_reg_warmup_epochs > 0:
            warm = max(1, training_cfg.curvature_reg_warmup_epochs)
            curv_weight *= min(epoch / warm, 1.0)

        for batch_idx, batch in enumerate(train_loader, start=1):
            sequences = _prepare_sequences(batch["sequences"].to(device), target_channels=model_cfg.in_channels)
            mask = batch["mask"].to(device)
            mask_bool = mask.bool()
            cond = _prepare_features(batch["features"].to(device), model_cfg.cond_dim)

            sequences = apply_default_augmentations(
                sequences,
                mask=mask_bool,
                time_stretch=float(data_cfg.time_stretch),
                jitter_std=float(data_cfg.jitter_std),
                mirror_prob=float(data_cfg.mirror_prob),
            )

            timesteps = torch.randint(0, schedule.timesteps, (sequences.size(0),), device=device)
            noise = torch.randn_like(sequences)

            with amp.autocast(device_type=autocast_device_type, enabled=autocast_enabled):
                xt, noise = q_sample(schedule, sequences, timesteps, noise=noise)
                alpha, sigma = schedule.coefficients(timesteps, device=device)

                preds = model(xt.permute(0, 2, 1), timesteps, cond=cond, mask=mask_bool)
                preds = preds.permute(0, 2, 1)

                if objective == "epsilon":
                    target = noise
                    x0_hat = x0_from_eps(xt, preds, alpha, sigma)
                else:
                    target = compute_v(sequences, noise, alpha, sigma)
                    x0_hat = x0_from_v(xt, preds, alpha, sigma)

                loss_weights = None
                if use_min_snr:
                    log_snr = schedule.log_snr_at(timesteps, device=device)
                    snr_weights = _min_snr_weight(log_snr, training_cfg.min_snr_gamma)
                    loss_weights = snr_weights.view(-1, 1, 1)
                if time_weight_map is not None:
                    tw = time_weight_cache.get(preds.dtype)
                    if tw is None:
                        tw = time_weight_map
                        if tw.dtype != preds.dtype:
                            tw = tw.to(dtype=preds.dtype)
                        time_weight_cache[preds.dtype] = tw
                    loss_weights = tw if loss_weights is None else loss_weights * tw
                mask_float = mask_bool.to(dtype=preds.dtype)
                err_sq = (preds - target) ** 2
                err_sq_unweighted = err_sq * mask_float.unsqueeze(-1)
                if loss_weights is not None:
                    weights = loss_weights
                    while weights.ndim < err_sq.ndim:
                        weights = weights.unsqueeze(-1)
                    err_sq = err_sq * weights
                err_sq = err_sq * mask_float.unsqueeze(-1)
                denom = (mask_float.sum() * preds.size(-1)).clamp_min(1.0)
                loss = err_sq.sum() / denom
                if training_cfg.log_loss_details and training_cfg.log_interval > 0 and (global_step % training_cfg.log_interval == 0):
                    per_t = err_sq_unweighted.sum(dim=(0, 2)) / (
                        mask_float.sum(dim=0).clamp_min(1.0) * preds.size(-1)
                    )
                    per_b = err_sq_unweighted.sum(dim=(1, 2)) / (
                        mask_float.sum(dim=1).clamp_min(1.0) * preds.size(-1)
                    )
                    top_t_k = int(min(3, per_t.numel()))
                    top_b_k = int(min(3, per_b.numel()))
                    if top_t_k > 0:
                        top_t_val, top_t_idx = torch.topk(per_t, k=top_t_k)
                        top_t = ", ".join(
                            f"t={int(idx)}:{float(val):.4f}"
                            for idx, val in zip(top_t_idx.tolist(), top_t_val.tolist())
                        )
                        logger.info("Loss by timestep (top): %s", top_t)
                    if top_b_k > 0:
                        top_b_val, top_b_idx = torch.topk(per_b, k=top_b_k)
                        top_b = ", ".join(
                            f"b={int(idx)}:{float(val):.4f}"
                            for idx, val in zip(top_b_idx.tolist(), top_b_val.tolist())
                        )
                        logger.info("Loss by sample (top): %s", top_b)
                if training_cfg.scale_reg_weight > 0:
                    std_hat = _masked_channel_std(x0_hat, mask_bool)
                    target_std = scale_target_std.to(device=std_hat.device, dtype=std_hat.dtype)
                    scale_penalty = torch.mean((std_hat - target_std) ** 2)
                    loss = loss + training_cfg.scale_reg_weight * scale_penalty
                if training_cfg.path_length_reg_weight > 0:
                    target_lengths = _sequence_path_lengths(sequences, mask_bool)
                    pred_lengths = _sequence_path_lengths(x0_hat, mask_bool)
                    path_length_penalty = F.mse_loss(pred_lengths, target_lengths)
                    loss = loss + training_cfg.path_length_reg_weight * path_length_penalty
                if dir_weight > 0:
                    direction_penalty = _direction_change_penalty(x0_hat, sequences, mask_bool)
                    loss = loss + dir_weight * direction_penalty
                if curv_weight > 0:
                    curvature_penalty = _curvature_penalty(x0_hat, sequences, mask_bool)
                    loss = loss + curv_weight * curvature_penalty
                loss_item = float(loss.detach())
                if training_cfg.loss_skip_threshold is not None:
                    if not math.isfinite(loss_item) or loss_item > training_cfg.loss_skip_threshold:
                        logger.warning(
                            "Skipping batch at step %d due to anomalous loss %.4f", global_step, loss_item
                        )
                        optimizer.zero_grad(set_to_none=True)
                        continue
            if global_step % training_cfg.log_interval == 0:
                print(
                    f"debug step={global_step} mask_sum={mask_bool.sum().item():.0f} target_std={target.std().item():.4f}"
                )
                norm_std = _masked_channel_std(sequences, mask_bool).detach().cpu().tolist()
                logger.info("Normalized channel std: %s", ", ".join(f"{s:.4f}" for s in norm_std))

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if training_cfg.grad_clip > 0:
                clip_grad_norm_(model.parameters(), training_cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            ema.update(model)

            mask_weight = float(mask_bool.sum().item() * sequences.shape[-1])
            epoch_loss += float(loss.item()) * max(mask_weight, 1.0)
            epoch_weight += max(mask_weight, 1.0)
            global_step += 1

            if global_step % training_cfg.log_interval == 0:
                logger.info(
                    "Epoch %d | Step %d | loss %.6f",
                    epoch,
                    global_step,
                    loss.item(),
                )

        if epoch_weight > 0:
            train_loss = epoch_loss / epoch_weight
        else:
            train_loss = 0.0
        logger.info("Epoch %d completed. Train loss: %.6f", epoch, train_loss)

        if val_loader and (epoch % training_cfg.eval_interval == 0):
            val_loss = evaluate(
                ema.shadow,
                val_loader,
                schedule,
                device=device,
                in_channels=model_cfg.in_channels,
                cond_dim=model_cfg.cond_dim,
                objective=training_cfg.objective,
                stats_dataset=train_loader.dataset,
                time_weights=time_weight_map,
            )
            last_val_loss = val_loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
            logger.info("Epoch %d validation loss (EMA): %.6f", epoch, val_loss)
            classifier_metrics: dict[str, float] = {}
            train_real_metrics: dict[str, float] = {}
            if training_cfg.eval_classifier:
                classifier_metrics = _diffusion_classifier_metrics(
                    dataset=train_loader.dataset,
                    real_dataset=val_loader.dataset,
                    real_label="val",
                    sampler=ema_sampler,
                    samples=training_cfg.classifier_samples,
                    seq_len=data_cfg.sequence_length,
                    steps=training_cfg.classifier_steps,
                    seed=experiment_cfg.seed + epoch,
                    prep_cfg=training_cfg.eval_prep,
                    timing_sampler=timing_sampler,
                    length_values=val_length_values,
                    match_length=training_cfg.eval_prep.match_length_distribution,
                    path_scale_min=training_cfg.eval_path_scale_min,
                    path_scale_max=training_cfg.eval_path_scale_max,
                )
                train_real_metrics = _diffusion_classifier_metrics(
                    dataset=train_loader.dataset,
                    real_dataset=train_loader.dataset,
                    real_label="train",
                    sampler=ema_sampler,
                    samples=training_cfg.classifier_samples,
                    seq_len=data_cfg.sequence_length,
                    steps=training_cfg.classifier_steps,
                    seed=experiment_cfg.seed + epoch,
                    prep_cfg=training_cfg.eval_prep,
                    timing_sampler=timing_sampler,
                    length_values=train_length_values,
                    match_length=training_cfg.eval_prep.match_length_distribution,
                    path_scale_min=training_cfg.eval_path_scale_min,
                    path_scale_max=training_cfg.eval_path_scale_max,
                )
                if classifier_metrics:
                    logger.info(
                        "Epoch %d diffusion C2ST accuracy=%.4f auc=%.4f",
                        epoch,
                        classifier_metrics.get("c2st_accuracy", float("nan")),
                        classifier_metrics.get("c2st_auc", float("nan")),
                    )
                    val_metrics = classifier_metrics
                if train_real_metrics:
                    logger.info(
                        "Epoch %d diffusion train-real C2ST accuracy=%.4f auc=%.4f",
                        epoch,
                        train_real_metrics.get("c2st_accuracy", float("nan")),
                        train_real_metrics.get("c2st_auc", float("nan")),
                    )
                    train_metrics = train_real_metrics
                if classifier_logger is not None and classifier_metrics:
                    payload = {
                        "epoch": epoch,
                        "c2st_val_accuracy": classifier_metrics["c2st_accuracy"],
                        "c2st_val_auc": classifier_metrics["c2st_auc"],
                    }
                    if train_real_metrics:
                        payload.update(
                            {
                                "c2st_train_accuracy": train_real_metrics["c2st_accuracy"],
                                "c2st_train_auc": train_real_metrics["c2st_auc"],
                            }
                        )
                    classifier_logger.log(payload)

        if training_cfg.summary_path:
            summary_path = Path(to_absolute_path(training_cfg.summary_path))
            summary = {
                "experiment_name": experiment_cfg.experiment_name,
                "last_val_loss": float(last_val_loss) if last_val_loss is not None else None,
                "best_val_loss": (float(best_val_loss) if best_val_loss != float("inf") else None),
                "val_c2st_accuracy": val_metrics.get("c2st_accuracy"),
                "val_c2st_auc": val_metrics.get("c2st_auc"),
                "val_feature_delta_mean": val_metrics.get("feature_delta_mean"),
                "train_c2st_accuracy": train_metrics.get("c2st_accuracy"),
                "train_c2st_auc": train_metrics.get("c2st_auc"),
            }
            sample_stats = {}
            if training_cfg.sample_eval_count > 0:
                sample_stats = _compute_sample_stats(
                    ema_sampler,
                    train_loader.dataset,
                    sample_count=training_cfg.sample_eval_count,
                    seq_len=data_cfg.sequence_length,
                    steps=min(training_cfg.sample_eval_steps, training_cfg.classifier_steps),
                    timing_sampler=timing_sampler,
                    path_scale_min=training_cfg.eval_path_scale_min,
                    path_scale_max=training_cfg.eval_path_scale_max,
                )
            summary.update(sample_stats)
            _write_summary(summary_path, summary)
            logger.info("Saved diffusion summary to %s", summary_path)

        if training_cfg.checkpoint_interval > 0 and (
            epoch % training_cfg.checkpoint_interval == 0 or epoch == training_cfg.epochs
        ):
            ckpt_dir = Path(to_absolute_path(training_cfg.checkpoint_dir))
            ckpt_path = ckpt_dir / f"{experiment_cfg.experiment_name}_epoch{epoch:03d}.pt"
            config_payload = {
                "model": asdict(model_cfg),
                "diffusion": asdict(diffusion_cfg),
                "training": {"objective": training_cfg.objective},
            }
            save_checkpoint(
                ckpt_path,
                model=model,
                ema=ema,
                optimizer=optimizer,
                scaler=scaler,
                epoch=epoch,
                global_step=global_step,
                config=config_payload,
            )


if __name__ == "__main__":
    main()
