"""Hydra-driven training entry point for GAN experiments."""
from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import shutil
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import hydra
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from torch import nn, optim
from hydra.utils import to_absolute_path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

from data.dataset import GestureDataset, GestureDatasetConfig
from models.discriminator import DiscriminatorConfig, GestureDiscriminator
from models.generator import ConditionalGenerator, GeneratorConfig
from models.gan_lstm import LSTMGenerator, LSTMDiscriminator
from models.seq2seq import Seq2SeqEncoder, Seq2SeqGenerator
from eval.sigma_log_baseline import run_baseline
from diffusion.utils import match_time_channel
from train.config_schemas import (
    DataConfig,
    GanExperimentConfig,
    GanModelConfig,
    GanTrainingConfig,
)
from train.replay_buffer import ReplayBuffer
from train.train_detector import load_detector_config, run_detector_training
from diffusion.sample import generate_diffusion_samples
from utils.housekeeping import tidy_checkpoint_artifacts
from utils.logging import CSVMetricLogger, LoggingConfig, experiment_logger, write_summary_json, log_wandb_artifact
from utils.plotting import plot_metric_trends, generate_replay_vs_real_plots
from utils.eval import feature_distribution_metrics, sequence_diversity_metric
from features import (
    compute_features_from_sequence,
    sigma_lognormal_features_torch,
)

logger = logging.getLogger(__name__)

_FEATURE_CACHE_MAX = 4096
_feature_cache: "OrderedDict[str, torch.Tensor]" = OrderedDict()


def _sequence_cache_key(sequence: torch.Tensor) -> str:
    arr = sequence.detach().to(dtype=torch.float32, device="cpu").contiguous().numpy()
    return hashlib.sha1(arr.tobytes()).hexdigest()


def _feature_cache_get(key: str) -> Optional[torch.Tensor]:
    cached = _feature_cache.get(key)
    if cached is None:
        return None
    _feature_cache.move_to_end(key)
    return cached


def _feature_cache_put(key: str, value: torch.Tensor) -> None:
    if key in _feature_cache:
        _feature_cache[key] = value
        _feature_cache.move_to_end(key)
    else:
        _feature_cache[key] = value
    while len(_feature_cache) > _FEATURE_CACHE_MAX:
        _feature_cache.popitem(last=False)


def _select_feature_device(device_pref: str, default: torch.device) -> torch.device:
    pref = (device_pref or "auto").lower()
    if pref == "auto":
        return default
    if pref == "cpu":
        return torch.device("cpu")
    if pref in {"cuda", "gpu"}:
        if torch.cuda.is_available():
            return torch.device("cuda")
        logger.warning("Requested GPU sigma features but CUDA is unavailable; falling back to CPU.")
        return torch.device("cpu")
    logger.warning("Unknown sigma_feature_device '%s'; using default %s", device_pref, default.type)
    return default

def _canonicalize_tensor_sequences(
    sequences: torch.Tensor,
    data_cfg: DataConfig,
) -> torch.Tensor:
    """Project sequences to unit path/ duration when configured."""
    if not data_cfg.canonicalize_path and not data_cfg.canonicalize_duration:
        return sequences
    eps = 1e-6
    seqs = sequences.clone()
    if data_cfg.canonicalize_path:
        path_lengths = torch.linalg.norm(sequences[..., :2], dim=-1).sum(dim=1, keepdim=True).clamp_min(eps)
        seqs[..., :2] = sequences[..., :2] / path_lengths.unsqueeze(-1)
    if data_cfg.canonicalize_duration:
        total_dt = sequences[..., 2].sum(dim=1, keepdim=True).clamp_min(eps)
        seqs[..., 2] = sequences[..., 2] / total_dt
    return seqs


def _deltas_to_positions_xy(sequences: torch.Tensor) -> torch.Tensor:
    return torch.cumsum(sequences[..., :2], dim=1)


def _positions_to_deltas_xy(positions: torch.Tensor) -> torch.Tensor:
    first_step = positions[:, :1, :]
    diffs = positions[:, 1:, :] - positions[:, :-1, :]
    return torch.cat([first_step, diffs], dim=1)


def _position_stats_from_dataset(dataset: GestureDataset, output_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute mean and std of cumulative positions across the dataset."""
    sum_vec = torch.zeros(output_dim, dtype=torch.float64)
    sum_sq = torch.zeros(output_dim, dtype=torch.float64)
    count = 0
    for sequence, _, label in dataset.samples:
        if label.item() < 0.5:
            continue
        deltas = sequence[..., :output_dim].double()
        positions = torch.cumsum(deltas, dim=0)
        sum_vec += positions.sum(dim=0)
        sum_sq += (positions**2).sum(dim=0)
        count += positions.size(0)

    if count == 0:
        mean = torch.zeros(output_dim)
        std = torch.ones(output_dim)
    else:
        mean = (sum_vec / count).float()
        var = (sum_sq / count - mean.double() ** 2).clamp_min(1e-6)
        std = var.sqrt().float()

    return mean.view(1, 1, output_dim), std.view(1, 1, output_dim)


def _reset_generator_weights(generator: torch.nn.Module, experiment_cfg: GanExperimentConfig) -> None:
    device = next(generator.parameters()).device
    architecture = experiment_cfg.model.architecture
    if architecture == "seq2seq":
        fresh = Seq2SeqGenerator(experiment_cfg.model.generator).to(device)
    elif architecture == "lstm":
        fresh = LSTMGenerator(experiment_cfg.model.generator).to(device)
    elif architecture == "tcn":
        fresh = ConditionalGenerator(experiment_cfg.model.generator).to(device)
    else:
        raise ValueError(f"Unsupported generator architecture: {architecture}")
    generator.load_state_dict(fresh.state_dict())


def _sample_noise_sequences(
    reference: torch.Tensor,
    *,
    use_absolute: bool,
    dt_value: Optional[float],
    dt_value_norm: Optional[float],
    canonicalize: bool,
    experiment_cfg: GanExperimentConfig,
) -> torch.Tensor:
    device = reference.device
    batch_size, seq_len, feat_dim = reference.shape
    deltas = torch.randn(batch_size, seq_len, 2, device=device)
    if feat_dim >= 3:
        if dt_value_norm is not None:
            dt = torch.full((batch_size, seq_len, 1), float(dt_value_norm), device=device)
        elif dt_value is not None:
            dt = torch.full((batch_size, seq_len, 1), float(dt_value), device=device)
        else:
            dt = torch.ones(batch_size, seq_len, 1, device=device)
        seq = torch.cat([deltas, dt], dim=-1)
    else:
        seq = deltas

    if use_absolute:
        positions = torch.cumsum(deltas, dim=1)
        return positions
    if canonicalize:
        return _canonicalize_tensor_sequences(seq, experiment_cfg.data)
    return seq


def _apply_sequence_augmentations(
    sequences: torch.Tensor,
    prob: float,
    *,
    max_rotation: float = math.pi,
    scale_jitter: float = 0.5,
    dt_jitter: float = 0.2,
    noise_std: float = 0.05,
) -> torch.Tensor:
    """Apply simple invertible augmentations (rotation + scaling) with probability `prob` per sample."""
    if prob <= 0.0:
        return sequences
    if sequences.dim() < 3 or sequences.size(-1) < 2:
        return sequences
    batch_size = sequences.size(0)
    if batch_size == 0:
        return sequences
    device = sequences.device
    mask = torch.rand(batch_size, device=device) < prob
    if not mask.any():
        return sequences
    aug_sequences = sequences.clone()
    idx = mask.nonzero(as_tuple=False).squeeze(1)
    if idx.numel() == 0:
        return sequences
    angles = (torch.rand(idx.size(0), device=device) - 0.5) * 2.0 * max_rotation
    cos_t = torch.cos(angles).view(-1, 1, 1)
    sin_t = torch.sin(angles).view(-1, 1, 1)
    dx = aug_sequences[idx, :, 0:1]
    dy = aug_sequences[idx, :, 1:2]
    rot_dx = cos_t * dx - sin_t * dy
    rot_dy = sin_t * dx + cos_t * dy
    aug_sequences[idx, :, 0:1] = rot_dx
    aug_sequences[idx, :, 1:2] = rot_dy
    if scale_jitter > 0.0:
        jitter = (torch.rand(idx.size(0), device=device) - 0.5) * 2.0 * scale_jitter
        scales = torch.exp(jitter).view(-1, 1, 1)
        aug_sequences[idx, :, :2] = aug_sequences[idx, :, :2] * scales
    if noise_std > 0.0:
        noise = torch.randn_like(aug_sequences[idx, :, :2]) * noise_std
        aug_sequences[idx, :, :2] = aug_sequences[idx, :, :2] + noise
    if dt_jitter > 0.0 and sequences.size(-1) >= 3:
        dt = aug_sequences[idx, :, 2:3]
        dt_scale = torch.exp((torch.rand(idx.size(0), device=device) - 0.5) * 2.0 * dt_jitter).view(-1, 1, 1)
        aug_sequences[idx, :, 2:3] = dt * dt_scale
    return aug_sequences


class WarmupSequenceEncoder(nn.Module):
    """Simple LSTM encoder that maps full sequences to latent vectors."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        latent_dim: int,
        num_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        effective_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=max(1, num_layers),
            batch_first=True,
            dropout=effective_dropout,
        )
        self.proj = nn.Linear(hidden_dim, latent_dim)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        _, (hidden, _) = self.lstm(sequence)
        last = hidden[-1]
        latent = self.proj(last)
        return latent


def _apply_fixed_dt(
    sequences: torch.Tensor,
    dataset: GestureDataset,
    dt_value: float,
) -> torch.Tensor:
    """Return sequences whose Δt column is clamped to a constant value."""
    if sequences.size(-1) < 3:
        return sequences
    seq_denorm = dataset.denormalize_sequences(sequences)
    seq_fixed = seq_denorm.clone()
    dt_tensor = seq_fixed.new_full(seq_fixed[..., 2].shape, float(dt_value))
    seq_fixed[..., 2] = dt_tensor
    if dataset.config.normalize_sequences:
        return dataset.normalize_sequences(seq_fixed)
    return seq_fixed


def _estimate_fixed_dt(
    dataset: GestureDataset,
    experiment_cfg: GanExperimentConfig,
) -> Optional[float]:
    """Infer a reasonable constant Δt for warm-start/metrics."""
    sampling_rate = _sampling_rate_to_float(experiment_cfg.data.sampling_rate)
    if sampling_rate and sampling_rate > 0:
        return 1.0 / sampling_rate
    if experiment_cfg.data.canonicalize_duration and experiment_cfg.data.sequence_length > 0:
        return 1.0 / float(experiment_cfg.data.sequence_length)

    dt_values: list[torch.Tensor] = []
    for sequence, _, label in dataset.samples:
        if label.item() < 0.5:
            continue
        if sequence.size(-1) < 3:
            continue
        seq_tensor = sequence.unsqueeze(0)
        denorm = dataset.denormalize_sequences(seq_tensor).squeeze(0)
        dt_column = denorm[:, 2]
        positive = dt_column > 0
        if positive.any():
            dt_values.append(dt_column[positive])
        if len(dt_values) >= 128:
            break

    if not dt_values:
        logger.warning("Unable to estimate fixed Δt from dataset; proceeding without Δt clamp.")
        return None

    stacked = torch.cat(dt_values).float()
    return float(stacked.median().item())


def _set_requires_grad(module: torch.nn.Module, flag: bool) -> None:
    for param in module.parameters():
        param.requires_grad = flag


def _theta_start(sequence: torch.Tensor) -> torch.Tensor:
    deltas = sequence[..., :2]
    first = deltas[:, 0, :]
    eps = 1e-6
    return torch.atan2(first[:, 1], torch.where(first[:, 0].abs() < eps, first[:, 0].sign() * eps, first[:, 0]))


def _path_efficiency(sequence: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    deltas = sequence[..., :2]
    displacement = torch.linalg.norm(deltas.sum(dim=1), dim=-1)
    path = torch.linalg.norm(deltas, dim=-1).sum(dim=1).clamp_min(eps)
    return displacement / path


def _canonical_frame(
    sequences: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotate trajectories so the overall displacement aligns with +x."""
    if sequences.size(-1) < 2:
        raise ValueError("Sequences must contain at least two spatial dimensions")

    deltas = sequences[..., :2]
    totals = deltas.sum(dim=1)
    angles = torch.atan2(totals[:, 1], totals[:, 0])
    cos = torch.cos(angles)
    sin = torch.sin(angles)
    zero_mask = totals.norm(dim=1) < eps
    if zero_mask.any():
        cos = cos.masked_fill(zero_mask, 1.0)
        sin = sin.masked_fill(zero_mask, 0.0)

    rot = torch.stack(
        [
            torch.stack([cos, sin], dim=-1),
            torch.stack([-sin, cos], dim=-1),
        ],
        dim=-2,
    )
    rotated_deltas = torch.matmul(deltas, rot)
    rotated_positions = torch.cumsum(rotated_deltas, dim=1)
    return rotated_deltas, rotated_positions


def _progress_along_path(rotated_positions: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    total = rotated_positions[:, -1, 0].abs().clamp_min(eps)
    progress = rotated_positions[..., 0] / total.unsqueeze(-1)
    return progress.clamp(0.0, 1.0)


def _end_segment_weights(progress: torch.Tensor, threshold: float = 0.6) -> torch.Tensor:
    scale = max(1.0 - threshold, 1e-6)
    weights = (progress - threshold) / scale
    return torch.clamp(weights, 0.0, 1.0)


def _mean_abs_heading_change(
    rotated_deltas: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Average absolute change in heading between successive steps."""
    headings = torch.atan2(rotated_deltas[..., 1], rotated_deltas[..., 0].clamp_min(eps))
    delta_headings = headings[:, 1:] - headings[:, :-1]
    delta_headings = torch.atan2(torch.sin(delta_headings), torch.cos(delta_headings))
    magnitude = rotated_deltas[:, 1:, :].norm(dim=-1)
    base_weights = (magnitude > eps).float()
    if weights is not None:
        weights = weights.to(rotated_deltas.device, rotated_deltas.dtype)
        if weights.shape != base_weights.shape:
            weights = weights[..., : base_weights.shape[-1]]
        combined = base_weights * weights
    else:
        combined = base_weights
    weighted = delta_headings.abs() * combined
    denom = combined.sum(dim=1).clamp_min(1.0)
    return weighted.sum(dim=1) / denom


def _lateral_rms(rotated_positions: torch.Tensor, weights: Optional[torch.Tensor] = None) -> torch.Tensor:
    lateral = rotated_positions[..., 1]
    if weights is not None:
        weights = weights.to(rotated_positions.device, rotated_positions.dtype)
        if weights.shape != lateral.shape:
            weights = weights[..., : lateral.shape[-1]]
        weighted_sq = (lateral**2) * weights
        denom = weights.sum(dim=1).clamp_min(1.0)
        return torch.sqrt(weighted_sq.sum(dim=1) / denom + 1e-8)
    return torch.sqrt((lateral**2).mean(dim=1) + 1e-8)


def _jerk_magnitude(sequences: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    deltas = sequences[..., :2]
    dt = sequences[..., 2:3].clamp_min(eps)
    velocity = deltas / dt
    accel = (velocity[:, 1:, :] - velocity[:, :-1, :]) / dt[:, 1:, :]
    jerk = (accel[:, 1:, :] - accel[:, :-1, :]) / dt[:, 2:, :]
    return torch.linalg.norm(jerk, dim=-1)


def _delta_variation(deltas: torch.Tensor) -> torch.Tensor:
    """Mean absolute delta change per axis for each sequence."""
    diffs = deltas[:, 1:, :] - deltas[:, :-1, :]
    return diffs.abs().mean(dim=1)


def _build_experiment_config(cfg: DictConfig) -> tuple[GanExperimentConfig, Dict[str, Any]]:
    cfg_dict: Dict[str, Any] = OmegaConf.to_container(cfg, resolve=True)  # type: ignore[assignment]
    if "experiment" in cfg_dict:
        cfg_dict = cfg_dict["experiment"]

    data_cfg = DataConfig(**cfg_dict["data"])
    model_section = cfg_dict["model"]
    model_cfg = GanModelConfig(
        architecture=model_section.get("architecture", "tcn"),
        generator=GeneratorConfig(**model_section["generator"]),
        discriminator=DiscriminatorConfig(**model_section["discriminator"]),
    )
    training_section = cfg_dict["training"]
    training_cfg = GanTrainingConfig(
        epochs=training_section["epochs"],
        lr_generator=training_section["lr"]["generator"],
        lr_discriminator=training_section["lr"]["discriminator"],
        beta1=training_section["betas"][0],
        beta2=training_section["betas"][1],
        gradient_penalty_weight=training_section["gradient_penalty_weight"],
        discriminator_steps=training_section["discriminator_steps"],
        log_interval=training_section["log_interval"],
        metric_log_interval=training_section.get("metric_log_interval"),
        sample_interval=training_section["sample_interval"],
        replay_buffer_path=training_section.get("replay_buffer_path", "checkpoints/gan/replay_buffer.pt"),
        replay_samples_per_epoch=training_section.get("replay_samples_per_epoch", 256),
        replay_buffer_max_size=training_section.get("replay_buffer_max_size", 5000),
        co_train_detector=training_section.get("co_train_detector", False),
        detector_update_every=training_section.get("detector_update_every", 0),
        detector_config_path=training_section.get("detector_config_path"),
        detector_epochs_per_update=training_section.get("detector_epochs_per_update", 1),
        reconstruction_epochs=training_section.get("reconstruction_epochs", 0),
        reconstruction_loss_weight=training_section.get("reconstruction_loss_weight", 1.0),
        adversarial_type=training_section.get("adversarial_type", "wgan"),
        sigma_eval_enabled=training_section.get("sigma_eval_enabled", False),
        sigma_eval_interval=training_section.get("sigma_eval_interval", 1),
        sigma_eval_samples=training_section.get("sigma_eval_samples", 512),
        sigma_eval_dataset_id=training_section.get("sigma_eval_dataset_id"),
        sigma_eval_max_gestures=training_section.get("sigma_eval_max_gestures", 4096),
        sigma_eval_make_plots=training_section.get("sigma_eval_make_plots", True),
        absolute_coordinates=training_section.get("absolute_coordinates", False),
        generator_init_path=training_section.get("generator_init_path"),
        curvature_match_weight=training_section.get("curvature_match_weight", 0.0),
        lateral_match_weight=training_section.get("lateral_match_weight", 0.0),
        direction_match_weight=training_section.get("direction_match_weight", 0.0),
        warmup_encoder_hidden_dim=training_section.get("warmup_encoder_hidden_dim", 128),
        warmup_encoder_layers=training_section.get("warmup_encoder_layers", 1),
        warmup_encoder_dropout=training_section.get("warmup_encoder_dropout", 0.0),
        warmup_noise_std=training_section.get("warmup_noise_std", 0.0),
        warmup_latent_normalize=training_section.get("warmup_latent_normalize", True),
        warmup_kl_weight=training_section.get("warmup_kl_weight", 0.1),
        reset_generator_after_warmup=training_section.get("reset_generator_after_warmup", False),
        cold_start_epochs=training_section.get("cold_start_epochs", 0),
        sigma_freeze_upper=training_section.get("sigma_freeze_upper"),
        sigma_freeze_lower=training_section.get("sigma_freeze_lower"),
        label_smoothing_real=training_section.get("label_smoothing_real", 1.0),
        label_smoothing_fake=training_section.get("label_smoothing_fake", 0.0),
        r1_gamma=training_section.get("r1_gamma", 0.0),
        feature_workers=training_section.get("feature_workers"),
        sigma_feature_device=training_section.get("sigma_feature_device", "auto"),
        adaptive_freeze_enabled=training_section.get("adaptive_freeze_enabled", False),
        adaptive_freeze_target=training_section.get("adaptive_freeze_target", 0.6),
        adaptive_freeze_margin=training_section.get("adaptive_freeze_margin", 0.05),
        adaptive_freeze_warmup=training_section.get("adaptive_freeze_warmup", 50),
        adaptive_freeze_smoothing=training_section.get("adaptive_freeze_smoothing", 0.01),
        adaptive_freeze_cooldown=training_section.get("adaptive_freeze_cooldown", 0),
        adaptive_freeze_freeze_generator=training_section.get("adaptive_freeze_freeze_generator", True),
        fake_batch_ratio=training_section.get("fake_batch_ratio", 0.5),
        fake_batch_ratio_start=training_section.get("fake_batch_ratio_start", 0.2),
        fake_batch_ratio_warmup=training_section.get("fake_batch_ratio_warmup", 50),
        ada_enabled=training_section.get("ada_enabled", False),
        ada_target=training_section.get("ada_target", 0.6),
        ada_interval=training_section.get("ada_interval", 16),
        ada_rate=training_section.get("ada_rate", 0.05),
        ada_p_init=training_section.get("ada_p_init", 0.0),
        ada_p_max=training_section.get("ada_p_max", 0.9),
        adaptive_lr_enabled=training_section.get("adaptive_lr_enabled", False),
        adaptive_lr_target=training_section.get("adaptive_lr_target", 0.75),
        adaptive_lr_warmup=training_section.get("adaptive_lr_warmup", 0),
        adaptive_lr_error_gain=training_section.get("adaptive_lr_error_gain", 2.0),
        adaptive_lr_derivative_gain=training_section.get("adaptive_lr_derivative_gain", 1.0),
        adaptive_lr_min=training_section.get("adaptive_lr_min", 1e-6),
        adaptive_lr_max=training_section.get("adaptive_lr_max", 2e-4),
        diffusion_eval_enabled=training_section.get("diffusion_eval_enabled", False),
        diffusion_eval_checkpoint=training_section.get("diffusion_eval_checkpoint"),
        diffusion_eval_samples=training_section.get("diffusion_eval_samples", 512),
        diffusion_eval_steps=training_section.get("diffusion_eval_steps", 50),
        diffusion_eval_eta=training_section.get("diffusion_eval_eta", 0.0),
        diffusion_eval_seq_len=training_section.get("diffusion_eval_seq_len"),
        diffusion_eval_interval=training_section.get("diffusion_eval_interval", 1),
        diffusion_eval_log_dir=training_section.get("diffusion_eval_log_dir", "diffusion_eval"),
    )
    logging_cfg = LoggingConfig(**cfg_dict["logging"])

    experiment_cfg = GanExperimentConfig(
        experiment_name=cfg_dict["experiment_name"],
        seed=cfg_dict["seed"],
        data=data_cfg,
        model=model_cfg,
        training=training_cfg,
        logging=logging_cfg,
    )
    return experiment_cfg, cfg_dict


def _set_seed(seed: int, *, include_cuda: bool = False) -> None:
    torch.manual_seed(seed)
    if include_cuda and torch.cuda.is_available():  # pragma: no cover
        torch.cuda.manual_seed_all(seed)


def _prepare_dataset_and_dataloader(
    experiment_cfg: GanExperimentConfig,
) -> Tuple[GestureDataset, torch.utils.data.DataLoader]:
    dataset_cfg = GestureDatasetConfig(
        dataset_id=experiment_cfg.data.dataset_id,
        sequence_length=experiment_cfg.data.sequence_length,
        max_gestures=experiment_cfg.data.max_gestures,
        sampling_rate=experiment_cfg.data.sampling_rate,
        min_events=experiment_cfg.data.min_events,
        use_generated_negatives=False,
        cache_enabled=experiment_cfg.data.cache_enabled,
        cache_dir=experiment_cfg.data.cache_dir,
        split=experiment_cfg.data.split,
        user_filter=experiment_cfg.data.user_filter,
        normalize_sequences=experiment_cfg.data.normalize_sequences,
        normalize_features=experiment_cfg.data.normalize_features,
        feature_mode=experiment_cfg.data.feature_mode,
        canonicalize_path=experiment_cfg.data.canonicalize_path,
        canonicalize_duration=experiment_cfg.data.canonicalize_duration,
        include_goal_geometry=experiment_cfg.data.include_goal_geometry,
        use_click_boundaries=experiment_cfg.data.use_click_boundaries,
        click_button=experiment_cfg.data.click_button,
        direction_buckets=experiment_cfg.data.direction_buckets,
        rotate_to_buckets=experiment_cfg.data.rotate_to_buckets,
        min_path_length=experiment_cfg.data.min_path_length,
        feature_reservoir_size=experiment_cfg.data.feature_reservoir_size,
    )
    dataset = GestureDataset(dataset_cfg)
    if len(dataset) == 0:
        raise RuntimeError("Gesture dataset produced zero samples; check preprocessing pipeline.")

    feature_tensor = dataset.get_positive_features_tensor()
    if feature_tensor.numel() == 0:
        raise RuntimeError("No positive gesture features available; cannot determine condition dimensionality.")
    feature_dim = feature_tensor.shape[-1]
    if experiment_cfg.model.generator.condition_dim > 0 and experiment_cfg.model.generator.condition_dim != feature_dim:
        logger.warning(
            "Adjusting generator condition_dim from %d to feature_dim %d",
            experiment_cfg.model.generator.condition_dim,
            feature_dim,
        )
        experiment_cfg.model.generator.condition_dim = feature_dim
    if experiment_cfg.model.discriminator.condition_dim > 0 and experiment_cfg.model.discriminator.condition_dim != feature_dim:
        logger.warning(
            "Adjusting discriminator condition_dim from %d to feature_dim %d",
            experiment_cfg.model.discriminator.condition_dim,
            feature_dim,
        )
        experiment_cfg.model.discriminator.condition_dim = feature_dim

    loader_kwargs = {
        "batch_size": experiment_cfg.data.batch_size,
        "shuffle": True,
        "num_workers": experiment_cfg.data.num_workers,
        "drop_last": True,
    }
    if experiment_cfg.data.num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = max(1, experiment_cfg.data.prefetch_factor)
    dataloader = torch.utils.data.DataLoader(dataset, **loader_kwargs)
    return dataset, dataloader


def _compute_gradient_penalty(
    discriminator: GestureDiscriminator,
    real_samples: torch.Tensor,
    fake_samples: torch.Tensor,
    cond: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    alpha = torch.rand(real_samples.size(0), 1, 1, device=device)
    interpolates = alpha * real_samples + (1 - alpha) * fake_samples
    interpolates.requires_grad_(True)

    d_interpolates, _ = discriminator(interpolates, cond)
    gradients = torch.autograd.grad(
        outputs=d_interpolates.sum(),
        inputs=interpolates,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.reshape(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


def _to_device(batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], device: torch.device):
    sequences, features, labels = batch
    return sequences.to(device), features.to(device), labels.to(device)


def _get_condition(features: torch.Tensor, target_dim: int) -> torch.Tensor:
    if target_dim <= 0:
        return features.new_zeros(features.size(0), 0)
    if features.size(-1) == target_dim:
        return features
    if features.size(-1) > target_dim:
        return features[..., :target_dim]
    pad = torch.zeros(features.size(0), target_dim - features.size(-1), device=features.device, dtype=features.dtype)
    return torch.cat([features, pad], dim=-1)


def _feature_tensor_from_sequences(
    dataset: GestureDataset,
    sequences: torch.Tensor,
    *,
    normalize: bool = True,
    extra_features: Optional[torch.Tensor] = None,
    feature_workers: Optional[int] = None,
    feature_device: Optional[torch.device] = None,
    target_device: Optional[torch.device] = None,
) -> torch.Tensor:
    if sequences.dim() == 1:
        sequences = sequences.unsqueeze(0)
    batch_size = sequences.size(0)
    if batch_size == 0:
        mean, _ = dataset.get_feature_stats()
        feature_dim = mean.shape[0]
        device = target_device or sequences.device
        return mean.new_zeros((0, feature_dim), device=device)

    target_device = target_device or sequences.device
    feature_device = feature_device or target_device
    if feature_device is not None and feature_device.type == "cuda" and not torch.cuda.is_available():
        feature_device = torch.device("cpu")
    if feature_device is None:
        feature_device = torch.device("cpu")

    sequences_cpu = sequences.detach().to("cpu")
    if sequences.device.type == "cuda":
        sequences_cpu = sequences_cpu.pin_memory()

    cached_entries = dataset.get_cached_feature_vectors(sequences_cpu)

    features_list: list[Optional[torch.Tensor]] = [None] * batch_size
    missing_indices: list[int] = []
    cache_keys: list[Optional[str]] = [None] * batch_size

    for idx, cached_feat in enumerate(cached_entries):
        seq_cpu = sequences_cpu[idx]
        key = _sequence_cache_key(seq_cpu)
        cache_keys[idx] = key
        cached_tensor = _feature_cache_get(key)
        if cached_tensor is not None:
            features_list[idx] = cached_tensor
            continue
        if cached_feat is not None:
            tensor = cached_feat if isinstance(cached_feat, torch.Tensor) else torch.from_numpy(cached_feat)
            tensor_cpu = tensor.to(device="cpu", dtype=torch.float32)
            features_list[idx] = tensor_cpu
            _feature_cache_put(key, tensor_cpu)
        else:
            missing_indices.append(idx)

    if missing_indices:
        seq_missing_cpu = sequences_cpu[missing_indices]
        if feature_device.type == "cuda":
            seq_missing = seq_missing_cpu.to(feature_device, non_blocking=True)
        else:
            seq_missing = seq_missing_cpu
        mode = dataset.config.feature_mode
        if mode == "neuromotor":
            computed = [
                compute_features_from_sequence(seq_missing_cpu[i])
                for i in range(seq_missing_cpu.size(0))
            ]
            computed_tensors = torch.stack(computed, dim=0).to(device="cpu", dtype=torch.float32)
        elif mode == "sigma_lognormal":
            seq_proc = seq_missing.to(feature_device)
            computed_tensors = sigma_lognormal_features_torch(seq_proc)
            computed_tensors = computed_tensors.to(device="cpu", dtype=torch.float32)
        else:
            raise ValueError(f"Unsupported feature_mode: {mode}")

        for idx, feat_tensor in zip(missing_indices, computed_tensors):
            features_list[idx] = feat_tensor
            key = cache_keys[idx]
            if key is not None:
                _feature_cache_put(key, feat_tensor)

    tensors: list[torch.Tensor] = []
    for feat in features_list:
        if feat is None:
            raise RuntimeError("Feature extraction cache returned an empty entry; this should not happen.")
        tensors.append(feat)
    feats_cpu = torch.stack(tensors, dim=0)
    if sequences.device.type == "cuda":
        feats_cpu = feats_cpu.pin_memory()
    feats = feats_cpu.to(target_device, dtype=torch.float32, non_blocking=(target_device.type == "cuda"))

    mean, _ = dataset.get_feature_stats()
    total_dim = mean.shape[0]
    if feats.dim() == 1:
        feats = feats.unsqueeze(0)

    current_dim = feats.size(-1)
    if current_dim < total_dim:
        extra_dim = total_dim - current_dim
        if extra_features is not None:
            extra = extra_features
            if extra.dim() == 1:
                extra = extra.unsqueeze(0)
            if extra.size(0) != feats.size(0):
                if extra.size(0) == 1:
                    extra = extra.expand(feats.size(0), -1)
                else:
                    extra = extra[: feats.size(0)]
            extra = extra.to(feats.device, feats.dtype)
            if extra.size(-1) >= extra_dim:
                extra = extra[..., -extra_dim:]
            else:
                pad = torch.zeros(feats.size(0), extra_dim - extra.size(-1), device=feats.device, dtype=feats.dtype)
                extra = torch.cat([extra, pad], dim=-1)
        else:
            extra = torch.zeros(feats.size(0), extra_dim, device=feats.device, dtype=feats.dtype)
        feats = torch.cat([feats, extra], dim=-1)
    elif current_dim > total_dim:
        feats = feats[..., :total_dim]

    if normalize and dataset.config.normalize_features:
        feats = dataset.normalize_features(feats)
    return feats


def _save_generated_samples(samples: torch.Tensor, out_dir: Path, epoch: int, step: int) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"samples_epoch{epoch:03d}_step{step:06d}.npz"
    np.savez_compressed(path, sequences=samples.detach().cpu().numpy())
    return path


def _sampling_rate_to_float(value) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _run_sigma_evaluation(
    experiment_cfg: GanExperimentConfig,
    generator: ConditionalGenerator,
    dataset: GestureDataset,
    feature_pool: torch.Tensor,
    device: torch.device,
    epoch: int,
    run,
    metrics_logger: Optional[CSVMetricLogger],
    *,
    position_stats: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
) -> Optional[dict]:
    training_cfg = experiment_cfg.training
    if not training_cfg.sigma_eval_enabled:
        return None

    if epoch % max(1, training_cfg.sigma_eval_interval) != 0:
        return None

    total_samples = max(1, training_cfg.sigma_eval_samples)
    batch_size = experiment_cfg.data.batch_size
    latent_dim = experiment_cfg.model.generator.latent_dim
    condition_dim = experiment_cfg.model.generator.condition_dim
    use_absolute = experiment_cfg.training.absolute_coordinates
    is_seq2seq = experiment_cfg.model.architecture == "seq2seq"
    sampling_rate = _sampling_rate_to_float(experiment_cfg.data.sampling_rate)
    dt_value = (1.0 / sampling_rate) if sampling_rate and sampling_rate > 0 else None

    generator_was_training = generator.training
    generator.eval()

    if position_stats is not None:
        pos_mean_eval, pos_std_eval = position_stats
        pos_mean_eval = pos_mean_eval.to(device)
        pos_std_eval = pos_std_eval.to(device)
    else:
        pos_mean_eval = pos_std_eval = None

    collected: list[torch.Tensor] = []
    samples_remaining = total_samples
    with torch.no_grad():
        while samples_remaining > 0:
            current_batch = min(batch_size, samples_remaining)
            idx = torch.randint(0, feature_pool.size(0), (current_batch,), device=device)
            cond = _get_condition(feature_pool[idx], condition_dim)
            z = torch.randn(current_batch, latent_dim, device=device)
            batch_output = generator(z, cond)
            if use_absolute:
                if is_seq2seq:
                    fake_positions = batch_output
                    if pos_mean_eval is not None and pos_std_eval is not None:
                        fake_positions = fake_positions * (pos_std_eval + 1e-6) + pos_mean_eval
                else:
                    fake_positions = _deltas_to_positions_xy(batch_output[..., :2])
                deltas_xy = _positions_to_deltas_xy(fake_positions)
            else:
                fake_positions = None
                if is_seq2seq:
                    if pos_mean_eval is not None and pos_std_eval is not None:
                        fake_positions = batch_output * (pos_std_eval + 1e-6) + pos_mean_eval
                        deltas_xy = _positions_to_deltas_xy(fake_positions)
                    else:
                        deltas_xy = batch_output
                else:
                    deltas_xy = batch_output[..., :2]

            seq = torch.zeros(current_batch, batch_output.size(1), 3, device=device, dtype=batch_output.dtype)
            seq[..., :2] = deltas_xy
            if dt_value is not None:
                seq[..., 2] = dt_value
            else:
                seq[..., 2] = 1.0

            batch_sequences = _canonicalize_tensor_sequences(seq, experiment_cfg.data)
            denorm_batch = dataset.denormalize_sequences(batch_sequences.detach().cpu())
            collected.append(denorm_batch)
            samples_remaining -= current_batch

    if generator_was_training:
        generator.train()

    generated_sequences = torch.cat(collected, dim=0)
    eval_dir = Path(experiment_cfg.logging.checkpoint_dir) / "sigma_eval" / f"epoch_{epoch:03d}"
    eval_dir.mkdir(parents=True, exist_ok=True)
    eval_np_path = eval_dir / "gan_samples.npz"
    np.savez_compressed(eval_np_path, sequences=generated_sequences.numpy())

    dataset_id = training_cfg.sigma_eval_dataset_id or experiment_cfg.data.dataset_id
    summary = run_baseline(
        dataset_id=dataset_id,
        sequence_length=experiment_cfg.data.sequence_length,
        max_gestures=training_cfg.sigma_eval_max_gestures,
        seed=experiment_cfg.seed + epoch,
        samples_per_case=min(total_samples, generated_sequences.shape[0]),
        generator_type="gan",
        gan_dir=eval_dir,
        output=eval_dir / "sigma_eval_summary.json",
        canonicalize_path=experiment_cfg.data.canonicalize_path,
        canonicalize_duration=experiment_cfg.data.canonicalize_duration,
        make_plots=training_cfg.sigma_eval_make_plots,
    )

    avg_accuracy = float(summary.get("average_accuracy", 0.0))
    avg_error = 1.0 - avg_accuracy
    result_payload = {
        "epoch": epoch,
        "avg_accuracy": avg_accuracy,
        "avg_error_rate": avg_error,
        "num_files": len(summary.get("results", [])),
        "samples": int(generated_sequences.shape[0]),
        "summary_path": str(eval_dir / "sigma_eval_summary.json"),
        "replay_dir": str(eval_dir),
        "samples_path": str(eval_np_path),
    }

    if run is not None:
        run.log(
            {
                "sigma_eval/epoch": epoch,
                "sigma_eval/accuracy": avg_accuracy,
                "sigma_eval/error_rate": avg_error,
            }
        )

    if metrics_logger is not None:
        metrics_logger.log(result_payload)

    logger.info(
        "Epoch %d | sigma_eval accuracy=%.4f error_rate=%.4f samples=%d",
        epoch,
        avg_accuracy,
        avg_error,
        generated_sequences.shape[0],
    )

    return result_payload


def _run_diffusion_evaluation(
    experiment_cfg: GanExperimentConfig,
    dataset: GestureDataset,
    device: torch.device,
    epoch: int,
    run,
    metrics_logger: Optional[CSVMetricLogger],
    *,
    real_features: Optional[torch.Tensor] = None,
) -> Optional[dict]:
    training_cfg = experiment_cfg.training
    if not training_cfg.diffusion_eval_enabled:
        return None

    if epoch % max(1, training_cfg.diffusion_eval_interval) != 0:
        return None

    checkpoint_path = training_cfg.diffusion_eval_checkpoint
    if not checkpoint_path:
        logger.warning("Diffusion evaluation enabled but no checkpoint path provided.")
        return None

    checkpoint_path = Path(to_absolute_path(checkpoint_path))
    if not checkpoint_path.exists():
        logger.warning("Diffusion checkpoint not found at %s", checkpoint_path)
        return None

    total_samples = max(1, training_cfg.diffusion_eval_samples)
    seq_len = training_cfg.diffusion_eval_seq_len or experiment_cfg.data.sequence_length

    try:
        generated = generate_diffusion_samples(
            total_samples,
            seq_len,
            checkpoint_path=checkpoint_path,
            steps=max(1, int(training_cfg.diffusion_eval_steps)),
            eta=float(training_cfg.diffusion_eval_eta),
            device=device,
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("Diffusion sampling failed for checkpoint %s: %s", checkpoint_path, exc)
        return None

    if generated.dim() != 3:
        logger.error("Diffusion sampler returned tensor with unexpected rank %d", generated.dim())
        return None

    target_dim = dataset.samples[0][0].size(-1) if dataset.samples else generated.size(-1)
    if generated.size(-1) < target_dim:
        pad = target_dim - generated.size(-1)
        generated = F.pad(generated, (0, pad))
        if pad > 0:
            generated[..., -pad:] = 0.0
    elif generated.size(-1) > target_dim:
        generated = generated[..., :target_dim]

    real_time_ref = _sample_real_time_sequences(
        dataset,
        generated.size(0),
        generated.size(1),
        seed=experiment_cfg.seed + epoch,
    )
    if real_time_ref is not None:
        generated = match_time_channel(
            generated,
            real_time_ref.to(device=generated.device, dtype=generated.dtype),
        )

    if experiment_cfg.data.canonicalize_path or experiment_cfg.data.canonicalize_duration:
        generated = _canonicalize_tensor_sequences(generated, experiment_cfg.data)

    normalized_sequences = generated.detach().cpu()
    denorm_sequences = dataset.denormalize_sequences(normalized_sequences)
    denorm_np = denorm_sequences.numpy()

    fake_feature_tensors: list[torch.Tensor] = []
    fake_feature_raw: list[torch.Tensor] = []
    for seq in denorm_sequences:
        feat_raw = compute_features_from_sequence(seq)
        feat_norm = dataset.normalize_features(feat_raw)
        fake_feature_raw.append(feat_raw)
        fake_feature_tensors.append(feat_norm)

    if not fake_feature_tensors:
        logger.warning("Diffusion evaluation could not compute any features from generated samples.")
        return None

    fake_features = torch.stack(fake_feature_tensors).cpu()
    if real_features is None or real_features.numel() == 0:
        real_features = dataset.get_positive_features_tensor(use_full=True)

    real_np = real_features.cpu().numpy() if real_features is not None else np.empty((0, fake_features.size(1)))
    fake_np = fake_features.numpy()

    if real_np.size == 0:
        logger.warning("Diffusion evaluation skipped because no real features were available.")
        return None

    feature_dim = min(real_np.shape[1], fake_np.shape[1])
    real_np = real_np[:, :feature_dim]
    fake_np = fake_np[:, :feature_dim]

    if real_np.shape[0] > fake_np.shape[0] and fake_np.shape[0] > 0:
        rng = np.random.default_rng(experiment_cfg.seed + epoch)
        idx = rng.choice(real_np.shape[0], size=fake_np.shape[0], replace=False)
        real_np = real_np[idx]

    metrics = feature_distribution_metrics(real_np, fake_np)
    metrics["diversity_xy"] = sequence_diversity_metric(denorm_np[..., :2])

    classifier_metrics = _diffusion_classifier_metrics(
        real_np,
        fake_np,
        seed=experiment_cfg.seed + epoch,
    )
    if classifier_metrics:
        metrics.update(classifier_metrics)

    eval_dir = (
        Path(experiment_cfg.logging.checkpoint_dir)
        / training_cfg.diffusion_eval_log_dir
        / f"epoch_{epoch:03d}"
    )
    eval_dir.mkdir(parents=True, exist_ok=True)

    sequences_path = eval_dir / "diffusion_samples.npz"
    seq_mean, seq_std = dataset.get_sequence_stats()
    np.savez_compressed(
        sequences_path,
        sequences=denorm_np,
        sequences_denorm=denorm_np,
        sequences_norm=normalized_sequences.numpy(),
        sequence_mean=seq_mean.cpu().numpy(),
        sequence_std=seq_std.cpu().numpy(),
        normalize_sequences=dataset.config.normalize_sequences,
    )

    features_path = eval_dir / "diffusion_features.npz"
    feat_mean, feat_std = dataset.get_feature_stats()
    np.savez_compressed(
        features_path,
        features=fake_np,
        features_norm=fake_np,
        features_denorm=torch.stack(fake_feature_raw).numpy(),
        feature_mean=feat_mean.cpu().numpy(),
        feature_std=feat_std.cpu().numpy(),
        normalize_features=dataset.config.normalize_features,
    )

    summary = {
        "epoch": epoch,
        "samples": int(fake_np.shape[0]),
        "samples_path": str(sequences_path),
        "features_path": str(features_path),
    }
    summary.update({k: float(v) for k, v in metrics.items()})

    summary_path = eval_dir / "metrics.json"
    with summary_path.open("w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2)
    summary["summary_path"] = str(summary_path)

    if run is not None:
        log_dict = {f"diffusion_eval/{k}": v for k, v in metrics.items()}
        log_dict["diffusion_eval/epoch"] = epoch
        log_dict["diffusion_eval/samples"] = summary["samples"]
        run.log(log_dict)

    if metrics_logger is not None:
        metrics_logger.log(summary)

    logger.info(
        "Epoch %d | diffusion_eval mean_l1=%.6f mean_l2=%.6f cov_diff=%.6f diversity=%.6f",
        epoch,
        metrics.get("mean_l1", 0.0),
        metrics.get("mean_l2", 0.0),
        metrics.get("cov_trace_diff", 0.0),
        metrics.get("diversity_xy", 0.0),
    )

    return summary


def _sample_real_time_sequences(
    dataset: GestureDataset,
    count: int,
    seq_len: int,
    *,
    seed: int,
) -> Optional[torch.Tensor]:
    """Draw positive real gestures and return tensor for borrowing Δt channels."""
    positives: list[torch.Tensor] = []
    for seq, _, label in dataset.samples:
        if label.item() >= 0.5:
            positives.append(seq)
    if not positives:
        return None
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(positives), size=count, replace=True)
    channels = positives[0].size(-1)
    stacked = torch.zeros(count, seq_len, channels, dtype=positives[0].dtype)
    for out_idx, src_idx in enumerate(idx):
        seq = positives[src_idx].detach().cpu()
        length = min(seq.size(0), seq_len)
        stacked[out_idx, :length, : seq.size(-1)] = seq[:length]
    return stacked


def _diffusion_classifier_metrics(real_np: np.ndarray, fake_np: np.ndarray, *, seed: int) -> dict[str, float]:
    min_samples = min(real_np.shape[0], fake_np.shape[0])
    if min_samples < 20:
        return {}
    feature_dim = min(real_np.shape[1], fake_np.shape[1])
    X_real = real_np[:, :feature_dim]
    X_fake = fake_np[:, :feature_dim]
    labels_real = np.zeros(X_real.shape[0])
    labels_fake = np.ones(X_fake.shape[0])
    X = np.concatenate([X_real, X_fake], axis=0)
    y = np.concatenate([labels_real, labels_fake], axis=0)
    try:
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
        return {"c2st_accuracy": accuracy, "c2st_auc": auc}
    except Exception as exc:  # pragma: no cover
        logger.warning("Diffusion classifier evaluation failed: %s", exc)
        return {}


def _seq2seq_warmup(
    experiment_cfg: GanExperimentConfig,
    generator: Seq2SeqGenerator,
    dataloader: torch.utils.data.DataLoader,
    dataset: GestureDataset,
    device: torch.device,
    run,
    *,
    conditioning_callback=None,
    fixed_dt: Optional[float] = None,
) -> None:
    epochs = experiment_cfg.training.reconstruction_epochs
    if epochs <= 0:
        return

    logger.info("Starting seq2seq warm-up for %d epoch(s)", epochs)
    output_dim = generator.output_dim
    pos_mean_cpu, pos_std_cpu = _position_stats_from_dataset(dataset, output_dim)
    encoder = Seq2SeqEncoder(
        input_dim=output_dim,
        hidden_dim=experiment_cfg.training.warmup_encoder_hidden_dim,
        latent_dim=experiment_cfg.model.generator.latent_dim,
        num_layers=experiment_cfg.training.warmup_encoder_layers,
        dropout=experiment_cfg.training.warmup_encoder_dropout,
    ).to(device)

    recon_criterion = nn.L1Loss()
    optimizer = optim.Adam(
        list(generator.parameters()) + list(encoder.parameters()),
        lr=experiment_cfg.training.lr_generator,
        betas=(experiment_cfg.training.beta1, experiment_cfg.training.beta2),
    )
    kl_weight = float(experiment_cfg.training.warmup_kl_weight)
    pos_mean = pos_mean_cpu.to(device)
    pos_std = pos_std_cpu.to(device)

    for epoch in range(1, epochs + 1):
        generator.train()
        encoder.train()
        total_recon = 0.0
        total_kl = 0.0
        total_samples = 0

        for batch in dataloader:
            real_sequences, features, _ = _to_device(batch, device)
            if fixed_dt is not None:
                real_sequences = _apply_fixed_dt(real_sequences, dataset, fixed_dt)
            condition = _get_condition(features, experiment_cfg.model.generator.condition_dim)
            if conditioning_callback is not None:
                condition = conditioning_callback(real_sequences, condition)

            targets = real_sequences[..., :output_dim]
            positions = torch.cumsum(targets, dim=1)
            targets_norm = (positions - pos_mean) / (pos_std + 1e-6)
            mu, logvar = encoder(targets_norm)
            eps = torch.randn_like(mu)
            z = mu + eps * torch.exp(0.5 * logvar)

            recon = generator(z, condition, teacher_forcing=targets_norm)
            recon_loss = recon_criterion(recon, targets_norm)
            kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
            loss = recon_loss + kl_weight * kl

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_recon += recon_loss.item() * real_sequences.size(0)
            total_kl += kl.item() * real_sequences.size(0)
            total_samples += real_sequences.size(0)

        avg_recon = total_recon / max(1, total_samples)
        avg_kl = total_kl / max(1, total_samples)
        logger.info(
            "Warm-up epoch %d | recon_loss=%.6f kl=%.6f",
            epoch,
            avg_recon,
            avg_kl,
        )
        if run is not None:
            run.log(
                {
                    "warmup/epoch": epoch,
                    "warmup/reconstruction_loss": avg_recon,
                    "warmup/kl": avg_kl,
                }
            )

    generator.eval()
    num_samples = min(1024, len(dataset))
    latent = torch.randn(num_samples, experiment_cfg.model.generator.latent_dim, device=device)
    cond_dim = experiment_cfg.model.generator.condition_dim
    if cond_dim > 0:
        feature_tensor = dataset.get_positive_features_tensor()
        if feature_tensor.numel() == 0:
            cond = torch.zeros(num_samples, cond_dim, device=device)
        else:
            cond = feature_tensor[:num_samples].to(device)
            if cond.size(1) > cond_dim:
                cond = cond[:, :cond_dim]
            elif cond.size(1) < cond_dim:
                pad = torch.zeros(cond.size(0), cond_dim - cond.size(1), device=device, dtype=cond.dtype)
                cond = torch.cat([cond, pad], dim=-1)
    else:
        cond = torch.zeros(num_samples, 0, device=device)

    with torch.no_grad():
        generated_norm = generator.sample(latent, cond, steps=experiment_cfg.data.sequence_length)
    generated_norm_cpu = generated_norm.cpu()
    positions_cpu = generated_norm_cpu * (pos_std_cpu + 1e-6) + pos_mean_cpu
    deltas_cpu = torch.zeros_like(positions_cpu)
    deltas_cpu[:, 0, :] = positions_cpu[:, 0, :]
    deltas_cpu[:, 1:, :] = positions_cpu[:, 1:, :] - positions_cpu[:, :-1, :]

    warmup_dir = Path(experiment_cfg.logging.checkpoint_dir) / "warmup_samples"
    warmup_dir.mkdir(parents=True, exist_ok=True)

    sample_dim = dataset.samples[0][0].size(-1) if dataset.samples else output_dim
    if sample_dim > output_dim:
        if fixed_dt is not None:
            dt_val = float(fixed_dt)
        else:
            sr_value = _sampling_rate_to_float(experiment_cfg.data.sampling_rate)
            if sr_value and sr_value > 0:
                dt_val = 1.0 / sr_value
            else:
                dt_val = 1.0
        dt_col = torch.full(
            (deltas_cpu.size(0), deltas_cpu.size(1), sample_dim - output_dim),
            dt_val,
        )
        generated_full = torch.cat([deltas_cpu, dt_col], dim=-1)
    else:
        generated_full = deltas_cpu

    np.savez_compressed(
        warmup_dir / "samples_epoch_warmup.npz",
        sequences=generated_full.numpy(),
    )

    denorm_generated = dataset.denormalize_sequences(generated_full)
    deltas = denorm_generated[..., :2]
    stats = {
        "warmup/recon_delta_std_x": float(deltas[..., 0].std().item()),
        "warmup/recon_delta_std_y": float(deltas[..., 1].std().item()),
        "warmup/recon_delta_mean_x": float(deltas[..., 0].mean().item()),
        "warmup/recon_delta_mean_y": float(deltas[..., 1].mean().item()),
        "warmup/recon_theta_std": float(torch.atan2(deltas[:, 0, 1], deltas[:, 0, 0]).std().item()),
    }
    logger.info(
        "Warm-up diagnostics | recon_delta_std=(%.4f, %.4f) theta_std=%.4f",
        stats["warmup/recon_delta_std_x"],
        stats["warmup/recon_delta_std_y"],
        stats["warmup/recon_theta_std"],
    )
    if run is not None:
        run.log(stats)

    del encoder


def _reconstruction_warmup(
    experiment_cfg: GanExperimentConfig,
    generator: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    dataset: GestureDataset,
    device: torch.device,
    run,
    *,
    conditioning_callback=None,
    fixed_dt: Optional[float] = None,
) -> None:
    if experiment_cfg.model.architecture == "seq2seq":
        _seq2seq_warmup(
            experiment_cfg,
            generator,
            dataloader,
            dataset,
            device,
            run,
            conditioning_callback=conditioning_callback,
            fixed_dt=fixed_dt,
        )
        return

    epochs = experiment_cfg.training.reconstruction_epochs
    if epochs <= 0:
        return

    logger.info("Starting reconstruction warm-up for %d epoch(s)", epochs)
    generator_cfg = experiment_cfg.model.generator
    generator_output_dim = getattr(getattr(generator, "config", generator_cfg), "output_dim", None)
    if generator_output_dim is None:
        sample_seq, _, _ = dataset.samples[0]
        generator_output_dim = sample_seq.shape[-1]
    use_absolute = experiment_cfg.training.absolute_coordinates
    input_dim = int(generator_output_dim)
    latent_dim = getattr(generator_cfg, "latent_dim", generator_output_dim)
    hidden_dim = experiment_cfg.training.warmup_encoder_hidden_dim or getattr(generator_cfg, "hidden_dim", latent_dim)
    hidden_dim = int(hidden_dim)
    encoder_layers = max(1, int(experiment_cfg.training.warmup_encoder_layers))
    encoder_dropout = max(0.0, float(experiment_cfg.training.warmup_encoder_dropout))
    warmup_noise_std = float(max(0.0, experiment_cfg.training.warmup_noise_std))
    normalize_latent = bool(experiment_cfg.training.warmup_latent_normalize)

    encoder = WarmupSequenceEncoder(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        num_layers=encoder_layers,
        dropout=encoder_dropout,
    ).to(device)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(
        list(generator.parameters()) + list(encoder.parameters()),
        lr=experiment_cfg.training.lr_generator,
        betas=(experiment_cfg.training.beta1, experiment_cfg.training.beta2),
    )

    loss_weight = float(experiment_cfg.training.reconstruction_loss_weight)
    if loss_weight <= 0.0:
        loss_weight = 1.0
        logger.warning(
            "Reconstruction warm-up requested but reconstruction_loss_weight <= 0; defaulting to %.2f",
            loss_weight,
        )

    def _prepare_targets(real_seq: torch.Tensor) -> torch.Tensor:
        target = real_seq[..., :input_dim]
        if use_absolute:
            target = _deltas_to_positions_xy(target)
        return target

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        total_samples = 0
        generator.train()
        encoder.train()
        for batch in dataloader:
            real_sequences, features, _ = _to_device(batch, device)
            if fixed_dt is not None:
                real_sequences = _apply_fixed_dt(real_sequences, dataset, fixed_dt)
            condition = _get_condition(features, experiment_cfg.model.generator.condition_dim)
            if conditioning_callback is not None:
                condition = conditioning_callback(real_sequences, condition)
            target_sequences = _prepare_targets(real_sequences)
            latent = encoder(target_sequences)
            if normalize_latent and latent.size(0) > 1:
                mean = latent.mean(dim=0, keepdim=True)
                std = latent.std(dim=0, keepdim=True).clamp_min(1e-6)
                latent = (latent - mean) / std
            if warmup_noise_std > 0.0:
                latent = latent + torch.randn_like(latent) * warmup_noise_std
            reconstructed = generator(latent, condition)
            loss = criterion(reconstructed, target_sequences) * loss_weight

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * real_sequences.size(0)
            total_samples += real_sequences.size(0)

        avg_loss = total_loss / max(1, total_samples)
        logger.info("Warm-up epoch %d | recon_loss=%.6f", epoch, avg_loss)
        if run is not None:
            run.log({"warmup/epoch": epoch, "warmup/reconstruction_loss": avg_loss})

    # Post-warm-up diagnostics
    generator.eval()
    encoder.eval()
    diag_sequences: Optional[torch.Tensor] = None
    try:
        sample_batch = next(iter(dataloader))
    except StopIteration:
        sample_batch = None
    if sample_batch is not None:
        real_seq_batch, features_batch, _ = _to_device(sample_batch, device)
        if fixed_dt is not None:
            real_seq_batch = _apply_fixed_dt(real_seq_batch, dataset, fixed_dt)
        condition_batch = _get_condition(features_batch, experiment_cfg.model.generator.condition_dim)
        if conditioning_callback is not None:
            condition_batch = conditioning_callback(real_seq_batch, condition_batch)
        target_batch = _prepare_targets(real_seq_batch)
        with torch.no_grad():
            latent = encoder(target_batch)
            if normalize_latent and latent.size(0) > 1:
                mean = latent.mean(dim=0, keepdim=True)
                std = latent.std(dim=0, keepdim=True).clamp_min(1e-6)
                latent = (latent - mean) / std
            recon = generator(latent, condition_batch)
        if use_absolute:
            recon_seq = _positions_to_deltas_xy(recon)
        else:
            recon_seq = recon
        if recon_seq.size(-1) < real_seq_batch.size(-1):
            pad = real_seq_batch[..., recon_seq.size(-1) : real_seq_batch.size(-1)]
            if pad.numel() > 0:
                recon_seq = torch.cat([recon_seq, pad.detach()], dim=-1)
        if recon_seq.size(-1) > real_seq_batch.size(-1):
            recon_seq = recon_seq[..., : real_seq_batch.size(-1)]
        diag_sequences = recon_seq.detach().cpu()

        real_denorm = dataset.denormalize_sequences(real_seq_batch.detach().cpu())
        recon_denorm = dataset.denormalize_sequences(recon_seq.detach().cpu())

        real_deltas = real_denorm[..., :2]
        recon_deltas = recon_denorm[..., :2]

        stats = {
            "warmup/real_delta_std_x": float(real_deltas[..., 0].std().item()),
            "warmup/real_delta_std_y": float(real_deltas[..., 1].std().item()),
            "warmup/recon_delta_std_x": float(recon_deltas[..., 0].std().item()),
            "warmup/recon_delta_std_y": float(recon_deltas[..., 1].std().item()),
            "warmup/real_theta_std": float(torch.atan2(real_deltas[:, 0, 1], real_deltas[:, 0, 0]).std().item()),
            "warmup/recon_theta_std": float(torch.atan2(recon_deltas[:, 0, 1], recon_deltas[:, 0, 0]).std().item()),
            "warmup/recon_displacement_mean_x": float(recon_deltas.sum(dim=1)[..., 0].mean().item()),
            "warmup/recon_displacement_mean_y": float(recon_deltas.sum(dim=1)[..., 1].mean().item()),
        }
        logger.info(
            "Warm-up diagnostics | recon_delta_std=(%.4f, %.4f) theta_std=%.4f disp_mean=(%.3f, %.3f)",
            stats["warmup/recon_delta_std_x"],
            stats["warmup/recon_delta_std_y"],
            stats["warmup/recon_theta_std"],
            stats["warmup/recon_displacement_mean_x"],
            stats["warmup/recon_displacement_mean_y"],
        )
        if run is not None:
            run.log(stats)

    if diag_sequences is not None:
        warmup_dir = Path(experiment_cfg.logging.checkpoint_dir) / "warmup_samples"
        warmup_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            warmup_dir / "samples_epoch_warmup.npz",
            sequences=diag_sequences.numpy(),
        )

    del encoder


def _training_loop(
    experiment_cfg: GanExperimentConfig,
    generator: torch.nn.Module,
    discriminator: torch.nn.Module,
    dataset: GestureDataset,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    feature_device: torch.device,
    run,
    metrics_logger: Optional[CSVMetricLogger],
    position_stats: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    profiler: Optional[Any] = None,
) -> Dict[str, Any]:
    training_cfg = experiment_cfg.training
    fixed_dt = _estimate_fixed_dt(dataset, experiment_cfg)
    if fixed_dt is not None:
        logger.info("Using fixed Δt=%.6f for warm-start and training", fixed_dt)

    _reconstruction_warmup(
        experiment_cfg,
        generator,
        dataloader,
        dataset,
        device,
        run,
        fixed_dt=fixed_dt,
    )

    if experiment_cfg.training.reset_generator_after_warmup:
        logger.info("Resetting generator weights after warm-up as configured")
        _reset_generator_weights(generator, experiment_cfg)

    g_optimizer = optim.Adam(
        generator.parameters(),
        lr=experiment_cfg.training.lr_generator,
        betas=(experiment_cfg.training.beta1, experiment_cfg.training.beta2),
    )
    d_optimizer = optim.Adam(
        discriminator.parameters(),
        lr=experiment_cfg.training.lr_discriminator,
        betas=(experiment_cfg.training.beta1, experiment_cfg.training.beta2),
    )

    buffer = ReplayBuffer(
        path=Path(experiment_cfg.training.replay_buffer_path),
        max_size=experiment_cfg.training.replay_buffer_max_size,
    )
    conditioning_features = dataset.get_positive_features_tensor()
    conditioning_features_denorm = dataset.denormalize_features(conditioning_features.clone())
    feature_pool = conditioning_features.to(device)
    geometry_feature_pool: Optional[torch.Tensor] = None
    if getattr(dataset, "_geometry_dim", 0) > 0:
        geometry_feature_pool = conditioning_features_denorm[:, -dataset._geometry_dim :].contiguous()
    real_feature_cpu = dataset.get_positive_features_tensor(use_full=True)
    real_feature_np = real_feature_cpu.numpy()
    real_features_plot = dataset.denormalize_features(real_feature_cpu).cpu().numpy() if real_feature_cpu.numel() > 0 else np.empty((0, 0))
    positive_sequences = [
        seq.detach().cpu()
        for seq, _, label in dataset.samples
        if label.item() > 0.5
    ]
    real_sequences_plot: Optional[np.ndarray] = None
    if positive_sequences:
        stacked_sequences = torch.stack(positive_sequences)
        real_sequences_plot = dataset.denormalize_sequences(stacked_sequences).cpu().numpy()
    sample_dir = Path(experiment_cfg.logging.checkpoint_dir) / "samples"
    step = 0
    detector_metrics_logger = CSVMetricLogger(
        Path(experiment_cfg.logging.checkpoint_dir) / "detector_metrics.csv",
        fieldnames=[
            "epoch",
            "train_loss",
            "train_acc",
            "val_loss",
            "val_acc",
            "roc_auc",
            "pr_auc",
            "fpr_at_95_tpr",
        ],
    )
    sigma_eval_logger = None
    sigma_eval_records: list[dict] = []
    if experiment_cfg.training.sigma_eval_enabled:
        sigma_eval_logger = CSVMetricLogger(
            Path(experiment_cfg.logging.checkpoint_dir) / "sigma_eval_metrics.csv",
            fieldnames=[
                "epoch",
                "avg_accuracy",
                "avg_error_rate",
                "num_files",
                "samples",
                "summary_path",
                "replay_dir",
                "samples_path",
            ],
        )

    diffusion_eval_logger = None
    diffusion_eval_records: list[dict] = []
    if experiment_cfg.training.diffusion_eval_enabled:
        diffusion_eval_logger = CSVMetricLogger(
            Path(experiment_cfg.logging.checkpoint_dir) / "diffusion_eval_metrics.csv",
            fieldnames=[
                "epoch",
                "samples",
                "mean_l1",
                "mean_l2",
                "cov_trace_diff",
                "diversity_xy",
                "c2st_accuracy",
                "c2st_auc",
                "samples_path",
                "features_path",
                "summary_path",
            ],
        )

    cold_start_epochs = max(0, experiment_cfg.training.cold_start_epochs)
    upper_sigma_thresh = experiment_cfg.training.sigma_freeze_upper
    lower_sigma_thresh = experiment_cfg.training.sigma_freeze_lower
    freeze_discriminator_next = False
    freeze_generator_next = False
    last_sigma_accuracy: Optional[float] = None
    min_freeze_epoch = max(cold_start_epochs + 1, 1)

    if position_stats is not None:
        pos_mean, pos_std = position_stats
        pos_mean = pos_mean.to(device)
        pos_std = pos_std.to(device)
    else:
        pos_mean = pos_std = None

    def _standardize_positions(pos: torch.Tensor) -> torch.Tensor:
        if pos_mean is None or pos_std is None:
            return pos
        return (pos - pos_mean) / (pos_std + 1e-6)

    last_metrics: Dict[str, Any] = {}
    adversarial_type = experiment_cfg.training.adversarial_type.lower()
    use_wgan = adversarial_type == "wgan"
    bce_loss = nn.BCEWithLogitsLoss() if not use_wgan else None
    best_sigma_error = float("inf")
    best_sigma_epoch: Optional[int] = None
    adaptive_enabled = experiment_cfg.training.adaptive_freeze_enabled
    adaptive_target = float(experiment_cfg.training.adaptive_freeze_target)
    adaptive_margin = float(experiment_cfg.training.adaptive_freeze_margin)
    adaptive_warmup = max(0, int(experiment_cfg.training.adaptive_freeze_warmup))
    adaptive_smoothing = float(max(1e-4, experiment_cfg.training.adaptive_freeze_smoothing))
    adaptive_cooldown_cfg = max(0, int(experiment_cfg.training.adaptive_freeze_cooldown))
    adaptive_freeze_generator = bool(experiment_cfg.training.adaptive_freeze_freeze_generator)
    disc_accuracy_ema: Optional[float] = None
    disc_accuracy_epoch: Optional[float] = None
    discriminator_cooldown = 0
    generator_cooldown = 0
    fake_ratio_target = float(max(1e-3, min(0.95, experiment_cfg.training.fake_batch_ratio)))
    fake_ratio_start = float(max(1e-3, min(0.95, experiment_cfg.training.fake_batch_ratio_start)))
    fake_ratio_warmup = max(0, int(experiment_cfg.training.fake_batch_ratio_warmup))
    ada_enabled = experiment_cfg.training.ada_enabled
    ada_target = float(experiment_cfg.training.ada_target)
    ada_interval = max(1, int(experiment_cfg.training.ada_interval))
    ada_rate = float(experiment_cfg.training.ada_rate)
    ada_p = float(experiment_cfg.training.ada_p_init)
    ada_p_max = float(max(0.0, min(0.99, experiment_cfg.training.ada_p_max)))
    ada_accumulator = 0.0
    ada_counter = 0
    adaptive_lr_enabled = experiment_cfg.training.adaptive_lr_enabled
    adaptive_lr_target = float(experiment_cfg.training.adaptive_lr_target)
    adaptive_lr_warmup = max(0, int(experiment_cfg.training.adaptive_lr_warmup))
    adaptive_lr_error_gain = float(experiment_cfg.training.adaptive_lr_error_gain)
    adaptive_lr_derivative_gain = float(experiment_cfg.training.adaptive_lr_derivative_gain)
    adaptive_lr_min = float(experiment_cfg.training.adaptive_lr_min)
    adaptive_lr_max = float(experiment_cfg.training.adaptive_lr_max)
    prev_disc_accuracy_epoch: Optional[float] = None

    if (
        experiment_cfg.training.sigma_eval_enabled
        and experiment_cfg.training.reconstruction_epochs > 0
        and sigma_eval_logger is not None
    ):
        warmup_summary = _run_sigma_evaluation(
            experiment_cfg,
            generator,
            dataset,
            feature_pool,
            device,
            0,
            run,
            sigma_eval_logger,
            position_stats=position_stats,
        )
        if warmup_summary is not None:
            sigma_eval_records.append(warmup_summary)
            warmup_error = warmup_summary.get("avg_error_rate")
            warmup_acc = warmup_summary.get("avg_accuracy")
            if warmup_acc is not None:
                last_sigma_accuracy = float(warmup_acc)
                next_epoch_idx = 1
                if next_epoch_idx >= min_freeze_epoch:
                    if upper_sigma_thresh is not None and warmup_acc >= upper_sigma_thresh:
                        freeze_discriminator_next = True
                        freeze_generator_next = False
                    elif lower_sigma_thresh is not None and warmup_acc <= lower_sigma_thresh:
                        freeze_generator_next = True
                        freeze_discriminator_next = False
            if warmup_error is not None and warmup_error + 1e-6 < best_sigma_error:
                best_sigma_error = warmup_error
                best_sigma_epoch = 0
                _save_checkpoints(generator, discriminator, experiment_cfg, suffix="_best")
                logger.info("Warm-up sigma error %.6f", warmup_error)

    use_absolute = experiment_cfg.training.absolute_coordinates
    is_seq2seq = experiment_cfg.model.architecture == "seq2seq"
    dt_value = fixed_dt
    dt_value_norm: Optional[float] = None
    if dt_value is not None:
        if experiment_cfg.data.normalize_sequences:
            seq_mean, seq_std = dataset.get_sequence_stats()
            dt_mean = seq_mean[2].item()
            dt_std = seq_std[2].item()
            if dt_std <= 0:
                dt_value_norm = 0.0
            else:
                dt_value_norm = (dt_value - dt_mean) / dt_std
        else:
            dt_value_norm = dt_value


    for epoch in range(1, experiment_cfg.training.epochs + 1):
        if fake_ratio_warmup <= 0:
            fake_batch_ratio = fake_ratio_target
        else:
            progress = min(1.0, max(0.0, (epoch - 1) / max(1, fake_ratio_warmup)))
            fake_batch_ratio = fake_ratio_start + (fake_ratio_target - fake_ratio_start) * progress
        fake_batch_ratio = float(max(1e-3, min(0.95, fake_batch_ratio)))
        freeze_discriminator_epoch = freeze_discriminator_next
        freeze_generator_epoch = freeze_generator_next
        freeze_discriminator_next = False
        freeze_generator_next = False
        if discriminator_cooldown > 0:
            discriminator_cooldown -= 1
        if generator_cooldown > 0:
            generator_cooldown -= 1
        disc_accuracy_epoch = None

        if freeze_discriminator_epoch:
            logger.info(
                "Epoch %d: freezing discriminator (sigma_accuracy=%.4f)",
                epoch,
                float(last_sigma_accuracy) if last_sigma_accuracy is not None else float("nan"),
            )
        if freeze_generator_epoch:
            logger.info(
                "Epoch %d: freezing generator (sigma_accuracy=%.4f)",
                epoch,
                float(last_sigma_accuracy) if last_sigma_accuracy is not None else float("nan"),
            )

        epoch_stats = {
            "real_elem": 0,
            "real_dx_sum": 0.0,
            "real_dy_sum": 0.0,
            "real_theta_sum": 0.0,
            "real_theta_sq_sum": 0.0,
            "real_seq_count": 0,
            "gen_elem": 0,
            "gen_dx_sum": 0.0,
            "gen_dy_sum": 0.0,
            "gen_theta_sum": 0.0,
            "gen_theta_sq_sum": 0.0,
            "gen_seq_count": 0,
            "disc_real_correct": 0,
            "disc_fake_correct": 0,
            "disc_real_total": 0,
            "disc_fake_total": 0,
            "disc_fake_ratio_sum": 0.0,
            "disc_updates": 0,
        }
        for batch in dataloader:
            real_sequences, features, _ = _to_device(batch, device)
            if dt_value is not None and real_sequences.size(-1) >= 3:
                real_sequences = _apply_fixed_dt(real_sequences, dataset, dt_value)
            condition = _get_condition(features, experiment_cfg.model.generator.condition_dim)
            disc_condition_dim = experiment_cfg.model.discriminator.condition_dim
            disc_cond_real = _get_condition(features, disc_condition_dim)

            with torch.no_grad():
                real_features_raw_cpu = dataset.denormalize_features(features.detach().cpu())
                geometry_extra_cpu = (
                    real_features_raw_cpu[:, -dataset._geometry_dim :] if getattr(dataset, "_geometry_dim", 0) > 0 else None
                )

            if use_absolute:
                real_pos = _deltas_to_positions_xy(real_sequences)
                real_disc_sequences = _standardize_positions(real_pos)
            else:
                real_disc_sequences = _canonicalize_tensor_sequences(real_sequences, experiment_cfg.data)
            real_phys = dataset.denormalize_sequences(real_sequences.detach().cpu())
            real_deltas = real_phys[..., :2]
            epoch_stats["real_elem"] += real_deltas[..., 0].numel()
            epoch_stats["real_dx_sum"] += float(real_deltas[..., 0].sum().item())
            epoch_stats["real_dy_sum"] += float(real_deltas[..., 1].sum().item())
            real_first = real_deltas[:, 0, :]
            real_denom = torch.where(real_first[:, 0].abs() < 1e-6, torch.full_like(real_first[:, 0], 1e-6), real_first[:, 0])
            real_theta_batch = torch.atan2(real_first[:, 1], real_denom)
            epoch_stats["real_theta_sum"] += float(real_theta_batch.sum().item())
            epoch_stats["real_theta_sq_sum"] += float((real_theta_batch ** 2).sum().item())
            epoch_stats["real_seq_count"] += real_first.size(0)

            need_r1 = (not use_wgan) and training_cfg.r1_gamma > 0
            real_disc_input = real_disc_sequences.clone().detach()
            if need_r1:
                real_disc_input.requires_grad_(True)

            if freeze_discriminator_epoch:
                discriminator.eval()
                _set_requires_grad(discriminator, False)
                d_loss = torch.zeros((), device=device)
            else:
                discriminator.train()
                generator.eval()
                _set_requires_grad(discriminator, True)
                _set_requires_grad(generator, False)

                for _ in range(experiment_cfg.training.discriminator_steps):
                    using_noise = epoch <= cold_start_epochs
                    fake_positions: Optional[torch.Tensor] = None
                    geometry_extra_fake_cpu: Optional[torch.Tensor]
                    if using_noise:
                        feature_sequence = _sample_noise_sequences(
                            real_sequences,
                            use_absolute=False,
                            dt_value=dt_value,
                            dt_value_norm=dt_value_norm,
                            canonicalize=not use_absolute,
                            experiment_cfg=experiment_cfg,
                        )
                        deltas_xy = feature_sequence[..., :2]
                        geometry_extra_fake_cpu = geometry_extra_cpu
                    else:
                        real_count = real_sequences.size(0)
                        fake_batch_size = real_count
                        if abs(fake_batch_ratio - 0.5) > 1e-6:
                            fake_batch_size = max(
                                1,
                                int(
                                    math.ceil(
                                        real_count * fake_batch_ratio / max(1.0 - fake_batch_ratio, 1e-3)
                                    )
                                ),
                            )
                        if fake_batch_size == real_count:
                            cond_fake = condition
                            geometry_extra_fake_cpu = geometry_extra_cpu
                        else:
                            idx_fake = torch.randint(0, feature_pool.size(0), (fake_batch_size,), device=device)
                            cond_fake = _get_condition(feature_pool[idx_fake], experiment_cfg.model.generator.condition_dim)
                            if geometry_feature_pool is not None:
                                idx_cpu = idx_fake.cpu()
                                geometry_extra_fake_cpu = geometry_feature_pool[idx_cpu].clone()
                            else:
                                geometry_extra_fake_cpu = None
                        z = torch.randn(fake_batch_size, experiment_cfg.model.generator.latent_dim, device=device)
                        fake_raw = generator(z, cond_fake).detach()
                        if use_absolute:
                            if is_seq2seq:
                                if pos_mean is not None and pos_std is not None:
                                    fake_positions = fake_raw * (pos_std + 1e-6) + pos_mean
                                else:
                                    fake_positions = fake_raw
                                deltas_xy = _positions_to_deltas_xy(fake_positions)
                            else:
                                fake_positions = _deltas_to_positions_xy(fake_raw[..., :2])
                                deltas_xy = fake_raw[..., :2]
                        else:
                            fake_positions = None
                            if is_seq2seq:
                                deltas_xy = fake_raw
                            else:
                                deltas_xy = fake_raw[..., :2]
                        if real_sequences.size(-1) >= 3:
                            if fake_batch_size == real_count:
                                dt_col = real_sequences[..., 2:3].detach()
                            else:
                                dt_fill = dt_value_norm if dt_value_norm is not None else (dt_value if dt_value is not None else 1.0)
                                dt_col = torch.full(
                                    (deltas_xy.size(0), deltas_xy.size(1), 1),
                                    float(dt_fill),
                                    device=deltas_xy.device,
                                    dtype=deltas_xy.dtype,
                                )
                        else:
                            dt_fill = dt_value_norm if dt_value_norm is not None else (dt_value if dt_value is not None else 1.0)
                            dt_col = torch.full_like(deltas_xy[..., :1], float(dt_fill))
                        feature_sequence = torch.cat([deltas_xy, dt_col], dim=-1)

                    if feature_sequence.size(-1) > real_sequences.size(-1):
                        feature_sequence = feature_sequence[..., : real_sequences.size(-1)]
                    elif feature_sequence.size(-1) < real_sequences.size(-1):
                        pad = torch.zeros(
                            feature_sequence.size(0),
                            feature_sequence.size(1),
                            real_sequences.size(-1) - feature_sequence.size(-1),
                            device=feature_sequence.device,
                            dtype=feature_sequence.dtype,
                        )
                        feature_sequence = torch.cat([feature_sequence, pad], dim=-1)

                    if use_absolute:
                        if fake_positions is None:
                            disc_positions = _deltas_to_positions_xy(feature_sequence[..., :2])
                        else:
                            disc_positions = fake_positions
                        fake_disc_input = _standardize_positions(disc_positions)
                    else:
                        fake_disc_input = _canonicalize_tensor_sequences(feature_sequence, experiment_cfg.data)

                    if geometry_extra_fake_cpu is not None:
                        if geometry_extra_fake_cpu.size(0) != feature_sequence.size(0):
                            geometry_extra_batch = geometry_extra_fake_cpu[: feature_sequence.size(0)]
                        else:
                            geometry_extra_batch = geometry_extra_fake_cpu
                    else:
                        geometry_extra_batch = None

                    fake_feature_norm = _feature_tensor_from_sequences(
                        dataset,
                        dataset.denormalize_sequences(feature_sequence.detach()),
                        normalize=True,
                        extra_features=geometry_extra_batch,
                        feature_workers=experiment_cfg.training.feature_workers,
                        feature_device=feature_device,
                        target_device=device,
                    )
                    cond_disc_fake = _get_condition(fake_feature_norm, disc_condition_dim)

                    if ada_enabled:
                        real_disc_eval = _apply_sequence_augmentations(real_disc_input, ada_p)
                        fake_disc_eval = _apply_sequence_augmentations(fake_disc_input, ada_p)
                    else:
                        real_disc_eval = real_disc_input
                        fake_disc_eval = fake_disc_input

                    real_scores, _ = discriminator(real_disc_eval, disc_cond_real)
                    fake_scores, _ = discriminator(fake_disc_eval, cond_disc_fake)
                    real_scores_detached = real_scores.detach()
                    fake_scores_detached = fake_scores.detach()
                    real_correct = (real_scores_detached > 0).sum().item()
                    fake_correct = (fake_scores_detached < 0).sum().item()
                    epoch_stats["disc_real_correct"] += int(real_correct)
                    epoch_stats["disc_fake_correct"] += int(fake_correct)
                    epoch_stats["disc_real_total"] += real_scores.numel()
                    epoch_stats["disc_fake_total"] += fake_scores.numel()
                    batch_total = real_scores.numel() + fake_scores.numel()
                    if batch_total > 0:
                        epoch_stats["disc_fake_ratio_sum"] += fake_scores.numel() / batch_total
                        epoch_stats["disc_updates"] += 1
                    if adaptive_enabled and batch_total > 0:
                        batch_accuracy = (real_correct + fake_correct) / batch_total
                        if disc_accuracy_ema is None:
                            disc_accuracy_ema = float(batch_accuracy)
                        else:
                            disc_accuracy_ema += adaptive_smoothing * (float(batch_accuracy) - disc_accuracy_ema)
                    if ada_enabled and batch_total > 0:
                        batch_accuracy = (real_correct + fake_correct) / batch_total
                        ada_accumulator += float(batch_accuracy)
                        ada_counter += 1
                        if ada_counter >= ada_interval:
                            mean_acc = ada_accumulator / ada_counter
                            ada_delta = ada_rate * (mean_acc - ada_target)
                            ada_p = float(max(0.0, min(ada_p + ada_delta, ada_p_max)))
                            ada_accumulator = 0.0
                            ada_counter = 0
                    if use_wgan:
                        gp = _compute_gradient_penalty(
                            discriminator,
                            real_disc_eval,
                            fake_disc_eval,
                            condition,
                            device,
                        )
                        d_loss = fake_scores.mean() - real_scores.mean() + experiment_cfg.training.gradient_penalty_weight * gp
                    else:
                        assert bce_loss is not None
                        real_label_val = float(training_cfg.label_smoothing_real)
                        fake_label_val = float(training_cfg.label_smoothing_fake)
                        real_labels = torch.full_like(real_scores, real_label_val)
                        fake_labels = torch.full_like(fake_scores, fake_label_val)
                        real_loss = bce_loss(real_scores, real_labels)
                        fake_loss = bce_loss(fake_scores, fake_labels)
                        d_loss = 0.5 * (real_loss + fake_loss)

                        if need_r1:
                            r1_grad = torch.autograd.grad(
                                real_scores.sum(),
                                real_disc_input,
                                create_graph=True,
                                retain_graph=True,
                            )[0]
                            r1_penalty = r1_grad.pow(2).reshape(r1_grad.size(0), -1).sum(dim=1).mean()
                            d_loss = d_loss + 0.5 * float(training_cfg.r1_gamma) * r1_penalty

                    d_optimizer.zero_grad()
                    d_loss.backward()
                    d_optimizer.step()

            generator_update = epoch > cold_start_epochs and not freeze_generator_epoch
            if generator_update:
                generator.train()
                _set_requires_grad(discriminator, False)
                _set_requires_grad(generator, True)
            else:
                generator.eval()
                _set_requires_grad(generator, False)
                _set_requires_grad(discriminator, False)

            using_noise_for_logging = epoch <= cold_start_epochs
            if epoch <= cold_start_epochs:
                generated_raw = None
                deltas_xy = torch.randn_like(real_sequences[..., :2])
                if real_sequences.size(-1) >= 3:
                    if dt_value_norm is not None:
                        dt_fill = dt_value_norm
                    elif dt_value is not None:
                        dt_fill = dt_value
                    else:
                        dt_fill = 1.0
                    dt_col = torch.full_like(deltas_xy[..., :1], float(dt_fill))
                else:
                    dt_col = torch.empty(0)
                if dt_col.numel() > 0:
                    generated_sequence = torch.cat([deltas_xy, dt_col], dim=-1)
                else:
                    generated_sequence = deltas_xy
                if use_absolute:
                    fake_positions = _deltas_to_positions_xy(deltas_xy)
                    disc_fake_input = _standardize_positions(fake_positions)
                else:
                    disc_fake_input = _canonicalize_tensor_sequences(generated_sequence, experiment_cfg.data)
                g_loss = torch.zeros((), device=device, dtype=real_disc_sequences.dtype)
            else:
                using_noise_for_logging = False
                z = torch.randn(real_sequences.size(0), experiment_cfg.model.generator.latent_dim, device=device)
                if generator_update:
                    generated_raw = generator(z, condition)
                else:
                    with torch.no_grad():
                        generated_raw = generator(z, condition)
                if use_absolute:
                    if is_seq2seq:
                        if pos_mean is not None and pos_std is not None:
                            fake_positions = generated_raw * (pos_std + 1e-6) + pos_mean
                        else:
                            fake_positions = generated_raw
                        deltas_xy = _positions_to_deltas_xy(fake_positions)
                    else:
                        fake_positions = _deltas_to_positions_xy(generated_raw[..., :2])
                        deltas_xy = generated_raw[..., :2]
                    if real_sequences.size(-1) >= 3:
                        dt_col = real_sequences[..., 2:3].detach()
                    else:
                        dt_fill = dt_value_norm if dt_value_norm is not None else (dt_value if dt_value is not None else 1.0)
                        dt_col = torch.full_like(deltas_xy[..., :1], float(dt_fill))
                    generated_sequence = torch.cat([deltas_xy, dt_col], dim=-1)
                    disc_fake_input = _standardize_positions(fake_positions)
                else:
                    if is_seq2seq:
                        if pos_mean is not None and pos_std is not None:
                            fake_positions = generated_raw * (pos_std + 1e-6) + pos_mean
                            deltas_xy = _positions_to_deltas_xy(fake_positions)
                        else:
                            fake_positions = None
                            deltas_xy = generated_raw
                    else:
                        fake_positions = None
                        deltas_xy = generated_raw[..., :2]
                    if generated_raw.size(-1) >= 3:
                        generated_sequence = generated_raw
                    else:
                        if real_sequences.size(-1) >= 3:
                            dt_col = real_sequences[..., 2:3].detach()
                        else:
                            dt_fill = dt_value_norm if dt_value_norm is not None else (dt_value if dt_value is not None else 1.0)
                            dt_col = torch.full_like(deltas_xy[..., :1], float(dt_fill))
                        generated_sequence = torch.cat([deltas_xy, dt_col], dim=-1)
                    disc_fake_input = _canonicalize_tensor_sequences(generated_sequence, experiment_cfg.data)

                if dt_value_norm is not None and generated_sequence.size(-1) >= 3:
                    generated_sequence = generated_sequence.clone()
                    generated_sequence[..., 2] = float(dt_value_norm)
                    if not use_absolute:
                        disc_fake_input = _canonicalize_tensor_sequences(generated_sequence, experiment_cfg.data)

                if geometry_extra_cpu is not None and geometry_extra_cpu.size(0) != generated_sequence.size(0):
                    geometry_extra_batch = geometry_extra_cpu[: generated_sequence.size(0)]
                else:
                    geometry_extra_batch = geometry_extra_cpu
                fake_feature_norm = _feature_tensor_from_sequences(
                    dataset,
                    dataset.denormalize_sequences(generated_sequence.detach()),
                    normalize=True,
                    extra_features=geometry_extra_batch,
                    feature_workers=experiment_cfg.training.feature_workers,
                    feature_device=feature_device,
                    target_device=device,
                )
                cond_disc_fake = _get_condition(fake_feature_norm, disc_condition_dim)

                if ada_enabled:
                    disc_fake_eval = _apply_sequence_augmentations(disc_fake_input, ada_p)
                else:
                    disc_fake_eval = disc_fake_input

                fake_scores, _ = discriminator(disc_fake_eval, cond_disc_fake)
                if generator_update:
                    if use_wgan:
                        g_loss = -fake_scores.mean()
                    else:
                        assert bce_loss is not None
                        target_labels = torch.full_like(fake_scores, float(training_cfg.label_smoothing_real))
                        g_loss = bce_loss(fake_scores, target_labels)
                else:
                    g_loss = torch.zeros((), device=device, dtype=fake_scores.dtype)

            curvature_penalty_value = None
            lateral_penalty_value = None
            direction_penalty_value = None

            need_denorm = (
                experiment_cfg.training.curvature_match_weight > 0
                or experiment_cfg.training.lateral_match_weight > 0
                or experiment_cfg.training.direction_match_weight > 0
            )
            if epoch > cold_start_epochs and need_denorm:
                real_denorm = dataset.denormalize_sequences(real_sequences)
                fake_denorm = dataset.denormalize_sequences(generated_sequence)
            else:
                real_denorm = fake_denorm = None

            if epoch > cold_start_epochs and need_denorm:
                real_xy = real_denorm[..., :2]
                fake_xy = fake_denorm[..., :2]

            if (
                epoch > cold_start_epochs
                and (
                    experiment_cfg.training.curvature_match_weight > 0
                    or experiment_cfg.training.lateral_match_weight > 0
                )
            ):
                real_rot_deltas, real_rot_pos = _canonical_frame(real_denorm)
                fake_rot_deltas, fake_rot_pos = _canonical_frame(fake_denorm)
                progress_real = _progress_along_path(real_rot_pos)
                progress_fake = _progress_along_path(fake_rot_pos)
                progress_step_real = 0.5 * (progress_real[:, 1:] + progress_real[:, :-1])
                progress_step_fake = 0.5 * (progress_fake[:, 1:] + progress_fake[:, :-1])
                curv_weights_real = _end_segment_weights(progress_step_real)
                curv_weights_fake = _end_segment_weights(progress_step_fake)
                lat_weights_real = _end_segment_weights(progress_real)
                lat_weights_fake = _end_segment_weights(progress_fake)

                if experiment_cfg.training.curvature_match_weight > 0:
                    real_curv = _mean_abs_heading_change(real_rot_deltas, weights=curv_weights_real)
                    fake_curv = _mean_abs_heading_change(fake_rot_deltas, weights=curv_weights_fake)
                    curvature_penalty = torch.abs(fake_curv.mean() - real_curv.mean())
                    g_loss = g_loss + experiment_cfg.training.curvature_match_weight * curvature_penalty
                    curvature_penalty_value = float(curvature_penalty.detach().item())

                if experiment_cfg.training.lateral_match_weight > 0:
                    real_lat = _lateral_rms(real_rot_pos, weights=lat_weights_real)
                    fake_lat = _lateral_rms(fake_rot_pos, weights=lat_weights_fake)
                    lateral_penalty = torch.abs(fake_lat.mean() - real_lat.mean())
                    g_loss = g_loss + experiment_cfg.training.lateral_match_weight * lateral_penalty
                    lateral_penalty_value = float(lateral_penalty.detach().item())

            if epoch > cold_start_epochs and experiment_cfg.training.direction_match_weight > 0:
                real_variation = _delta_variation(real_xy)
                fake_variation = _delta_variation(fake_xy)
                variation_diff = torch.abs(fake_variation.mean(dim=0) - real_variation.mean(dim=0)).sum()
                g_loss = g_loss + experiment_cfg.training.direction_match_weight * variation_diff
                direction_penalty_value = float(variation_diff.detach().item())

            if generator_update:
                generator.train()
                g_optimizer.zero_grad()
                g_loss.backward()
                g_optimizer.step()
            _set_requires_grad(discriminator, True)

            generated_sequence_detached = generated_sequence.detach()
            generated_metrics = _canonicalize_tensor_sequences(generated_sequence, experiment_cfg.data)
            fake_phys = dataset.denormalize_sequences(generated_sequence_detached.cpu())
            fake_deltas = fake_phys[..., :2]
            if not using_noise_for_logging:
                epoch_stats["gen_elem"] += fake_deltas[..., 0].numel()
                epoch_stats["gen_dx_sum"] += float(fake_deltas[..., 0].sum().item())
                epoch_stats["gen_dy_sum"] += float(fake_deltas[..., 1].sum().item())
                fake_first = fake_deltas[:, 0, :]
                fake_denom = torch.where(fake_first[:, 0].abs() < 1e-6, torch.full_like(fake_first[:, 0], 1e-6), fake_first[:, 0])
                fake_theta_batch = torch.atan2(fake_first[:, 1], fake_denom)
                epoch_stats["gen_theta_sum"] += float(fake_theta_batch.sum().item())
                epoch_stats["gen_theta_sq_sum"] += float((fake_theta_batch ** 2).sum().item())
                epoch_stats["gen_seq_count"] += fake_first.size(0)

            log_interval = max(1, int(experiment_cfg.training.log_interval))
            log_metrics = (step == 0) or (step % log_interval == 0)
            metric_interval_cfg = experiment_cfg.training.metric_log_interval
            metric_interval = max(1, int(metric_interval_cfg)) if metric_interval_cfg else log_interval
            heavy_metrics = (step == 0) or (step % metric_interval == 0)
            sample_output = step % experiment_cfg.training.sample_interval == 0

            d_loss_value = float(d_loss.detach().cpu())
            g_loss_value = float(g_loss.detach().cpu())

            metrics_payload = {
                "epoch": epoch,
                "step": step,
                "d_loss": d_loss_value,
                "g_loss": g_loss_value,
                "freeze_discriminator": int(freeze_discriminator_epoch),
                "freeze_generator": int(freeze_generator_epoch),
            }
            if last_sigma_accuracy is not None:
                metrics_payload["sigma_accuracy_last"] = float(last_sigma_accuracy)

            if curvature_penalty_value is not None:
                metrics_payload["curvature_penalty"] = curvature_penalty_value
            if lateral_penalty_value is not None:
                metrics_payload["lateral_penalty"] = lateral_penalty_value
            if direction_penalty_value is not None:
                metrics_payload["direction_penalty"] = direction_penalty_value

            denorm_generated = None
            if sample_output or heavy_metrics:
                generated_cpu = generated_metrics.detach().cpu()
                denorm_generated = dataset.denormalize_sequences(generated_cpu)

            if heavy_metrics:
                theta_real = _theta_start(real_phys)
                theta_fake = _theta_start(fake_phys)
                eff_real = _path_efficiency(real_phys)
                eff_fake = _path_efficiency(fake_phys)
                jerk_real = _jerk_magnitude(real_phys).mean().item()
                jerk_fake = _jerk_magnitude(fake_phys).mean().item()

                with torch.no_grad():
                    extra_raw = None
                    if dataset.config.include_goal_geometry:
                        extra_raw = dataset.denormalize_features(features.detach()).cpu()
                    fake_feature_batch = _feature_tensor_from_sequences(
                        dataset,
                        denorm_generated,
                        extra_features=extra_raw,
                        feature_workers=experiment_cfg.training.feature_workers,
                    )
                    fd_metrics = feature_distribution_metrics(
                        real_feature_np,
                        fake_feature_batch.numpy(),
                    )
                    diversity = sequence_diversity_metric(denorm_generated.numpy())
                metrics_payload.update(
                    {
                        "feature_l1": fd_metrics["mean_l1"],
                        "feature_cov_diff": fd_metrics["cov_trace_diff"],
                        "diversity_xy": diversity,
                        "theta_start_real": float(theta_real.mean().item()),
                        "theta_start_fake": float(theta_fake.mean().item()),
                        "path_eff_real": float(eff_real.mean().item()),
                        "path_eff_fake": float(eff_fake.mean().item()),
                        "jerk_real": jerk_real,
                        "jerk_fake": jerk_fake,
                    }
                )
                if run is not None:
                    run.log(metrics_payload)
                logger.info(
                    "Step %d | d_loss=%.4f g_loss=%.4f feature_l1=%.4f cov_diff=%.4f diversity=%.4f",
                    step,
                    d_loss.item(),
                    g_loss.item(),
                    metrics_payload["feature_l1"],
                    metrics_payload["feature_cov_diff"],
                    metrics_payload["diversity_xy"],
                )

            if sample_output and denorm_generated is not None:
                sample_path = _save_generated_samples(denorm_generated[:16], sample_dir, epoch, step)
                metrics_payload["sample_path"] = str(sample_path)

            if metrics_logger is not None and (log_metrics or sample_output):
                metrics_logger.log(metrics_payload)
            if profiler is not None:
                profiler.step()
            last_metrics = metrics_payload
            step += 1

        def _finalize_stats(prefix: str, elem: int, dx_sum: float, dy_sum: float, theta_sum: float, theta_sq_sum: float, count: int) -> dict[str, float]:
            stats: dict[str, float] = {}
            if elem > 0:
                stats[f"{prefix}_dx_mean"] = dx_sum / elem
                stats[f"{prefix}_dy_mean"] = dy_sum / elem
            if count > 0:
                theta_mean = theta_sum / count
                theta_var = max(theta_sq_sum / count - theta_mean * theta_mean, 0.0)
                stats[f"{prefix}_theta_mean"] = theta_mean
                stats[f"{prefix}_theta_std"] = theta_var ** 0.5
            return stats

        real_epoch_stats = _finalize_stats(
            "real",
            epoch_stats["real_elem"],
            epoch_stats["real_dx_sum"],
            epoch_stats["real_dy_sum"],
            epoch_stats["real_theta_sum"],
            epoch_stats["real_theta_sq_sum"],
            epoch_stats["real_seq_count"],
        )
        gen_epoch_stats = _finalize_stats(
            "gen",
            epoch_stats["gen_elem"],
            epoch_stats["gen_dx_sum"],
            epoch_stats["gen_dy_sum"],
            epoch_stats["gen_theta_sum"],
            epoch_stats["gen_theta_sq_sum"],
            epoch_stats["gen_seq_count"],
        )
        if gen_epoch_stats or real_epoch_stats:
            logger.info(
                "Epoch %d diagnostics | %s",
                epoch,
                ", ".join([f"{k}={v:.4f}" for stats in (real_epoch_stats, gen_epoch_stats) for k, v in stats.items()]),
            )
        disc_real_total = epoch_stats["disc_real_total"]
        disc_fake_total = epoch_stats["disc_fake_total"]
        disc_stats: dict[str, float] = {}
        if disc_real_total + disc_fake_total > 0:
            disc_real_acc = epoch_stats["disc_real_correct"] / max(1, disc_real_total)
            disc_fake_acc = epoch_stats["disc_fake_correct"] / max(1, disc_fake_total)
            disc_accuracy = (epoch_stats["disc_real_correct"] + epoch_stats["disc_fake_correct"]) / (disc_real_total + disc_fake_total)
            disc_accuracy_epoch = disc_accuracy
            disc_stats = {
                "overall": disc_accuracy,
                "real": disc_real_acc,
                "fake": disc_fake_acc,
            }
            log_parts = [
                f"overall={disc_accuracy:.4f}",
                f"real={disc_real_acc:.4f}",
                f"fake={disc_fake_acc:.4f}",
            ]
            if epoch_stats["disc_updates"] > 0:
                fake_ratio_avg = epoch_stats["disc_fake_ratio_sum"] / max(1, epoch_stats["disc_updates"])
                disc_stats["fake_ratio"] = fake_ratio_avg
                log_parts.append(f"fake_ratio={fake_ratio_avg:.3f}")
            if adaptive_enabled and disc_accuracy_ema is not None:
                disc_stats["ema"] = disc_accuracy_ema
                log_parts.append(f"ema={disc_accuracy_ema:.4f}")
            if ada_enabled:
                disc_stats["ada_p"] = ada_p
                log_parts.append(f"ada_p={ada_p:.4f}")
            logger.info("Epoch %d discriminator accuracy | %s", epoch, " ".join(log_parts))
        adaptive_delta = None
        if adaptive_enabled and disc_accuracy_ema is not None:
            adaptive_delta = disc_accuracy_ema - adaptive_target

        adaptive_lr_log_value: Optional[float] = None
        if (
            adaptive_lr_enabled
            and disc_accuracy_epoch is not None
            and epoch >= adaptive_lr_warmup
            and not freeze_discriminator_epoch
        ):
            error = disc_accuracy_epoch - adaptive_lr_target
            derivative = 0.0
            if prev_disc_accuracy_epoch is not None:
                derivative = disc_accuracy_epoch - prev_disc_accuracy_epoch
            lr_current = float(d_optimizer.param_groups[0]["lr"])
            delta = -(adaptive_lr_error_gain * error + adaptive_lr_derivative_gain * derivative)
            new_lr = lr_current * math.exp(delta)
            new_lr = float(max(adaptive_lr_min, min(adaptive_lr_max, new_lr)))
            for group in d_optimizer.param_groups:
                group["lr"] = new_lr
            adaptive_lr_log_value = new_lr

        if disc_accuracy_epoch is not None:
            prev_disc_accuracy_epoch = float(disc_accuracy_epoch)
        if run is not None and (gen_epoch_stats or real_epoch_stats):
            payload = {"epoch_diag": epoch}
            payload.update({f"real/{k.split('_', 1)[1]}": v for k, v in real_epoch_stats.items()})
            payload.update({f"gen/{k.split('_', 1)[1]}": v for k, v in gen_epoch_stats.items()})
            if disc_stats:
                payload["disc/accuracy"] = disc_stats["overall"]
                payload["disc/real_accuracy"] = disc_stats["real"]
                payload["disc/fake_accuracy"] = disc_stats["fake"]
                if "fake_ratio" in disc_stats:
                    payload["disc/fake_ratio"] = disc_stats["fake_ratio"]
                if "ema" in disc_stats:
                    payload["disc/accuracy_ema"] = disc_stats["ema"]
                if "ada_p" in disc_stats:
                    payload["control/ada_p"] = disc_stats["ada_p"]
            elif adaptive_enabled and disc_accuracy_ema is not None:
                payload["disc/accuracy_ema"] = disc_accuracy_ema
            if adaptive_delta is not None:
                payload["control/adaptive_delta"] = adaptive_delta
            elif ada_enabled:
                payload["control/ada_p"] = ada_p
            if adaptive_lr_log_value is not None:
                payload["control/d_lr"] = adaptive_lr_log_value
            run.log(payload)

        # optional epoch summary log
        if run is not None:
            epoch_log = {
                "epoch": epoch,
                "d_loss": d_loss.item(),
                "g_loss": g_loss.item(),
                "control/freeze_discriminator": int(freeze_discriminator_epoch),
                "control/freeze_generator": int(freeze_generator_epoch),
            }
            if disc_stats:
                epoch_log["disc/accuracy_epoch"] = disc_stats["overall"]
                epoch_log["disc/real_accuracy_epoch"] = disc_stats["real"]
                epoch_log["disc/fake_accuracy_epoch"] = disc_stats["fake"]
                if "fake_ratio" in disc_stats:
                    epoch_log["disc/fake_ratio_epoch"] = disc_stats["fake_ratio"]
            if adaptive_enabled and disc_accuracy_ema is not None:
                epoch_log["disc/accuracy_ema"] = disc_accuracy_ema
            if adaptive_delta is not None:
                epoch_log["control/adaptive_delta"] = adaptive_delta
            if ada_enabled:
                epoch_log["control/ada_p"] = ada_p
            if adaptive_lr_log_value is not None:
                epoch_log["control/d_lr"] = adaptive_lr_log_value
            if last_sigma_accuracy is not None:
                epoch_log["control/sigma_accuracy_last"] = float(last_sigma_accuracy)
            run.log(epoch_log)

        sigma_summary = _run_sigma_evaluation(
            experiment_cfg,
            generator,
            dataset,
            feature_pool,
            device,
            epoch,
            run,
            sigma_eval_logger,
            position_stats=position_stats,
        )
        if sigma_summary is not None:
            sigma_eval_records.append(sigma_summary)
            sigma_acc = sigma_summary.get("avg_accuracy")
            if sigma_acc is not None:
                last_sigma_accuracy = float(sigma_acc)
                next_epoch_idx = epoch + 1
                if next_epoch_idx >= min_freeze_epoch:
                    if upper_sigma_thresh is not None and sigma_acc >= upper_sigma_thresh:
                        freeze_discriminator_next = True
                        freeze_generator_next = False
                    elif lower_sigma_thresh is not None and sigma_acc <= lower_sigma_thresh:
                        freeze_generator_next = True
                        freeze_discriminator_next = False
            replay_dir_str = sigma_summary.get("replay_dir")
            if (
                real_sequences_plot is not None
                and real_sequences_plot.shape[0] > 0
                and real_features_plot.size > 0
                and replay_dir_str
            and training_cfg.sigma_eval_make_plots
            ):
                try:
                    plot_dir = Path(replay_dir_str) / "plots"
                    feature_samples = min(1024, real_features_plot.shape[0])
                    plot_count = min(24, real_sequences_plot.shape[0])
                    plot_summary = generate_replay_vs_real_plots(
                        real_sequences_plot,
                        real_features_plot,
                        replay_dir_str,
                        plot_dir,
                        seed=experiment_cfg.seed + epoch,
                        plot_count=plot_count,
                        feature_samples=feature_samples,
                        sequence_length=experiment_cfg.data.sequence_length,
                    )
                    sigma_summary["plot_summary_path"] = str(plot_dir / "histogram_summary.json")
                    logger.info(
                        "Generated replay diagnostics at %s", plot_dir
                    )
                    # Also copy plots into the active Hydra run directory for quick inspection.
                    local_epoch_dir = Path.cwd() / "plots_comparison" / f"epoch_{epoch:03d}"
                    local_epoch_dir.mkdir(parents=True, exist_ok=True)
                    for key in ("sequence_plot_path", "feature_hist_path"):
                        src_path = Path(plot_summary[key])
                        if src_path.exists():
                            shutil.copy2(src_path, local_epoch_dir / src_path.name)
                    summary_src = Path(plot_dir / "histogram_summary.json")
                    if summary_src.exists():
                        shutil.copy2(summary_src, local_epoch_dir / summary_src.name)
                    sigma_summary["plots_local_dir"] = str(local_epoch_dir)
                    sigma_summary.update(
                        {
                            "trajectory_plot_path": plot_summary["sequence_plot_path"],
                            "feature_hist_path": plot_summary["feature_hist_path"],
                        }
                    )
                except Exception as exc:
                    logger.warning("Replay diagnostics generation failed: %s", exc)
            error_rate = sigma_summary.get("avg_error_rate")
            if error_rate is not None and error_rate + 1e-6 < best_sigma_error:
                best_sigma_error = error_rate
                best_sigma_epoch = epoch
                _save_checkpoints(generator, discriminator, experiment_cfg, suffix="_best")
                logger.info("New best sigma error %.6f at epoch %d", error_rate, epoch)

        diffusion_summary = _run_diffusion_evaluation(
            experiment_cfg,
            dataset,
            device,
            epoch,
            run,
            diffusion_eval_logger,
            real_features=real_feature_cpu,
        )
        if diffusion_summary is not None:
            diffusion_eval_records.append(diffusion_summary)

        if adaptive_enabled:
            ready_for_adaptive = epoch >= adaptive_warmup
            if ready_for_adaptive and disc_accuracy_ema is not None:
                delta = disc_accuracy_ema - adaptive_target
                if delta > adaptive_margin:
                    if discriminator_cooldown == 0:
                        freeze_discriminator_next = True
                        freeze_generator_next = False
                        logger.info(
                            "Adaptive freeze: keeping discriminator frozen (ema=%.4f target=%.2f)",
                            disc_accuracy_ema,
                            adaptive_target,
                        )
                    else:
                        logger.debug(
                            "Adaptive freeze: discriminator in cooldown (%d epochs remaining), ema=%.4f",
                            discriminator_cooldown,
                            disc_accuracy_ema,
                        )
                elif delta < -adaptive_margin:
                    if adaptive_freeze_generator and not freeze_generator_next and generator_cooldown == 0:
                        freeze_generator_next = True
                        generator_cooldown = adaptive_cooldown_cfg
                        if adaptive_cooldown_cfg > 0:
                            logger.info(
                                "Adaptive freeze: scheduling generator freeze (ema=%.4f target=%.2f cooldown=%d)",
                                disc_accuracy_ema,
                                adaptive_target,
                                adaptive_cooldown_cfg,
                            )
                        else:
                            logger.info(
                                "Adaptive freeze: scheduling generator freeze (ema=%.4f target=%.2f)",
                                disc_accuracy_ema,
                                adaptive_target,
                            )
                    if discriminator_cooldown == 0 and freeze_discriminator_next:
                        logger.info(
                            "Adaptive freeze: releasing discriminator (ema=%.4f target=%.2f)",
                            disc_accuracy_ema,
                            adaptive_target,
                        )
                        freeze_discriminator_next = False
                else:
                    if freeze_discriminator_next and discriminator_cooldown == 0:
                        logger.info(
                            "Adaptive freeze: discriminator back in band (ema=%.4f target=%.2f)",
                            disc_accuracy_ema,
                            adaptive_target,
                        )
                        freeze_discriminator_next = False
        if experiment_cfg.training.replay_samples_per_epoch > 0:
            with torch.no_grad():
                idx = torch.randperm(feature_pool.size(0))[: experiment_cfg.training.replay_samples_per_epoch]
                cond_batch = _get_condition(feature_pool[idx], experiment_cfg.model.generator.condition_dim)
                z = torch.randn(cond_batch.size(0), experiment_cfg.model.generator.latent_dim, device=device)
                generated_batch_raw = generator(z, cond_batch).detach()
                if use_absolute:
                    deltas_xy = _positions_to_deltas_xy(generated_batch_raw)
                    seq = torch.zeros(cond_batch.size(0), generated_batch_raw.size(1), 3, device=device, dtype=generated_batch_raw.dtype)
                    seq[..., :2] = deltas_xy
                    if dt_value is not None:
                        seq[..., 2] = dt_value
                    else:
                        seq[..., 2] = 1.0
                    generated_batch = _canonicalize_tensor_sequences(seq, experiment_cfg.data)
                else:
                    generated_batch = _canonicalize_tensor_sequences(generated_batch_raw, experiment_cfg.data)
                generated_batch_cpu = generated_batch.cpu()
                denorm_batch = dataset.denormalize_sequences(generated_batch_cpu)
                feature_vectors = _feature_tensor_from_sequences(
                    dataset,
                    denorm_batch,
                    normalize=False,
                    feature_workers=experiment_cfg.training.feature_workers,
                    feature_device=feature_device,
                )
                buffer.add(denorm_batch, feature_vectors)
                buffer.save()

        if (
            experiment_cfg.training.co_train_detector
            and experiment_cfg.training.detector_update_every > 0
            and experiment_cfg.training.detector_config_path
            and epoch % experiment_cfg.training.detector_update_every == 0
        ):
            detector_cfg = load_detector_config(experiment_cfg.training.detector_config_path)
            detector_cfg.training.epochs = max(1, experiment_cfg.training.detector_epochs_per_update)
            detector_cfg.data.dataset_id = experiment_cfg.data.dataset_id
            detector_cfg.data.sequence_length = experiment_cfg.data.sequence_length
            detector_cfg.data.batch_size = experiment_cfg.data.batch_size
            detector_cfg.data.num_workers = experiment_cfg.data.num_workers
            detector_cfg.data.max_gestures = experiment_cfg.data.max_gestures
            detector_cfg.data.cache_enabled = experiment_cfg.data.cache_enabled
            detector_cfg.data.cache_dir = experiment_cfg.data.cache_dir
            detector_cfg.data.use_generated_negatives = True
            detector_cfg.data.replay_path = str(buffer.path)
            detector_cfg.data.replay_sample_ratio = 1.0
            detector_cfg.data.normalize_sequences = experiment_cfg.data.normalize_sequences
            detector_cfg.data.normalize_features = experiment_cfg.data.normalize_features
            detector_cfg.data.feature_mode = experiment_cfg.data.feature_mode
            detector_cfg.logging.target = "none"
            detector_cfg.logging.checkpoint_dir = experiment_cfg.logging.checkpoint_dir

            detector_metrics = run_detector_training(
                detector_cfg,
                wandb_run=run,
                metrics_logger=detector_metrics_logger,
                device=device,
            )
            if run is not None:
                prefixed = {f"detector_co_train/{k}": v for k, v in detector_metrics.items() if k != "epoch"}
                prefixed["detector_co_train/epoch"] = epoch
                run.log(prefixed)

    summary = {
        "final_epoch": experiment_cfg.training.epochs,
        "final_step": last_metrics.get("step"),
        "d_loss": last_metrics.get("d_loss"),
        "g_loss": last_metrics.get("g_loss"),
        "feature_l1": last_metrics.get("feature_l1"),
        "feature_cov_diff": last_metrics.get("feature_cov_diff"),
        "diversity_xy": last_metrics.get("diversity_xy"),
        "replay_buffer_size": len(buffer),
    }
    if sigma_eval_records:
        summary["sigma_eval"] = sigma_eval_records
        if best_sigma_epoch is not None:
            summary["best_sigma_epoch"] = best_sigma_epoch
            summary["best_sigma_error"] = best_sigma_error
    if diffusion_eval_records:
        summary["diffusion_eval"] = diffusion_eval_records
    if metrics_logger is not None:
        summary["metrics_csv"] = str(metrics_logger.path)
    return summary


def _save_checkpoints(
    generator: torch.nn.Module,
    discriminator: torch.nn.Module,
    experiment_cfg: GanExperimentConfig,
    *,
    suffix: str = "",
) -> None:
    checkpoint_dir = Path(experiment_cfg.logging.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    name = f"{experiment_cfg.experiment_name}{suffix}"
    torch.save(generator.state_dict(), checkpoint_dir / f"generator_{name}.pt")
    torch.save(discriminator.state_dict(), checkpoint_dir / f"discriminator_{name}.pt")
    if suffix:
        logger.info("Saved %s checkpoints to %s", suffix.strip("_"), checkpoint_dir)
    else:
        logger.info("Saved GAN checkpoints to %s", checkpoint_dir)


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    experiment_cfg, cfg_dict = _build_experiment_config(cfg)
    _set_seed(experiment_cfg.seed, include_cuda=False)

    dataset, dataloader = _prepare_dataset_and_dataloader(experiment_cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":  # seed CUDA after dataloader workers spawn
        logger.info("Seeding CUDA context now (device=%s)", device)
        torch.cuda.manual_seed_all(experiment_cfg.seed)
    sigma_feature_device = _select_feature_device(experiment_cfg.training.sigma_feature_device, device)
    if experiment_cfg.model.architecture == "tcn":
        logger.info("Moving generator/discriminator to %s", device)
        generator = ConditionalGenerator(experiment_cfg.model.generator).to(device)
        discriminator = GestureDiscriminator(experiment_cfg.model.discriminator).to(device)
    elif experiment_cfg.model.architecture == "lstm":
        generator = LSTMGenerator(experiment_cfg.model.generator).to(device)
        discriminator = LSTMDiscriminator(experiment_cfg.model.discriminator).to(device)
    elif experiment_cfg.model.architecture == "seq2seq":
        generator = Seq2SeqGenerator(experiment_cfg.model.generator).to(device)
        discriminator = LSTMDiscriminator(experiment_cfg.model.discriminator).to(device)
    else:
        raise ValueError(f"Unsupported model architecture: {experiment_cfg.model.architecture}")

    init_path = experiment_cfg.training.generator_init_path
    pos_mean_tensor: Optional[torch.Tensor] = None
    pos_std_tensor: Optional[torch.Tensor] = None

    if init_path:
        ckpt_path = Path(to_absolute_path(init_path))
        logger.info("Warm-start generator checking %s", ckpt_path)
        if ckpt_path.is_file():
            state = torch.load(ckpt_path, map_location=device, weights_only=False)
            state_dict = state.get("state_dict", state)
            missing, unexpected = generator.load_state_dict(state_dict, strict=False)
            if missing:
                logger.warning("Generator warm-start missing keys: %s", missing)
            if unexpected:
                logger.warning("Generator warm-start unexpected keys: %s", unexpected)
            logger.info("Loaded generator weights from %s", ckpt_path)

            if isinstance(state, dict):
                pos_mean_state = state.get("position_mean")
                pos_std_state = state.get("position_std")
                if pos_mean_state is not None and pos_std_state is not None:
                    pos_mean_tensor = torch.tensor(pos_mean_state, dtype=torch.float32).view(1, 1, -1)
                    pos_std_tensor = torch.tensor(pos_std_state, dtype=torch.float32).view(1, 1, -1)
        else:
            logger.warning("generator_init_path %s not found; skipping warm-start", ckpt_path)
    metrics_logger = CSVMetricLogger(
        Path(experiment_cfg.logging.checkpoint_dir) / "metrics.csv",
        fieldnames=[
            "epoch",
            "step",
            "d_loss",
            "g_loss",
            "curvature_penalty",
            "lateral_penalty",
            "direction_penalty",
            "feature_l1",
            "feature_cov_diff",
            "diversity_xy",
            "theta_start_real",
            "theta_start_fake",
            "path_eff_real",
            "path_eff_fake",
            "jerk_real",
            "jerk_fake",
            "freeze_discriminator",
            "freeze_generator",
            "sigma_accuracy_last",
            "sample_path",
        ],
    )

    with experiment_logger(experiment_cfg.logging, cfg_dict) as run:
        if experiment_cfg.training.absolute_coordinates:
            if pos_mean_tensor is None or pos_std_tensor is None:
                pos_mean_cpu, pos_std_cpu = _position_stats_from_dataset(dataset, 2)
                pos_mean_tensor = pos_mean_cpu[:, :, :2]
                pos_std_tensor = pos_std_cpu[:, :, :2]
            else:
                pos_mean_tensor = pos_mean_tensor[:, :, :2]
                pos_std_tensor = pos_std_tensor[:, :, :2]
        else:
            pos_mean_tensor = pos_std_tensor = None

        position_stats = None
        if pos_mean_tensor is not None and pos_std_tensor is not None:
            position_stats = (pos_mean_tensor, pos_std_tensor)

        summary = _training_loop(
            experiment_cfg,
            generator,
            discriminator,
            dataset,
            dataloader,
            device,
            sigma_feature_device,
            run,
            metrics_logger,
            position_stats=position_stats,
        )

        metrics_csv = metrics_logger.path
        plots_dir = Path(experiment_cfg.logging.checkpoint_dir) / "plots"
        if metrics_csv.exists():
            df = pd.read_csv(metrics_csv)
            summary["metrics_csv"] = str(metrics_csv)
            if "feature_l1" in df.columns:
                best_idx = df["feature_l1"].idxmin()
            else:
                best_idx = df["g_loss"].idxmin()
            best_row = df.loc[best_idx].to_dict()
            summary["best_step"] = best_row
            plot_metric_trends(metrics_csv, plots_dir / "gan_losses.png", "step", ["d_loss", "g_loss"], "GAN Losses")
            plot_metric_trends(
                metrics_csv,
                plots_dir / "gan_features.png",
                "step",
                ["feature_l1", "feature_cov_diff", "diversity_xy"],
                "Feature / Diversity Metrics",
            )

        _save_checkpoints(generator, discriminator, experiment_cfg)
        summary_path = Path(experiment_cfg.logging.checkpoint_dir) / "gan_summary.json"
        write_summary_json(summary_path, summary)

        sample_dir = Path(experiment_cfg.logging.checkpoint_dir) / "samples"
        artifact_files = [
            Path(experiment_cfg.logging.checkpoint_dir)
            / f"generator_{experiment_cfg.experiment_name}.pt",
            Path(experiment_cfg.logging.checkpoint_dir)
            / f"discriminator_{experiment_cfg.experiment_name}.pt",
            summary_path,
        ]
        if summary.get("metrics_csv"):
            artifact_files.append(Path(summary["metrics_csv"]))
        detector_metrics_csv = Path(experiment_cfg.logging.checkpoint_dir) / "detector_metrics.csv"
        if detector_metrics_csv.exists():
            artifact_files.append(detector_metrics_csv)
        if "cross_dataset_csv" in summary:
            artifact_files.append(Path(summary["cross_dataset_csv"]))

        sigma_eval_dir = Path(experiment_cfg.logging.checkpoint_dir) / "sigma_eval"
        diffusion_eval_dir = Path(experiment_cfg.logging.checkpoint_dir) / experiment_cfg.training.diffusion_eval_log_dir
        artifact_dirs = [sample_dir, plots_dir]
        if sigma_eval_dir.exists():
            artifact_dirs.append(sigma_eval_dir)
        if diffusion_eval_dir.exists():
            artifact_dirs.append(diffusion_eval_dir)

        log_wandb_artifact(
            run,
            f"gan_run_{experiment_cfg.experiment_name}",
            "gan-run",
            artifact_files,
            artifact_dirs,
        )

        if experiment_cfg.logging.target == "wandb":
            tidy_checkpoint_artifacts(
                experiment_cfg.logging.checkpoint_dir,
                targets=("samples", "plots", "sigma_eval", experiment_cfg.training.diffusion_eval_log_dir),
            )


if __name__ == "__main__":
    main()
