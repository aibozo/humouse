"""Hydra-driven training entry point for GAN experiments."""
from __future__ import annotations

import logging
from pathlib import Path
import shutil
from typing import Any, Dict, Optional, Tuple

import hydra
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf
from torch import nn, optim
from hydra.utils import to_absolute_path

from data.dataset import GestureDataset, GestureDatasetConfig
from models.discriminator import DiscriminatorConfig, GestureDiscriminator
from models.generator import ConditionalGenerator, GeneratorConfig
from models.gan_lstm import LSTMGenerator, LSTMDiscriminator
from eval.sigma_log_baseline import run_baseline
from train.config_schemas import (
    DataConfig,
    GanExperimentConfig,
    GanModelConfig,
    GanTrainingConfig,
)
from train.replay_buffer import ReplayBuffer
from train.train_detector import load_detector_config, run_detector_training
from utils.housekeeping import tidy_checkpoint_artifacts
from utils.logging import CSVMetricLogger, LoggingConfig, experiment_logger, write_summary_json, log_wandb_artifact
from utils.plotting import plot_metric_trends, generate_replay_vs_real_plots
from utils.eval import feature_distribution_metrics, sequence_diversity_metric
from features import (
    compute_features_from_sequence,
    sigma_lognormal_features_from_sequence,
)

logger = logging.getLogger(__name__)


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
    sampling_rate = experiment_cfg.data.sampling_rate
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


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():  # pragma: no cover
        torch.cuda.manual_seed_all(seed)


def _prepare_dataset_and_dataloader(
    experiment_cfg: GanExperimentConfig,
) -> Tuple[GestureDataset, torch.utils.data.DataLoader]:
    dataset_cfg = GestureDatasetConfig(
        dataset_id=experiment_cfg.data.dataset_id,
        sequence_length=experiment_cfg.data.sequence_length,
        max_gestures=experiment_cfg.data.max_gestures,
        sampling_rate=experiment_cfg.data.sampling_rate,
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

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=experiment_cfg.data.batch_size,
        shuffle=True,
        num_workers=experiment_cfg.data.num_workers,
        drop_last=True,
    )
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
) -> torch.Tensor:
    if dataset.config.feature_mode == "neuromotor":
        feats = torch.stack([compute_features_from_sequence(seq) for seq in sequences])
    elif dataset.config.feature_mode == "sigma_lognormal":
        feats = torch.stack([sigma_lognormal_features_from_sequence(seq) for seq in sequences])
    else:
        raise ValueError(f"Unsupported feature_mode: {dataset.config.feature_mode}")

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


def _run_sigma_evaluation(
    experiment_cfg: GanExperimentConfig,
    generator: ConditionalGenerator,
    dataset: GestureDataset,
    feature_pool: torch.Tensor,
    device: torch.device,
    epoch: int,
    run,
    metrics_logger: Optional[CSVMetricLogger],
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
    sampling_rate = experiment_cfg.data.sampling_rate
    dt_value = (1.0 / sampling_rate) if sampling_rate and sampling_rate > 0 else None

    generator_was_training = generator.training
    generator.eval()

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
                deltas_xy = _positions_to_deltas_xy(batch_output)
                seq = torch.zeros(current_batch, batch_output.size(1), 3, device=device, dtype=batch_output.dtype)
                seq[..., :2] = deltas_xy
                if dt_value is not None:
                    seq[..., 2] = dt_value
                else:
                    seq[..., 2] = 1.0
                batch_sequences = _canonicalize_tensor_sequences(seq, experiment_cfg.data)
            else:
                batch_sequences = _canonicalize_tensor_sequences(batch_output, experiment_cfg.data)
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
    run,
    metrics_logger: Optional[CSVMetricLogger],
) -> Dict[str, Any]:
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
    real_feature_cpu = dataset.get_positive_features_tensor()
    real_feature_tensor = real_feature_cpu.to(device)
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
    feature_pool = real_feature_tensor
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

    last_metrics: Dict[str, Any] = {}
    adversarial_type = experiment_cfg.training.adversarial_type.lower()
    use_wgan = adversarial_type == "wgan"
    bce_loss = nn.BCEWithLogitsLoss() if not use_wgan else None
    best_sigma_error = float("inf")
    best_sigma_epoch: Optional[int] = None

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
        )
        if warmup_summary is not None:
            sigma_eval_records.append(warmup_summary)
            warmup_error = warmup_summary.get("avg_error_rate")
            if warmup_error is not None and warmup_error + 1e-6 < best_sigma_error:
                best_sigma_error = warmup_error
                best_sigma_epoch = 0
                _save_checkpoints(generator, discriminator, experiment_cfg, suffix="_best")
                logger.info("Warm-up sigma error %.6f", warmup_error)

    use_absolute = experiment_cfg.training.absolute_coordinates
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
        for batch in dataloader:
            real_sequences, features, _ = _to_device(batch, device)
            if dt_value is not None and real_sequences.size(-1) >= 3:
                real_sequences = _apply_fixed_dt(real_sequences, dataset, dt_value)
            condition = _get_condition(features, experiment_cfg.model.generator.condition_dim)

            real_disc_sequences = _deltas_to_positions_xy(real_sequences) if use_absolute else real_sequences

            discriminator.train()
            generator.eval()
            _set_requires_grad(discriminator, True)
            _set_requires_grad(generator, False)

            for _ in range(experiment_cfg.training.discriminator_steps):
                z = torch.randn(real_sequences.size(0), experiment_cfg.model.generator.latent_dim, device=device)
                fake_raw = generator(z, condition).detach()
                fake_disc_sequences = fake_raw if use_absolute else _canonicalize_tensor_sequences(fake_raw, experiment_cfg.data)

                real_scores, _ = discriminator(real_disc_sequences, condition)
                fake_scores, _ = discriminator(fake_disc_sequences, condition)
                if use_wgan:
                    gp = _compute_gradient_penalty(
                        discriminator,
                        real_disc_sequences,
                        fake_disc_sequences,
                        condition,
                        device,
                    )
                    d_loss = fake_scores.mean() - real_scores.mean() + experiment_cfg.training.gradient_penalty_weight * gp
                else:
                    assert bce_loss is not None
                    real_labels = torch.ones_like(real_scores)
                    fake_labels = torch.zeros_like(fake_scores)
                    real_loss = bce_loss(real_scores, real_labels)
                    fake_loss = bce_loss(fake_scores, fake_labels)
                    d_loss = 0.5 * (real_loss + fake_loss)

                d_optimizer.zero_grad()
                d_loss.backward()
                d_optimizer.step()

            generator.train()
            _set_requires_grad(discriminator, False)
            _set_requires_grad(generator, True)
            z = torch.randn(real_sequences.size(0), experiment_cfg.model.generator.latent_dim, device=device)
            generated_raw = generator(z, condition)

            if use_absolute:
                deltas_xy = _positions_to_deltas_xy(generated_raw)
                if real_sequences.size(-1) >= 3:
                    dt_col = real_sequences[..., 2:3].detach()
                else:
                    dt_fill = dt_value_norm if dt_value_norm is not None else (dt_value if dt_value is not None else 1.0)
                    dt_col = torch.full_like(deltas_xy[..., :1], float(dt_fill))
                generated_sequence = torch.cat([deltas_xy, dt_col], dim=-1)
                disc_fake_input = generated_raw
            else:
                if generated_raw.size(-1) >= 3:
                    generated_sequence = generated_raw
                else:
                    if real_sequences.size(-1) >= 3:
                        dt_col = real_sequences[..., 2:3].detach()
                    else:
                        dt_fill = dt_value_norm if dt_value_norm is not None else (dt_value if dt_value is not None else 1.0)
                        dt_col = torch.full_like(generated_raw[..., :1], float(dt_fill))
                    generated_sequence = torch.cat([generated_raw, dt_col], dim=-1)
                disc_fake_input = _canonicalize_tensor_sequences(generated_sequence, experiment_cfg.data)

            if dt_value_norm is not None and generated_sequence.size(-1) >= 3:
                generated_sequence = generated_sequence.clone()
                generated_sequence[..., 2] = float(dt_value_norm)
                if not use_absolute:
                    disc_fake_input = _canonicalize_tensor_sequences(generated_sequence, experiment_cfg.data)

            fake_scores, _ = discriminator(disc_fake_input, condition)
            if use_wgan:
                g_loss = -fake_scores.mean()
            else:
                assert bce_loss is not None
                target_labels = torch.ones_like(fake_scores)
                g_loss = bce_loss(fake_scores, target_labels)

            curvature_penalty_value = None
            lateral_penalty_value = None
            direction_penalty_value = None

            need_denorm = (
                experiment_cfg.training.curvature_match_weight > 0
                or experiment_cfg.training.lateral_match_weight > 0
                or experiment_cfg.training.direction_match_weight > 0
            )
            if need_denorm:
                real_denorm = dataset.denormalize_sequences(real_sequences)
                fake_denorm = dataset.denormalize_sequences(generated_sequence)
            else:
                real_denorm = fake_denorm = None

            if need_denorm:
                real_xy = real_denorm[..., :2]
                fake_xy = fake_denorm[..., :2]

            if (
                experiment_cfg.training.curvature_match_weight > 0
                or experiment_cfg.training.lateral_match_weight > 0
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

            if experiment_cfg.training.direction_match_weight > 0:
                real_variation = _delta_variation(real_xy)
                fake_variation = _delta_variation(fake_xy)
                variation_diff = torch.abs(fake_variation.mean(dim=0) - real_variation.mean(dim=0)).sum()
                g_loss = g_loss + experiment_cfg.training.direction_match_weight * variation_diff
                direction_penalty_value = float(variation_diff.detach().item())

            g_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()
            _set_requires_grad(discriminator, True)

            generated_sequence_detached = generated_sequence.detach()
            generated_metrics = _canonicalize_tensor_sequences(generated_sequence, experiment_cfg.data)

            log_metrics = step % experiment_cfg.training.log_interval == 0
            sample_output = step % experiment_cfg.training.sample_interval == 0

            metrics_payload = {
                "epoch": epoch,
                "step": step,
                "d_loss": d_loss.item(),
                "g_loss": g_loss.item(),
            }

            if curvature_penalty_value is not None:
                metrics_payload["curvature_penalty"] = curvature_penalty_value
            if lateral_penalty_value is not None:
                metrics_payload["lateral_penalty"] = lateral_penalty_value
            if direction_penalty_value is not None:
                metrics_payload["direction_penalty"] = direction_penalty_value

            generated_cpu = generated_metrics.detach().cpu()
            denorm_generated = dataset.denormalize_sequences(generated_cpu)

            if log_metrics:
                real_phys = dataset.denormalize_sequences(real_sequences.detach().cpu())
                fake_phys = dataset.denormalize_sequences(generated_sequence_detached.cpu())

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

            if sample_output:
                sample_path = _save_generated_samples(denorm_generated[:16], sample_dir, epoch, step)
                metrics_payload["sample_path"] = str(sample_path)

            if metrics_logger is not None and (log_metrics or sample_output):
                metrics_logger.log(metrics_payload)
            last_metrics = metrics_payload
            step += 1

        # optional epoch summary log
        if run is not None:
            run.log({"epoch": epoch, "d_loss": d_loss.item(), "g_loss": g_loss.item()})

        sigma_summary = _run_sigma_evaluation(
            experiment_cfg,
            generator,
            dataset,
            feature_pool,
            device,
            epoch,
            run,
            sigma_eval_logger,
        )
        if sigma_summary is not None:
            sigma_eval_records.append(sigma_summary)
            replay_dir_str = sigma_summary.get("replay_dir")
            if (
                real_sequences_plot is not None
                and real_sequences_plot.shape[0] > 0
                and real_features_plot.size > 0
                and replay_dir_str
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
                feature_vectors = _feature_tensor_from_sequences(dataset, denorm_batch, normalize=False)
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
    _set_seed(experiment_cfg.seed)

    dataset, dataloader = _prepare_dataset_and_dataloader(experiment_cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if experiment_cfg.model.architecture == "tcn":
        generator = ConditionalGenerator(experiment_cfg.model.generator).to(device)
        discriminator = GestureDiscriminator(experiment_cfg.model.discriminator).to(device)
    elif experiment_cfg.model.architecture == "lstm":
        generator = LSTMGenerator(experiment_cfg.model.generator).to(device)
        discriminator = LSTMDiscriminator(experiment_cfg.model.discriminator).to(device)
    else:
        raise ValueError(f"Unsupported model architecture: {experiment_cfg.model.architecture}")

    init_path = experiment_cfg.training.generator_init_path
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
            "sample_path",
        ],
    )

    with experiment_logger(experiment_cfg.logging, cfg_dict) as run:
        summary = _training_loop(
            experiment_cfg,
            generator,
            discriminator,
            dataset,
            dataloader,
            device,
            run,
            metrics_logger,
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
        artifact_dirs = [sample_dir, plots_dir]
        if sigma_eval_dir.exists():
            artifact_dirs.append(sigma_eval_dir)

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
                targets=("samples", "plots", "sigma_eval"),
            )


if __name__ == "__main__":
    main()
