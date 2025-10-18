"""Hydra-driven training entry point for GAN experiments."""
from __future__ import annotations

import logging
from pathlib import Path
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
from utils.plotting import plot_metric_trends
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


def _set_requires_grad(module: torch.nn.Module, flag: bool) -> None:
    for param in module.parameters():
        param.requires_grad = flag


def _jerk_magnitude(sequences: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    deltas = sequences[..., :2]
    dt = sequences[..., 2:3].clamp_min(eps)
    velocity = deltas / dt
    accel = (velocity[:, 1:, :] - velocity[:, :-1, :]) / dt[:, 1:, :]
    jerk = (accel[:, 1:, :] - accel[:, :-1, :]) / dt[:, 2:, :]
    return torch.linalg.norm(jerk, dim=-1)


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
) -> torch.Tensor:
    if dataset.config.feature_mode == "neuromotor":
        feats = torch.stack([compute_features_from_sequence(seq) for seq in sequences])
    elif dataset.config.feature_mode == "sigma_lognormal":
        feats = torch.stack([sigma_lognormal_features_from_sequence(seq) for seq in sequences])
    else:
        raise ValueError(f"Unsupported feature_mode: {dataset.config.feature_mode}")
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
    generator: ConditionalGenerator,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    run,
    *,
    conditioning_callback=None,
) -> None:
    epochs = experiment_cfg.training.reconstruction_epochs
    if epochs <= 0:
        return

    logger.info("Starting reconstruction warm-up for %d epoch(s)", epochs)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(
        generator.parameters(),
        lr=experiment_cfg.training.lr_generator,
        betas=(experiment_cfg.training.beta1, experiment_cfg.training.beta2),
    )

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        total_samples = 0
        generator.train()
        for batch in dataloader:
            real_sequences, features, _ = _to_device(batch, device)
            condition = _get_condition(features, experiment_cfg.model.generator.condition_dim)
            if conditioning_callback is not None:
                condition = conditioning_callback(real_sequences, condition)
            z = torch.zeros(real_sequences.size(0), experiment_cfg.model.generator.latent_dim, device=device)
            reconstructed = generator(z, condition)
            loss = criterion(reconstructed, real_sequences) * experiment_cfg.training.reconstruction_loss_weight

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * real_sequences.size(0)
            total_samples += real_sequences.size(0)

        avg_loss = total_loss / max(1, total_samples)
        logger.info("Warm-up epoch %d | recon_loss=%.6f", epoch, avg_loss)
        if run is not None:
            run.log({"warmup/epoch": epoch, "warmup/reconstruction_loss": avg_loss})


def _training_loop(
    experiment_cfg: GanExperimentConfig,
    generator: ConditionalGenerator,
    discriminator: GestureDiscriminator,
    dataset: GestureDataset,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    run,
    metrics_logger: Optional[CSVMetricLogger],
) -> Dict[str, Any]:
    _reconstruction_warmup(experiment_cfg, generator, dataloader, device, run)

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
    sampling_rate = experiment_cfg.data.sampling_rate
    dt_value = (1.0 / sampling_rate) if sampling_rate and sampling_rate > 0 else None

    for epoch in range(1, experiment_cfg.training.epochs + 1):
        for batch in dataloader:
            real_sequences, features, _ = _to_device(batch, device)
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
            fake_disc_sequences = generated_raw if use_absolute else _canonicalize_tensor_sequences(generated_raw, experiment_cfg.data)
            fake_scores, _ = discriminator(fake_disc_sequences, condition)
            if use_wgan:
                g_loss = -fake_scores.mean()
            else:
                assert bce_loss is not None
                target_labels = torch.ones_like(fake_scores)
                g_loss = bce_loss(fake_scores, target_labels)

            jerk_penalty_value = None
            if experiment_cfg.training.jerk_match_weight > 0:
                real_jerk = _jerk_magnitude(real_sequences)
                fake_jerk = _jerk_magnitude(generated_metrics)
                jerk_penalty = torch.abs(fake_jerk.mean() - real_jerk.mean())
                g_loss = g_loss + experiment_cfg.training.jerk_match_weight * jerk_penalty
                jerk_penalty_value = jerk_penalty.item()

            g_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()
            _set_requires_grad(discriminator, True)

            if use_absolute:
                deltas_xy = _positions_to_deltas_xy(generated_raw)
                generated_metrics = real_sequences.new_zeros(real_sequences.size(0), real_sequences.size(1), 3)
                generated_metrics[..., :2] = deltas_xy
                if dt_value is not None:
                    generated_metrics[..., 2] = dt_value
                else:
                    generated_metrics[..., 2] = real_sequences[..., 2]
                generated_metrics = _canonicalize_tensor_sequences(generated_metrics, experiment_cfg.data)
            else:
                generated_metrics = fake_disc_sequences

            log_metrics = step % experiment_cfg.training.log_interval == 0
            sample_output = step % experiment_cfg.training.sample_interval == 0

            metrics_payload = {
                "epoch": epoch,
                "step": step,
                "d_loss": d_loss.item(),
                "g_loss": g_loss.item(),
            }

            if jerk_penalty_value is not None:
                metrics_payload["jerk_penalty"] = jerk_penalty_value

            generated_cpu = generated_metrics.detach().cpu()
            denorm_generated = dataset.denormalize_sequences(generated_cpu)

            if log_metrics:
                with torch.no_grad():
                    fake_feature_batch = _feature_tensor_from_sequences(dataset, denorm_generated)
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
    generator: ConditionalGenerator,
    discriminator: GestureDiscriminator,
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
            "feature_l1",
            "feature_cov_diff",
            "diversity_xy",
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
