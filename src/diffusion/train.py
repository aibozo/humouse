"""Hydra entry point for training diffusion models on mouse trajectories."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict, fields
from pathlib import Path
from typing import Optional

import hydra
import torch
import torch.nn.functional as F
from hydra.core.config_store import ConfigStore
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.nn.utils import clip_grad_norm_
import numpy as np

from diffusion.data import DiffusionDataConfig, create_dataloader
from diffusion.models import UNet1D, UNet1DConfig
from diffusion.noise import DiffusionScheduleConfig, build_schedule, compute_v, q_sample, x0_from_eps, x0_from_v
from diffusion.utils import EMAModel, masked_mse, match_time_channel
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

    def __post_init__(self) -> None:
        if not isinstance(self.betas, tuple):
            self.betas = tuple(self.betas)


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


def _masked_channel_std(tensor: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    valid = mask.to(dtype=tensor.dtype).unsqueeze(-1)
    count = valid.sum(dim=(0, 1)).clamp_min(1.0)
    mean = (tensor * valid).sum(dim=(0, 1)) / count
    centered = tensor - mean.view(1, 1, -1)
    var = ((centered**2) * valid).sum(dim=(0, 1)) / count
    return torch.sqrt(var + 1e-6)


def _load_experiment_config(cfg: DictConfig) -> DiffusionExperimentConfig:
    container = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(container, dict):
        raise TypeError("Expected dict container for diffusion config")
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
        loss = masked_mse(preds, target, mask)
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
) -> dict[str, float]:
    rng = np.random.default_rng(seed)

    real_sequences = _sample_positive_sequences(real_dataset, samples, rng=rng)
    if len(real_sequences) < 40:
        return {}
    num_samples = min(samples, len(real_sequences))

    real_batch = torch.stack(real_sequences[:num_samples])
    real_denorm = real_dataset.denormalize_sequences(real_batch)
    real_features: list[torch.Tensor] = []
    for seq in real_denorm:
        feat = compute_features_from_sequence(seq)
        feat = dataset.normalize_features(feat)
        real_features.append(feat)
    if not real_features:
        return {}
    real_np = torch.stack(real_features).numpy()

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
    fake_denorm = match_time_channel(fake_denorm, real_denorm)
    fake_features: list[torch.Tensor] = []
    for seq in fake_denorm:
        feat = compute_features_from_sequence(seq)
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
    fake = match_time_channel(fake, real_batch)
    fake_norm_std = fake.std(dim=(0, 1))
    fake_denorm = dataset.denormalize_sequences(fake)
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
    scaler: GradScaler,
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
    scaler = GradScaler(enabled=training_cfg.amp and device.type == "cuda")
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

    global_step = 0
    for epoch in range(1, training_cfg.epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_weight = 0.0
        for batch_idx, batch in enumerate(train_loader, start=1):
            sequences = _prepare_sequences(batch["sequences"].to(device), target_channels=model_cfg.in_channels)
            mask = batch["mask"].to(device)
            cond = _prepare_features(batch["features"].to(device), model_cfg.cond_dim)

            sequences = apply_default_augmentations(
                sequences,
                mask=mask,
                time_stretch=float(data_cfg.time_stretch),
                jitter_std=float(data_cfg.jitter_std),
                mirror_prob=float(data_cfg.mirror_prob),
            )

            timesteps = torch.randint(0, schedule.timesteps, (sequences.size(0),), device=device)
            noise = torch.randn_like(sequences)

            with autocast(enabled=scaler.is_enabled()):
                xt, noise = q_sample(schedule, sequences, timesteps, noise=noise)
                alpha, sigma = schedule.coefficients(timesteps, device=device)

                preds = model(xt.permute(0, 2, 1), timesteps, cond=cond, mask=mask)
                preds = preds.permute(0, 2, 1)

                obj = training_cfg.objective.lower()
                if obj == "epsilon":
                    target = noise
                    x0_hat = x0_from_eps(xt, preds, alpha, sigma)
                else:
                    target = compute_v(sequences, noise, alpha, sigma)
                    x0_hat = x0_from_v(xt, preds, alpha, sigma)

                weights = None
                if training_cfg.min_snr_gamma not in (None, 0):
                    log_snr = schedule.log_snr_at(timesteps, device=device)
                    weights = _min_snr_weight(log_snr, training_cfg.min_snr_gamma)
                loss = masked_mse(preds, target, mask, weights=weights)
                if training_cfg.scale_reg_weight > 0:
                    std_hat = _masked_channel_std(x0_hat, mask)
                    target_std = scale_target_std.to(device=std_hat.device, dtype=std_hat.dtype)
                    scale_penalty = torch.mean((std_hat - target_std) ** 2)
                    loss = loss + training_cfg.scale_reg_weight * scale_penalty
                if global_step % training_cfg.log_interval == 0:
                    print(
                        f"debug step={global_step} mask_sum={mask.sum().item():.0f} target_std={target.std().item():.4f}"
                    )

            scaler.scale(loss).backward()
            if training_cfg.grad_clip > 0:
                scaler.unscale_(optimizer)
                clip_grad_norm_(model.parameters(), training_cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            ema.update(model)

            mask_weight = float(mask.sum().item() * sequences.shape[-1])
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
            )
            last_val_loss = val_loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
            logger.info("Epoch %d validation loss (EMA): %.6f", epoch, val_loss)
            classifier_metrics = {}
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
        if training_cfg.sample_eval_count > 0:
            sample_stats = _compute_sample_stats(
                ema_sampler,
                train_loader.dataset,
                sample_count=training_cfg.sample_eval_count,
                seq_len=data_cfg.sequence_length,
                steps=min(training_cfg.sample_eval_steps, training_cfg.classifier_steps),
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
