"""Hydra entry for training timing models."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Optional

import hydra
import torch
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf
from torch.optim import AdamW
from torch.utils.data import DataLoader

from timing.data import TimingDataset, collate_timing
from timing.losses import dirichlet_nll, lognormal_nll, progress_cdf_loss
from timing.model import TimingModel, TimingModelConfig

logger = logging.getLogger(__name__)


@dataclass
class TimingTrainingConfig:
    epochs: int = 20
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 512
    num_workers: int = 0
    pin_memory: bool = True
    dirichlet_weight: float = 0.5
    resume_from: Optional[str] = None
    checkpoint_path: Optional[str] = None
    sigma_prior_weight: float = 0.1
    log_sigma_range: float = 0.5
    profile_smoothing: float = 1e-3
    progress_weight: float = 0.1
    balance_time_steps: bool = True
    template_target_weight: float = 0.2


@dataclass
class TimingExperimentConfig:
    dataset_id: str = "balabit"
    cache_dir: str = "datasets"
    sequence_length: int = 64
    model: TimingModelConfig = field(default_factory=TimingModelConfig)
    training: TimingTrainingConfig = field(default_factory=TimingTrainingConfig)

    def __post_init__(self) -> None:
        if not isinstance(self.model, TimingModelConfig):
            self.model = TimingModelConfig(**self.model)
        if not isinstance(self.training, TimingTrainingConfig):
            self.training = TimingTrainingConfig(**self.training)


cs = ConfigStore.instance()
cs.store(name="timing_base", node=TimingExperimentConfig)


def _filter_kwargs(datatype, data: dict) -> dict:
    valid = {f.name for f in fields(datatype)}
    return {k: v for k, v in data.items() if k in valid}


def _loader(cfg: TimingExperimentConfig, split: str) -> DataLoader:
    cache_path = Path(cfg.cache_dir) / cfg.dataset_id / f"{split}_timing.pt"
    dataset = TimingDataset(cache_path)
    batch_size = cfg.training.batch_size
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=cfg.training.num_workers,
        pin_memory=cfg.training.pin_memory,
        drop_last=(split == "train"),
        collate_fn=collate_timing,
    )


@hydra.main(version_base=None, config_path="../../conf", config_name="timing/base")
def main(cfg: DictConfig) -> None:
    OmegaConf.set_struct(cfg, False)
    container = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(container, dict):
        raise TypeError("Expected dict config for timing experiment")
    if "timing" in container and isinstance(container["timing"], dict):
        container = container["timing"]
    exp_cfg = TimingExperimentConfig(**_filter_kwargs(TimingExperimentConfig, container))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Training timing model on %s", device)

    train_loader = _loader(exp_cfg, "train")
    val_loader = _loader(exp_cfg, "val")

    model_cfg = exp_cfg.model
    model_cfg.feature_dim = train_loader.dataset.features.shape[1]
    model_cfg.sequence_length = train_loader.dataset.profiles.size(1)
    model = TimingModel(model_cfg).to(device)
    optimizer = AdamW(model.parameters(), lr=exp_cfg.training.lr, weight_decay=exp_cfg.training.weight_decay)

    real_stats = train_loader.dataset.stats
    log_duration_std = float(real_stats.get("duration_log_std", 0.6))
    model_cfg.log_sigma_base = log_duration_std
    model_cfg.log_sigma_range = exp_cfg.training.log_sigma_range
    sigma_prior_center = torch.tensor(log_duration_std, device=device, dtype=torch.float32)

    time_weights: torch.Tensor | None = None
    if exp_cfg.training.balance_time_steps:
        counts = train_loader.dataset.masks.float().sum(dim=0)
        counts = torch.where(counts > 0, counts, torch.ones_like(counts))
        time_weights = (counts.max() / counts).to(device=device, dtype=torch.float32)

    stats_profile_key = "profile_mean_masked" if "profile_mean_masked" in real_stats else "profile_mean"
    real_profile_mean = real_stats.get(stats_profile_key)
    if real_profile_mean is not None:
        real_profile_mean = torch.tensor(real_profile_mean, device=device, dtype=torch.float32)

    def _sequence_weights(mask_tensor: torch.Tensor) -> torch.Tensor:
        mask_bool = mask_tensor.bool()
        weights: list[torch.Tensor] = []
        for seq_mask in mask_bool:
            idx = seq_mask.nonzero(as_tuple=True)[0]
            if idx.numel() < 2:
                weights.append(torch.tensor(0.0, device=mask_tensor.device))
            elif time_weights is None:
                weights.append(torch.tensor(1.0, device=mask_tensor.device))
            else:
                weights.append(time_weights[idx].mean())
        if not weights:
            return torch.zeros(0, device=mask_tensor.device)
        return torch.stack(weights).to(dtype=torch.float32)

    def _log_stats(tag: str, outputs: dict, mask: torch.Tensor) -> None:
        duration = outputs["mu_log_duration"].exp()
        template = outputs["template"]
        mask_float = mask.to(device=template.device, dtype=template.dtype)
        counts = mask_float.sum(dim=0).clamp_min(1e-6)
        template_sum = (template * mask_float).sum(dim=0)
        template_mean = template_sum / counts
        conc_mean = float(outputs["concentration"].mean().item())
        msg = (
            f"{tag} duration μ={float(duration.mean().item()):.3f} σ={float(duration.std().item()):.3f} | "
            f"conc μ={conc_mean:.3f} | template first/mid/last = "
            f"{float(template_mean[0].item()):.4f} / "
            f"{float(template_mean[len(template_mean) // 2].item()):.4f} / "
            f"{float(template_mean[-1].item()):.4f}"
        )
        if real_profile_mean is not None:
            msg += (
                f" | real first/mid/last = "
                f"{float(real_profile_mean[0].item()):.4f} / "
                f"{float(real_profile_mean[len(real_profile_mean) // 2].item()):.4f} / "
                f"{float(real_profile_mean[-1].item()):.4f}"
            )
        logger.info(msg)

    def _log_duration_diagnostics(tag: str, outputs: dict) -> None:
        log_sigma = outputs["log_sigma_log_duration"].detach()
        mu_log = outputs["mu_log_duration"].detach()
        device = log_sigma.device
        quantiles = torch.tensor([0.5, 0.9, 0.99], device=device)
        log_sigma_q = torch.quantile(log_sigma, quantiles).cpu().tolist()
        mu_log_q = torch.quantile(mu_log, quantiles).cpu().tolist()
        sigma = log_sigma.exp()
        base = model_cfg.log_sigma_base
        rng = model_cfg.log_sigma_range
        upper = base + rng
        lower = base - rng
        logger.info(
            "%s logσ q50/q90/q99 = %.3f / %.3f / %.3f | μ_log q50/q90/q99 = %.3f / %.3f / %.3f | logσ range=[%.3f, %.3f] | σ_max=%.3f",
            tag,
            *log_sigma_q,
            *mu_log_q,
            lower,
            upper,
            float(sigma.max().item()),
        )

    for epoch in range(1, exp_cfg.training.epochs + 1):
        model.train()
        total_loss = 0.0
        steps = 0
        for batch in train_loader:
            features = batch["features"].to(device)
            log_duration = batch["log_duration"].to(device)
            profile = batch["profile"].to(device)
            mask = batch["mask"].to(device)
            mask_bool = mask.bool()
            mask_float = mask.to(dtype=features.dtype)

            outputs = model(features, mask=mask_bool)
            template = outputs["template"]
            log_sigma = outputs["log_sigma_log_duration"]
            duration_loss = lognormal_nll(log_duration, outputs["mu_log_duration"], log_sigma).mean()

            target_profile = profile * mask_float
            target_profile = target_profile / target_profile.sum(dim=1, keepdim=True).clamp_min(1e-6)
            target_profile = target_profile * mask_float

            seq_weights = _sequence_weights(mask)
            if seq_weights.numel() == 0:
                seq_weights = torch.ones(1, device=device)
            profile_terms = dirichlet_nll(
                target_profile,
                template,
                outputs["concentration"],
                mask_bool,
                smoothing=exp_cfg.training.profile_smoothing,
                reduction="none",
            )
            denom = seq_weights.sum().clamp_min(1.0)
            profile_loss = (profile_terms * seq_weights).sum() / denom

            progress_terms = progress_cdf_loss(
                template,
                target_profile,
                mask_bool,
                reduction="none",
            )
            progress_loss = (progress_terms * seq_weights).sum() / seq_weights.sum().clamp_min(1.0)

            template_target_loss = torch.tensor(0.0, device=device)
            if real_profile_mean is not None and exp_cfg.training.template_target_weight > 0.0:
                target_global = real_profile_mean.unsqueeze(0).expand_as(template)
                target_global = target_global * mask_float
                target_global = target_global / target_global.sum(dim=1, keepdim=True).clamp_min(1e-6)
                template_masked = template * mask_float
                template_masked = template_masked / template_masked.sum(dim=1, keepdim=True).clamp_min(1e-6)
                target_diff = ((template_masked - target_global) ** 2 * mask_float).sum(dim=1)
                template_target_loss = (target_diff * seq_weights).sum() / seq_weights.sum().clamp_min(1.0)

            sigma_reg = ((log_sigma - sigma_prior_center) ** 2).mean()

            loss = duration_loss + exp_cfg.training.dirichlet_weight * profile_loss + exp_cfg.training.sigma_prior_weight * sigma_reg
            if exp_cfg.training.progress_weight > 0.0:
                loss = loss + exp_cfg.training.progress_weight * progress_loss
            if exp_cfg.training.template_target_weight > 0.0:
                loss = loss + exp_cfg.training.template_target_weight * template_target_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            steps += 1

        logger.info("Epoch %d timing loss %.6f", epoch, total_loss / max(steps, 1))

        with torch.no_grad():
            model.eval()
            val_losses = []
            for batch in val_loader:
                features = batch["features"].to(device)
                log_duration = batch["log_duration"].to(device)
                profile = batch["profile"].to(device)
                mask = batch["mask"].to(device)
                mask_bool = mask.bool()
                mask_float = mask.to(dtype=features.dtype)
                outputs = model(features, mask=mask_bool)
                template = outputs["template"]
                duration_loss = lognormal_nll(
                    log_duration,
                    outputs["mu_log_duration"],
                    outputs["log_sigma_log_duration"],
                ).mean()

                target_profile = profile * mask_float
                target_profile = target_profile / target_profile.sum(dim=1, keepdim=True).clamp_min(1e-6)
                target_profile = target_profile * mask_float
                seq_weights = _sequence_weights(mask)
                if seq_weights.numel() == 0:
                    seq_weights = torch.ones(1, device=device)
                profile_terms = dirichlet_nll(
                    target_profile,
                    template,
                    outputs["concentration"],
                    mask_bool,
                    smoothing=exp_cfg.training.profile_smoothing,
                    reduction="none",
                )
                profile_loss = (profile_terms * seq_weights).sum() / seq_weights.sum().clamp_min(1.0)

                progress_terms = progress_cdf_loss(
                    template,
                    target_profile,
                    mask_bool,
                    reduction="none",
                )
                progress_loss = (progress_terms * seq_weights).sum() / seq_weights.sum().clamp_min(1.0)

                sigma_reg_val = ((outputs["log_sigma_log_duration"] - sigma_prior_center) ** 2).mean()
                template_target_loss = torch.tensor(0.0, device=device)
                if real_profile_mean is not None and exp_cfg.training.template_target_weight > 0.0:
                    target_global = real_profile_mean.unsqueeze(0).expand_as(template)
                    target_global = target_global * mask_float
                    target_global = target_global / target_global.sum(dim=1, keepdim=True).clamp_min(1e-6)
                    template_masked = template * mask_float
                    template_masked = template_masked / template_masked.sum(dim=1, keepdim=True).clamp_min(1e-6)
                    target_diff = ((template_masked - target_global) ** 2 * mask_float).sum(dim=1)
                    template_target_loss = (target_diff * seq_weights).sum() / seq_weights.sum().clamp_min(1.0)

                total = (
                    duration_loss
                    + exp_cfg.training.dirichlet_weight * profile_loss
                    + exp_cfg.training.sigma_prior_weight * sigma_reg_val
                )
                if exp_cfg.training.progress_weight > 0.0:
                    total = total + exp_cfg.training.progress_weight * progress_loss
                if exp_cfg.training.template_target_weight > 0.0:
                    total = total + exp_cfg.training.template_target_weight * template_target_loss
                val_losses.append(total.item())
            logger.info("Epoch %d validation loss %.6f", epoch, sum(val_losses) / max(len(val_losses), 1))
            example_batch = next(iter(val_loader))
            example_mask = example_batch["mask"].to(device)
            example_outputs = model(example_batch["features"].to(device), mask=example_mask.bool())
            _log_stats("Val", example_outputs, example_mask)
            _log_duration_diagnostics("Val", example_outputs)

    if exp_cfg.training.checkpoint_path:
        path = Path(exp_cfg.training.checkpoint_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": model.state_dict(), "config": model_cfg}, path)
        logger.info("Saved timing checkpoint to %s", path)


if __name__ == "__main__":
    main()
