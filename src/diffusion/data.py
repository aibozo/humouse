"""Data utilities and loaders for diffusion training."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import torch
from torch.utils.data import DataLoader

from data.dataset import GestureDataset, GestureDatasetConfig
from diffusion.utils import infer_mask_from_deltas

logger = logging.getLogger(__name__)


@dataclass
class DiffusionDataConfig:
    """Dataset and dataloader settings for diffusion training."""

    dataset_id: str = "balabit"
    sequence_length: int = 128
    batch_size: int = 64
    num_workers: int = 8
    prefetch_factor: int = 2
    pin_memory: bool = True
    train_split: str = "train"
    val_split: Optional[str] = "val"
    max_train_gestures: Optional[int] = None
    max_val_gestures: Optional[int] = 512
    normalize_sequences: bool = True
    normalize_features: bool = True
    canonicalize_path: bool = True
    canonicalize_duration: bool = True
    include_goal_geometry: bool = False
    feature_mode: str = "neuromotor"
    sampling_rate: Optional[float | str] = None
    time_stretch: float = 0.0
    jitter_std: float = 0.0
    mirror_prob: float = 0.0
    max_channel_std: Optional[float] = None
    max_delta_abs: Optional[float] = None
    outlier_log_limit: int = 10

    def __post_init__(self) -> None:
        if isinstance(self.feature_mode, str):
            self.feature_mode = self.feature_mode.lower()
        if self.max_channel_std is not None and self.max_channel_std <= 0:
            raise ValueError("max_channel_std must be positive when set.")
        if self.max_delta_abs is not None and self.max_delta_abs <= 0:
            raise ValueError("max_delta_abs must be positive when set.")
        self.outlier_log_limit = max(0, int(self.outlier_log_limit))


def _filter_dataset_outliers(
    dataset: GestureDataset,
    *,
    max_channel_std: Optional[float],
    max_delta_abs: Optional[float],
    log_limit: int,
) -> int:
    if max_channel_std is None and max_delta_abs is None:
        return 0

    kept: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    removed: list[tuple[int, float, float]] = []
    for idx, (seq, feats, label) in enumerate(dataset.samples):
        mask = (seq.abs().sum(dim=-1) > 0)
        if mask.any():
            valid = seq[mask]
            xy = valid[:, :2]
            max_std = float(xy.std(dim=0, unbiased=False).max())
            max_abs = float(xy.abs().max())
        else:
            max_std = 0.0
            max_abs = 0.0

        drop = False
        if max_channel_std is not None and max_std > max_channel_std:
            drop = True
        if max_delta_abs is not None and max_abs > max_delta_abs:
            drop = True

        if drop:
            removed.append((idx, max_std, max_abs))
            continue
        kept.append((seq, feats, label))

    if removed:
        dataset.samples = kept
        msg_details = ", ".join(
            f"[idx={idx} std={std:.3f} abs={abs_val:.3f}]" for idx, std, abs_val in removed[:log_limit]
        )
        if log_limit and len(removed) > log_limit:
            msg_details += f", ... (+{len(removed) - log_limit} more)"
        logger.warning(
            "Filtered %d diffusion samples exceeding thresholds (max_std=%s, max_abs=%s): %s",
            len(removed),
            max_channel_std,
            max_delta_abs,
            msg_details if msg_details else "details suppressed",
        )
    return len(removed)


def build_diffusion_dataset(
    cfg: DiffusionDataConfig,
    *,
    split: str,
    max_gestures: Optional[int],
) -> GestureDataset:
    """Instantiate a GestureDataset configured for diffusion training."""
    dataset_cfg = GestureDatasetConfig(
        dataset_id=cfg.dataset_id,
        sequence_length=cfg.sequence_length,
        max_gestures=max_gestures,
        sampling_rate=cfg.sampling_rate,
        split=split,
        use_generated_negatives=False,
        normalize_sequences=cfg.normalize_sequences,
        normalize_features=cfg.normalize_features,
        feature_mode=cfg.feature_mode,
        canonicalize_path=cfg.canonicalize_path,
        canonicalize_duration=cfg.canonicalize_duration,
        include_goal_geometry=cfg.include_goal_geometry,
        cache_enabled=True,
    )
    return GestureDataset(dataset_cfg)


def diffusion_collate_fn(batch: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Collate gesture sequences into a batch with masks and conditioning features."""
    sequences, features, labels = zip(*batch)
    del labels  # diffusion uses only real gestures
    seq_tensor = torch.stack(sequences, dim=0)  # [B, T, C]
    feat_tensor = torch.stack(features, dim=0)  # [B, F]
    mask = infer_mask_from_deltas(seq_tensor)
    return {
        "sequences": seq_tensor,
        "features": feat_tensor,
        "mask": mask,
    }


def create_dataloader(
    cfg: DiffusionDataConfig,
    *,
    split: str,
    max_gestures: Optional[int],
    shuffle: bool,
) -> DataLoader:
    dataset = build_diffusion_dataset(cfg, split=split, max_gestures=max_gestures)
    removed = _filter_dataset_outliers(
        dataset,
        max_channel_std=cfg.max_channel_std,
        max_delta_abs=cfg.max_delta_abs,
        log_limit=cfg.outlier_log_limit,
    )
    if removed:
        logger.info("Dataset size after outlier filtering: %d samples", len(dataset))
    loader_kwargs = {
        "batch_size": cfg.batch_size,
        "shuffle": shuffle,
        "num_workers": cfg.num_workers,
        "pin_memory": cfg.pin_memory,
        "drop_last": shuffle,
        "collate_fn": diffusion_collate_fn,
    }
    if cfg.num_workers > 0:
        loader_kwargs["prefetch_factor"] = cfg.prefetch_factor
    return DataLoader(dataset, **loader_kwargs)
