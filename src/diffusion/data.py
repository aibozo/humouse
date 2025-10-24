"""Data utilities and loaders for diffusion training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch.utils.data import DataLoader

from data.dataset import GestureDataset, GestureDatasetConfig
from diffusion.utils import infer_mask_from_deltas


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

    def __post_init__(self) -> None:
        if isinstance(self.feature_mode, str):
            self.feature_mode = self.feature_mode.lower()


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
