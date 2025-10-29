"""Datasets for training timing models from cached durations and profiles."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset


@dataclass
class TimingDataConfig:
    cache_dir: str = "datasets"
    dataset_id: str = "balabit"
    split: str = "train"
    batch_size: int = 256
    num_workers: int = 0
    pin_memory: bool = True

    @property
    def cache_path(self) -> Path:
        return Path(self.cache_dir) / self.dataset_id / f"{self.split}_timing.pt"


class TimingDataset(Dataset):
    """Loads timing cache tensors saved by scripts/build_timing_cache.py."""

    def __init__(self, cache_path: str | Path):
        payload = torch.load(Path(cache_path), map_location="cpu")
        self.durations = payload["durations"].float()
        self.profiles = payload["profiles"].float()
        self.masks = payload["masks"].bool()
        features = payload.get("features")
        if features is None or features.numel() == 0:
            self.features = torch.zeros(self.durations.size(0), 1)
        else:
            self.features = features.float()
        self.stats = payload.get("stats", {})
        self.config = payload.get("config", {})
        self.split = payload.get("split")

    def __len__(self) -> int:  # noqa: D401
        return self.durations.size(0)

    def __getitem__(self, idx: int) -> dict:
        return {
            "duration": self.durations[idx],
            "log_duration": torch.log(self.durations[idx].clamp_min(1e-6)),
            "profile": self.profiles[idx],
            "mask": self.masks[idx],
            "features": self.features[idx],
        }


def collate_timing(batch: list[dict]) -> dict:
    durations = torch.stack([b["duration"] for b in batch])
    log_durations = torch.stack([b["log_duration"] for b in batch])
    profiles = torch.stack([b["profile"] for b in batch])
    masks = torch.stack([b["mask"] for b in batch])
    features = torch.stack([b["features"] for b in batch])
    return {
        "duration": durations,
        "log_duration": log_durations,
        "profile": profiles,
        "mask": masks,
        "features": features,
    }

