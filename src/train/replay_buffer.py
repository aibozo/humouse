"""Replay buffer for generated gesture sequences."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import torch


@dataclass
class ReplayBuffer:
    path: Path
    max_size: int = 5000

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        self.sequences: list[torch.Tensor] = []
        self.features: list[torch.Tensor] = []
        if self.path.exists():
            self._load()

    def __len__(self) -> int:
        return len(self.sequences)

    @property
    def device(self) -> torch.device:
        return self.sequences[0].device if self.sequences else torch.device("cpu")

    def add(self, sequences: torch.Tensor, features: torch.Tensor) -> None:
        if sequences.ndim != 3:
            raise ValueError("Expected sequences tensor of shape (N, L, 3)")
        if features.ndim != 2:
            raise ValueError("Expected features tensor of shape (N, F)")
        if sequences.size(0) != features.size(0):
            raise ValueError("Sequences/features batch size mismatch")

        for seq, feat in zip(sequences, features):
            self.sequences.append(seq.detach().cpu())
            self.features.append(feat.detach().cpu())

        overflow = len(self.sequences) - self.max_size
        if overflow > 0:
            self.sequences = self.sequences[overflow:]
            self.features = self.features[overflow:]

    def sample(self, count: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if len(self.sequences) == 0:
            raise ValueError("Replay buffer is empty")
        count = min(count, len(self.sequences))
        indices = torch.randperm(len(self.sequences))[:count]
        seqs = torch.stack([self.sequences[i] for i in indices])
        feats = torch.stack([self.features[i] for i in indices])
        return seqs, feats

    def save(self) -> None:
        if not self.sequences:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "sequences": torch.stack(self.sequences),
                "features": torch.stack(self.features),
            },
            self.path,
        )

    def _load(self) -> None:
        data = torch.load(self.path)
        sequences: torch.Tensor = data["sequences"]
        features: torch.Tensor = data["features"]
        self.sequences = [seq.cpu() for seq in sequences]
        self.features = [feat.cpu() for feat in features]

    @classmethod
    def load(cls, path: str | Path, max_size: int = 5000) -> "ReplayBuffer":
        buffer = cls(path=Path(path), max_size=max_size)
        return buffer

