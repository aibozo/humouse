from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from data.dataset import GestureDataset, GestureDatasetConfig  # noqa: E402
from train.replay_buffer import ReplayBuffer  # noqa: E402


def test_gesture_dataset_produces_samples():
    config = GestureDatasetConfig(dataset_id="balabit", sequence_length=32, max_gestures=20)
    dataset = GestureDataset(config)
    assert len(dataset) > 0
    sequence, features, label = dataset[0]
    assert sequence.shape == (32, 3)
    assert features.dim() == 1
    assert label.ndim == 0
    assert torch.isfinite(sequence).all()


def test_gesture_dataset_cache_roundtrip(tmp_path):
    cache_dir = tmp_path / "cache"
    config = GestureDatasetConfig(
        dataset_id="balabit",
        sequence_length=32,
        max_gestures=10,
        cache_enabled=True,
        cache_dir=str(cache_dir),
        use_generated_negatives=False,
    )
    dataset_first = GestureDataset(config)
    assert len(dataset_first) > 0
    dataset_second = GestureDataset(config)
    assert len(dataset_second) == len(dataset_first)


def test_replay_buffer_add_and_sample(tmp_path):
    buffer_path = tmp_path / "replay.pt"
    buffer = ReplayBuffer(path=buffer_path, max_size=10)
    sequences = torch.randn(5, 32, 3)
    sequences[..., 2] = sequences[..., 2].abs() + 1e-2
    features = torch.randn(5, 16)
    buffer.add(sequences, features)
    buffer.save()

    loaded = ReplayBuffer.load(buffer_path, max_size=10)
    seq_sample, feat_sample = loaded.sample(3)
    assert seq_sample.shape[0] == 3
    assert feat_sample.shape[0] == 3
