import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from features.neuromotor import tensor_to_gesture
from features.sigma_lognormal import (
    sigma_lognormal_features_numpy,
    sigma_lognormal_features_torch,
)


def _random_sequence(length: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    dx = torch.randn(length, generator=g)
    dy = torch.randn(length, generator=g)
    dt = torch.rand(length, generator=g).abs() + 0.01
    return torch.stack([dx, dy, dt], dim=-1)


def test_sigma_features_torch_matches_numpy_single():
    for i in range(10):
        seq = _random_sequence(64, 1234 + i)
        gesture = tensor_to_gesture(seq)
        expected = sigma_lognormal_features_numpy(gesture)
        result = sigma_lognormal_features_torch(seq)
        np.testing.assert_allclose(result.cpu().numpy(), expected, rtol=1e-4, atol=1e-4)


def test_sigma_features_torch_matches_numpy_batch():
    sequences = torch.stack([_random_sequence(64, 2000 + i) for i in range(8)], dim=0)
    expected = []
    for seq in sequences:
        gesture = tensor_to_gesture(seq)
        expected.append(sigma_lognormal_features_numpy(gesture))
    expected = np.stack(expected, axis=0)

    result = sigma_lognormal_features_torch(sequences).cpu().numpy()
    np.testing.assert_allclose(result, expected, rtol=1e-4, atol=1e-4)


def test_sigma_features_gpu_matches_cpu():
    if not torch.cuda.is_available():  # pragma: no cover - depends on CI hardware
        return
    seq = _random_sequence(64, 4242)
    gesture = tensor_to_gesture(seq)
    expected = sigma_lognormal_features_numpy(gesture)

    result_gpu = sigma_lognormal_features_torch(seq.cuda()).cpu().numpy()
    np.testing.assert_allclose(result_gpu, expected, rtol=1e-4, atol=1e-4)
