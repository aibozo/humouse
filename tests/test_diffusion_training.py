import types

import torch

from diffusion.train import (
    DiffusionEvalPrepConfig,
    _diffusion_classifier_metrics_with_val,
    _min_snr_weight,
)


class _DummyDataset:
    def __init__(self, sequences):
        self.samples = [(seq, torch.zeros(1), torch.tensor(1.0)) for seq in sequences]

    def denormalize_sequences(self, sequences: torch.Tensor) -> torch.Tensor:
        return sequences

    def normalize_features(self, features: torch.Tensor) -> torch.Tensor:
        return features

    def normalize_sequences(self, sequences: torch.Tensor) -> torch.Tensor:
        return sequences


class _IdentitySampler:
    def __init__(self, sequences: torch.Tensor, timesteps: int = 8):
        self._sequences = sequences
        self.schedule = types.SimpleNamespace(timesteps=timesteps)

    def sample(self, n: int, seq_len: int, steps: int = 1) -> torch.Tensor:
        return self._sequences[:n].clone()


def test_diffusion_classifier_metrics_real_equals_fake():
    torch.manual_seed(7)
    seq_len = 16
    samples = []
    for scale in torch.linspace(0.5, 1.5, steps=80):
        seq = torch.zeros(seq_len, 3)
        seq[:, 0] = scale / seq_len
        seq[:, 1] = (scale * 0.5) / seq_len
        seq[:, 2] = 1.0 / seq_len
        samples.append(seq)
    dataset = _DummyDataset(samples)
    stacked = torch.stack(samples)
    sampler = _IdentitySampler(stacked)

    metrics = _diffusion_classifier_metrics_with_val(
        dataset,
        real_dataset=dataset,
        sampler=sampler,
        samples=64,
        seq_len=seq_len,
        steps=5,
        seed=1337,
        real_label="dummy",
        prep_cfg=DiffusionEvalPrepConfig(),
        timing_sampler=None,
    )
    assert metrics, "Expected classifier metrics when using identical real/fake batches."
    assert abs(metrics["c2st_accuracy"] - 0.5) < 0.15
    assert abs(metrics["c2st_auc"] - 0.5) < 0.2


def test_min_snr_weight_matches_reference():
    log_snr = torch.tensor([-4.0, 0.0, 2.0])
    gamma = 5.0
    expected = (gamma + log_snr.exp()) / (log_snr.exp() + 1.0)
    actual = _min_snr_weight(log_snr, gamma)
    torch.testing.assert_close(actual, expected)
    ones = _min_snr_weight(log_snr, 0.0)
    assert torch.allclose(ones, torch.ones_like(log_snr))
