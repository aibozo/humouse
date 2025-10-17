from pathlib import Path
import sys

import numpy as np

import pytest
from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from utils.logging import LoggingConfig, write_summary_json  # noqa: E402
from utils.eval import (
    compute_roc_metrics,
    feature_distribution_metrics,
    sequence_diversity_metric,
    roc_curve_points,
)  # noqa: E402


@pytest.mark.parametrize("config_name", ["train_gan", "train_detector"])
def test_experiment_config_loads(config_name: str):
    path = PROJECT_ROOT / "conf" / "experiment" / f"{config_name}.yaml"
    cfg = OmegaConf.load(path)
    assert "logging" in cfg
    assert "data" in cfg


def test_logging_config_instantiation():
    cfg = LoggingConfig(target="wandb", project="mouse-gan", mode="offline")
    assert cfg.project == "mouse-gan"
    assert cfg.target == "wandb"


def test_compute_roc_metrics():
    labels = np.array([0, 0, 1, 1])
    logits = np.array([-2.0, -1.0, 1.0, 2.0])
    metrics = compute_roc_metrics(labels, logits)
    assert 0.0 <= metrics["roc_auc"] <= 1.0
    assert 0.0 <= metrics["pr_auc"] <= 1.0


def test_feature_distribution_metrics():
    real = np.random.randn(100, 5)
    fake = np.random.randn(100, 5)
    metrics = feature_distribution_metrics(real, fake)
    assert "mean_l1" in metrics
    assert metrics["mean_l1"] >= 0


def test_sequence_diversity_metric():
    seq = np.random.randn(16, 64, 3)
    div = sequence_diversity_metric(seq)
    assert isinstance(div, float)


def test_roc_curve_points():
    labels = np.array([0, 0, 1, 1])
    logits = np.array([-2.0, -1.0, 1.0, 2.0])
    fpr, tpr = roc_curve_points(labels, logits)
    assert fpr.shape == tpr.shape


def test_write_summary_json(tmp_path):
    summary_path = tmp_path / "summary.json"
    write_summary_json(summary_path, {"a": 1})
    assert summary_path.exists()
