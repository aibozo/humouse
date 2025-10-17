from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from tools.aggregate_checkpoints import aggregate_checkpoints  # noqa: E402


@pytest.fixture
def checkpoint_root(tmp_path: Path) -> Path:
    root = tmp_path / "checkpoints"
    root.mkdir()

    gan_run = root / "gan" / "run_001"
    gan_run.mkdir(parents=True)
    gan_summary = {
        "final_epoch": 10,
        "final_step": 1234,
        "g_loss": 0.1234,
        "d_loss": 0.4321,
        "feature_l1": 0.05,
        "feature_cov_diff": 0.02,
        "diversity_xy": 0.88,
        "replay_buffer_size": 256,
    }
    (gan_run / "gan_summary.json").write_text(json.dumps(gan_summary))

    detector_run = root / "detector" / "baseline"
    detector_run.mkdir(parents=True)
    detector_summary = {
        "best_val_epoch": 7,
        "final_metrics": {
            "val_loss": 0.22,
            "roc_auc": 0.94,
            "pr_auc": 0.91,
            "fpr_at_95_tpr": 0.12,
        },
        "cross_dataset": {
            "balabit_test": {"roc_auc": 0.9, "pr_auc": 0.87, "fpr_at_95_tpr": 0.2},
        },
        "cross_dataset_details": {
            "balabit_test": {
                "dataset_id": "balabit",
                "split": "test",
                "user_filter": ["user01", "user02"],
                "description": "Balabit test split",
            },
        },
    }
    (detector_run / "detector_summary.json").write_text(json.dumps(detector_summary))

    return root


def test_aggregate_checkpoints_collects_runs(checkpoint_root: Path) -> None:
    result = aggregate_checkpoints(checkpoint_root)

    assert len(result.gan_rows) == 1
    gan_row = result.gan_rows[0]
    assert pytest.approx(0.1234, rel=1e-4) == gan_row["g_loss"]
    assert gan_row["run"] == "gan/run_001"

    assert len(result.detector_rows) == 1
    detector_row = result.detector_rows[0]
    assert detector_row["run"] == "detector/baseline"
    assert pytest.approx(0.94, rel=1e-4) == detector_row["roc_auc"]

    assert len(result.cross_rows) == 1
    cross_row = result.cross_rows[0]
    assert cross_row["dataset"] == "balabit_test"
    assert cross_row["dataset_id"] == "balabit"
    assert cross_row["split"] == "test"
    assert cross_row["user_filter"] == "user01,user02"
    assert cross_row["description"] == "Balabit test split"


def test_aggregate_checkpoints_handles_missing_root(tmp_path: Path) -> None:
    result = aggregate_checkpoints(tmp_path / "missing")
    assert not result.gan_rows
    assert not result.detector_rows
    assert not result.cross_rows
