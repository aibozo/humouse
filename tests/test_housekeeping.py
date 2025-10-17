from __future__ import annotations

from pathlib import Path
import sys
import time

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from utils.housekeeping import tidy_checkpoint_artifacts  # noqa: E402


def _touch_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    file_path = path / "dummy.txt"
    file_path.write_text("dummy")


def test_tidy_checkpoint_artifacts_archives_and_prunes(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "checkpoints" / "gan"
    _touch_directory(checkpoint_dir / "samples")
    _touch_directory(checkpoint_dir / "plots")

    result = tidy_checkpoint_artifacts(checkpoint_dir, keep_archives=1)
    assert len(result["archived"]) == 2
    assert not (checkpoint_dir / "samples").exists()
    assert not (checkpoint_dir / "plots").exists()

    archive_dir = checkpoint_dir / "archive"
    archived_items = list(archive_dir.iterdir())
    assert archived_items, "expected archived artifacts"

    # Create additional archive entries to exercise pruning logic
    time.sleep(0.01)  # ensure differing mtime
    _touch_directory(checkpoint_dir / "samples")
    tidy_checkpoint_artifacts(checkpoint_dir, keep_archives=1)
    assert len(list(archive_dir.iterdir())) == 1


def test_tidy_checkpoint_artifacts_dry_run(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "checkpoints" / "detector"
    _touch_directory(checkpoint_dir / "samples")

    result = tidy_checkpoint_artifacts(checkpoint_dir, dry_run=True)
    assert result["archived"], "dry run still reports planned actions"
    assert (checkpoint_dir / "samples").exists(), "dry run should not move files"
