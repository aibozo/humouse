"""Experiment logging helpers (Weights & Biases friendly)."""
from __future__ import annotations

import csv
import json
import logging
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Generator, Iterable, Optional

logger = logging.getLogger(__name__)


@dataclass
class LoggingConfig:
    target: str = "wandb"
    project: str = "mouse-gan"
    entity: Optional[str] = None
    mode: str = "offline"
    log_interval: int = 100
    checkpoint_dir: str = "checkpoints"
    tags: list[str] = field(default_factory=list)


def _init_wandb(config: LoggingConfig, run_config: Dict[str, Any]):
    try:
        import wandb
    except ImportError:  # pragma: no cover - optional dependency
        logger.warning("wandb not installed; skipping logging.")
        return None

    wandb.init(
        project=config.project,
        entity=config.entity,
        mode=config.mode,
        tags=config.tags or None,
        config=run_config,
    )
    return wandb.run


def _finish_wandb(run) -> None:  # pragma: no cover - simple wrapper
    if run is None:
        return
    try:
        run.finish()
    except Exception as exc:  # pragma: no cover
        logger.warning("Failed to finish wandb run: %s", exc)


@contextmanager
def experiment_logger(config: LoggingConfig, run_config: Dict[str, Any]) -> Generator[Any, None, None]:
    """Context manager that initialises experiment logging if enabled."""
    run = None
    if config.target == "wandb":
        run = _init_wandb(config, run_config)
    else:
        logger.info("Experiment logging disabled (target=%s)", config.target)
    try:
        yield run
    finally:
        _finish_wandb(run)


class CSVMetricLogger:
    """Append-only CSV metric logger."""

    def __init__(self, path: Path | str, fieldnames: Optional[list[str]] = None) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fieldnames = fieldnames
        if not self.path.exists() and self.fieldnames:
            self._write_header()

    def _write_header(self) -> None:
        if not self.fieldnames:
            return
        with self.path.open("w", newline="") as fp:
            writer = csv.DictWriter(fp, fieldnames=self.fieldnames)
            writer.writeheader()

    def log(self, metrics: Dict[str, Any]) -> None:
        if self.fieldnames is None:
            self.fieldnames = sorted(metrics.keys())
            if not self.path.exists():
                self._write_header()
        row = {key: metrics.get(key, "") for key in self.fieldnames}
        with self.path.open("a", newline="") as fp:
            writer = csv.DictWriter(fp, fieldnames=self.fieldnames)
            if fp.tell() == 0:
                writer.writeheader()
            writer.writerow(row)


def write_summary_json(path: Path | str, summary: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fp:
        json.dump(summary, fp, indent=2)


def log_wandb_artifact(
    run,
    artifact_name: str,
    artifact_type: str,
    files: Iterable[Path],
    dirs: Iterable[Path],
) -> None:
    if run is None:
        return
    try:
        import wandb
    except ImportError:  # pragma: no cover
        return

    artifact = wandb.Artifact(artifact_name, type=artifact_type)
    for file_path in files:
        path = Path(file_path)
        if path.exists():
            artifact.add_file(str(path), name=path.name)
    for dir_path in dirs:
        path = Path(dir_path)
        if path.exists():
            artifact.add_dir(str(path), name=path.name)
    run.log_artifact(artifact)


__all__ = [
    "LoggingConfig",
    "experiment_logger",
    "CSVMetricLogger",
    "write_summary_json",
    "log_wandb_artifact",
]
