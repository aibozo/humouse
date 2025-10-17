"""Helpers for tidying checkpoint directories after artifact packaging."""
from __future__ import annotations

import shutil
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import logging

logger = logging.getLogger(__name__)


def tidy_checkpoint_artifacts(
    checkpoint_dir: Path | str,
    *,
    targets: Optional[Iterable[str]] = None,
    archive_subdir: str = "archive",
    keep_archives: int = 3,
    dry_run: bool = False,
) -> Dict[str, List[str]]:
    """Archive or prune bulky artifacts (e.g. samples/plots) under a checkpoint directory.

    Parameters
    ----------
    checkpoint_dir
        The directory containing run artifacts (e.g. ``checkpoints/gan``).
    targets
        Directory names relative to ``checkpoint_dir`` that should be archived.
        Defaults to ``("samples", "plots")``.
    archive_subdir
        Name of the folder within ``checkpoint_dir`` that will hold archived
        artifacts.
    keep_archives
        Number of most-recent archive folders to retain; older ones are removed.
    dry_run
        When ``True`` no filesystem changes are made and the function reports what
        it *would* do.

    Returns
    -------
    Dict[str, List[str]]
        Mapping containing lists of archived and pruned paths.
    """

    checkpoint_path = Path(checkpoint_dir).expanduser().resolve()
    if not checkpoint_path.exists():
        logger.debug("Checkpoint directory %s does not exist; skipping tidy", checkpoint_path)
        return {"archived": [], "pruned": []}

    target_names = list(targets) if targets is not None else ["samples", "plots"]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_dir = checkpoint_path / archive_subdir
    actions: Dict[str, List[str]] = defaultdict(list)

    if not dry_run:
        archive_dir.mkdir(parents=True, exist_ok=True)

    for name in target_names:
        source = checkpoint_path / name
        if not source.exists():
            continue
        dest = archive_dir / f"{name}_{timestamp}"
        # Avoid collisions by appending a counter if needed
        counter = 1
        while dest.exists():
            dest = archive_dir / f"{name}_{timestamp}_{counter}"
            counter += 1
        logger.debug("Archiving %s to %s", source, dest)
        if not dry_run:
            shutil.move(str(source), str(dest))
        actions["archived"].append(str(dest))

    # Prune old archives beyond retention limit
    if keep_archives >= 0 and archive_dir.exists():
        archive_entries = sorted(
            [p for p in archive_dir.iterdir() if p.is_dir()],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for stale in archive_entries[keep_archives:]:
            logger.debug("Removing stale archive %s", stale)
            if not dry_run:
                shutil.rmtree(stale, ignore_errors=True)
            actions["pruned"].append(str(stale))

    return {key: list(value) for key, value in actions.items()}


__all__ = ["tidy_checkpoint_artifacts"]
