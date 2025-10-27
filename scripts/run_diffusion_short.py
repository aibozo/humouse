#!/usr/bin/env python3
"""Run a short diffusion training job (e.g., 5 epochs) and collect logs."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import List


def _build_overrides(args: argparse.Namespace, summary_path: Path, checkpoint_dir: Path, run_dir: Path) -> List[str]:
    base = [
        f"+experiment_name={args.run_name}",
        f"+seed={args.seed}",
        f"training.epochs={args.epochs}",
        f"training.summary_path={summary_path}",
        f"training.checkpoint_dir={checkpoint_dir}",
        f"++hydra.run.dir={run_dir}",
        "++hydra.output_subdir=null",
    ]
    overrides = base + args.override

    def _ensure_prefix(item: str) -> str:
        prefixes = ("training.", "model.", "diffusion.", "data.", "seed=", "experiment_name=", "hydra.")
        if item.startswith(("+", "-")):
            return item
        if any(item.startswith(pref) for pref in prefixes):
            return "+" + item
        return item

    return [_ensure_prefix(item) for item in overrides]


def run_short_training(args: argparse.Namespace) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.name or f"diff_short_{timestamp}"
    run_dir = Path(args.output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    summary_path = run_dir / "summary.json"
    checkpoint_dir = run_dir / "checkpoints"

    overrides = _build_overrides(args, summary_path, checkpoint_dir, run_dir)

    cmd = [
        ".venv/bin/python",
        "-m",
        "diffusion.train",
        f"--config-name={args.config}",
        *overrides,
    ]

    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")

    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=args.repo_root, env=env)

    if summary_path.exists():
        print(summary_path.read_text())
    else:
        print("Warning: summary file not found at", summary_path)
    return run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a short diffusion training job for quick diagnostics.")
    parser.add_argument("--config", default="diffusion/exp_short_cosine", help="Hydra config to use.")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs.")
    parser.add_argument("--seed", type=int, default=1337, help="Random seed.")
    parser.add_argument("--output-dir", default="outputs/diffusion_short_runs", help="Directory for run artifacts.")
    parser.add_argument("--name", default=None, help="Optional run name (auto-generated if omitted).")
    parser.add_argument("--run-name", default="diff_short_run", help="Experiment name override.")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Additional Hydra overrides (repeatable). Example: --override data.batch_size=64",
    )
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[1]), help="Repository root path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = run_short_training(args)
    summary_path = run_dir / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
