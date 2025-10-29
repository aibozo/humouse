#!/usr/bin/env python3
"""Optuna-based hyperparameter search for diffusion training."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import optuna

DEFAULT_CONFIG = "diffusion/exp_short_cosine"


def _build_channel_mults(trial: optuna.Trial) -> List[float]:
    options = {
        "narrow": [1.0, 1.5, 2.0],
        "balanced": [1.0, 1.5, 2.0, 2.0],
        "wide": [1.0, 2.0, 3.0],
    }
    key = trial.suggest_categorical("channel_mults", list(options.keys()))
    return options[key]


def _objective(trial: optuna.Trial, args: argparse.Namespace) -> float:
    trial_dir = Path(args.log_dir) / f"trial_{trial.number:03d}"
    if trial_dir.exists():
        shutil.rmtree(trial_dir)
    trial_dir.mkdir(parents=True, exist_ok=True)
    summary_path = trial_dir / "summary.json"
    checkpoint_dir = trial_dir / "checkpoints"

    params: Dict[str, Any] = {}
    params["objective"] = trial.suggest_categorical("objective", ["epsilon", "v"])
    params["lr"] = trial.suggest_float("lr", 2e-5, 2e-4, log=True)
    params["ema_decay"] = trial.suggest_float("ema_decay", 0.999, 0.9999)
    params["weight_decay"] = trial.suggest_categorical("weight_decay", [0.0, 1e-4, 1e-3])
    params["base_channels"] = trial.suggest_categorical("base_channels", [96, 128, 160])
    params["dropout"] = trial.suggest_categorical("dropout", [0.0, 0.1, 0.2])
    params["self_condition"] = trial.suggest_categorical("self_condition", [True, False])
    params["timesteps"] = trial.suggest_categorical("timesteps", [256, 512, 1000])
    params["schedule"] = trial.suggest_categorical("schedule", ["cosine", "linear"])
    params["time_stretch"] = trial.suggest_categorical("time_stretch", [0.0, 0.15])
    params["jitter_std"] = trial.suggest_categorical("jitter_std", [0.0, 0.02])
    params["mirror_prob"] = trial.suggest_categorical("mirror_prob", [0.0, 0.5])
    params["channel_mults"] = _build_channel_mults(trial)

    overrides = [
        f"+experiment_name=search_trial_{trial.number:03d}",
        f"+seed={args.seed + trial.number}",
        f"training.epochs={args.epochs}",
        "training.eval_interval=1",
        "training.eval_classifier=true",
        f"training.summary_path={summary_path}",
        f"training.checkpoint_dir={checkpoint_dir}",
        f"training.objective={params['objective']}",
        f"training.lr={params['lr']}",
        f"training.ema_decay={params['ema_decay']}",
        f"training.weight_decay={params['weight_decay']}",
        f"training.sample_eval_count={args.sample_eval_count}",
        f"training.sample_eval_steps={args.sample_eval_steps}",
        f"model.base_channels={params['base_channels']}",
        f"model.dropout={params['dropout']}",
        f"model.self_condition={'true' if params['self_condition'] else 'false'}",
        f"model.channel_mults={params['channel_mults']}",
        f"diffusion.timesteps={params['timesteps']}",
        f"diffusion.schedule={params['schedule']}",
        f"data.time_stretch={params['time_stretch']}",
        f"data.jitter_std={params['jitter_std']}",
        f"data.mirror_prob={params['mirror_prob']}",
        f"data.max_train_gestures={args.max_train_gestures}",
        f"data.max_val_gestures={args.max_val_gestures}",
        f"training.classifier_samples={args.classifier_samples}",
        f"hydra.run.dir={trial_dir}",
        "hydra.output_subdir=null",
    ]

    channel_mult_str = ",".join(f"{v:.2f}" for v in params["channel_mults"])
    overrides = [
        ov if "model.channel_mults" not in ov else f"model.channel_mults=[{channel_mult_str}]"
        for ov in overrides
    ]

    def _prefix_override(s: str) -> str:
        prefixes = ("training.", "model.", "diffusion.", "data.", "seed=")
        for pref in prefixes:
            if s.startswith(pref):
                return "+" + s
        return s

    overrides = [_prefix_override(ov) for ov in overrides]

    cmd = [
        ".venv/bin/python",
        "-m",
        "diffusion.train",
        f"--config-name={args.config}",
        *overrides,
    ]

    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")

    try:
        subprocess.run(cmd, check=True, cwd=args.repo_root, env=env)
    except subprocess.CalledProcessError as exc:
        raise optuna.TrialPruned(f"Training failed: {exc}") from exc

    if not summary_path.exists():
        raise optuna.TrialPruned("Missing summary output")

    summary = json.loads(summary_path.read_text())
    (trial_dir / "params.json").write_text(json.dumps(params, indent=2))

    val_loss = summary.get("last_val_loss")
    if val_loss is None:
        raise optuna.TrialPruned("No validation loss recorded")
    scale_origin = summary.get("real_norm_std_mean", 1.0)
    scale_target = summary.get("sample_norm_std_mean", 1.0)
    scale_gap = abs(scale_target - scale_origin)
    c2st_val = summary.get("val_c2st_accuracy", 1.0)
    feature_delta = summary.get("val_feature_delta_mean", 0.0) or 0.0

    c2st_penalty = max(0.0, c2st_val - 0.5)
    score = float(val_loss) + args.scale_weight * scale_gap + args.c2st_weight * c2st_penalty + args.feature_weight * feature_delta

    trial.set_user_attr("val_loss", float(val_loss))
    trial.set_user_attr("scale_gap", float(scale_gap))
    trial.set_user_attr("val_c2st", float(c2st_val))
    trial.set_user_attr("feature_delta", float(feature_delta))

    return score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diffusion hyperparameter search")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Hydra config name")
    parser.add_argument("--trials", type=int, default=10, help="Number of Optuna trials")
    parser.add_argument("--epochs", type=int, default=5, help="Epochs per trial")
    parser.add_argument("--seed", type=int, default=1337, help="Base random seed")
    parser.add_argument("--log-dir", default="outputs/search_trials", help="Directory for trial artifacts")
    parser.add_argument("--scale-weight", type=float, default=1.0)
    parser.add_argument("--c2st-weight", type=float, default=1.0)
    parser.add_argument("--feature-weight", type=float, default=1e-6)
    parser.add_argument("--sample-eval-count", type=int, default=256)
    parser.add_argument("--sample-eval-steps", type=int, default=30)
    parser.add_argument("--classifier-samples", type=int, default=512)
    parser.add_argument("--max-train-gestures", type=int, default=4096)
    parser.add_argument("--max-val-gestures", type=int, default=512)
    parser.add_argument("--storage", default=None, help="Optuna storage URI (optional)")
    parser.add_argument("--study-name", default="diffusion_search")
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[1]))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    if args.storage:
        study = optuna.create_study(
            direction="minimize",
            study_name=args.study_name,
            storage=args.storage,
            load_if_exists=True,
        )
    else:
        study = optuna.create_study(direction="minimize", study_name=args.study_name)

    objective = lambda trial: _objective(trial, args)
    study.optimize(objective, n_trials=args.trials)

    best = study.best_trial
    print("Best trial:")
    print(f"  number={best.number}")
    print(f"  value={best.value:.6f}")
    print("  params:")
    for k, v in best.params.items():
        print(f"    {k}: {v}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(1)
