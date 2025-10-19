"""Plotting utilities for experiment metrics."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd
import torch

from features import sigma_lognormal_features_from_sequence


def _build_feature_names() -> list[str]:
    stats = ["first_max", "first_min", "first_mean", "second_max", "second_min", "second_mean"]
    groups = ["distance", "t0", "mu", "sigma", "theta_start", "theta_end"]
    names: list[str] = []
    for group in groups:
        for stat in stats:
            names.append(f"{group}_{stat}")
    names.append("stroke_count")
    return names


FEATURE_NAMES = _build_feature_names()


def plot_metric_trends(
    csv_path: Path | str,
    output_path: Path | str,
    x_column: str,
    metric_columns: Sequence[str],
    title: str,
) -> Path:
    csv_path = Path(csv_path)
    if not csv_path.exists():
        return Path(output_path)
    df = pd.read_csv(csv_path)
    if x_column not in df.columns:
        raise ValueError(f"{x_column} not present in {csv_path}")

    plt.figure(figsize=(10, 6))
    for metric in metric_columns:
        if metric in df.columns:
            plt.plot(df[x_column], df[metric], label=metric)
    plt.title(title)
    plt.xlabel(x_column)
    plt.ylabel("value")
    plt.legend()
    plt.grid(True, alpha=0.3)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    return output_path


def plot_roc_curve(
    fpr: Iterable[float],
    tpr: Iterable[float],
    output_path: Path | str,
    title: str = "ROC Curve",
) -> Path:
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label="ROC")
    plt.plot([0, 1], [0, 1], "--", color="gray", label="random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    return output_path


def _plot_sequences(ax, sequences: Sequence[np.ndarray], title: str) -> None:
    for seq in sequences:
        deltas = seq[:, :2]
        cumulative = np.concatenate(
            [np.zeros((1, 2), dtype=deltas.dtype), np.cumsum(deltas, axis=0)],
            axis=0,
        )
        ax.plot(cumulative[:, 0], cumulative[:, 1], alpha=0.5)
    ax.set_title(title)
    ax.set_xlabel("Δx accumulated")
    ax.set_ylabel("Δy accumulated")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.2)


def _plot_sequence_grid(
    output_path: Path,
    real_sequences: Sequence[np.ndarray],
    replay_sequences: Sequence[np.ndarray],
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    _plot_sequences(axes[0], real_sequences, "Real gestures")
    _plot_sequences(axes[1], replay_sequences, "Replay gestures")
    fig.suptitle("Accumulated Trajectory Samples", fontsize=14)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _compute_sigma_features(sequences: Sequence[np.ndarray]) -> np.ndarray:
    tensors = [torch.from_numpy(seq) for seq in sequences]
    return np.stack([sigma_lognormal_features_from_sequence(t).numpy() for t in tensors])


def _plot_feature_histograms(
    output_path: Path,
    real_features: np.ndarray,
    replay_features: np.ndarray,
    bins: int,
) -> None:
    feature_dim = real_features.shape[1]
    cols = 4
    rows = math.ceil(feature_dim / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3), constrained_layout=True)
    axes_iter = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

    for idx, ax in enumerate(axes_iter):
        if idx >= feature_dim:
            ax.axis("off")
            continue
        ax.hist(
            real_features[:, idx],
            bins=bins,
            alpha=0.6,
            density=True,
            label="Real" if idx == 0 else None,
            color="tab:blue",
        )
        ax.hist(
            replay_features[:, idx],
            bins=bins,
            alpha=0.6,
            density=True,
            label="Replay" if idx == 0 else None,
            color="tab:orange",
        )
        feature_name = FEATURE_NAMES[idx] if idx < len(FEATURE_NAMES) else f"feature_{idx}"
        ax.set_title(feature_name, fontsize=10)
        ax.tick_params(labelsize=8)

    handles, labels = axes_iter[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right")
    fig.suptitle("Sigma-Lognormal Feature Distributions", fontsize=14)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _load_replay_sequences(replay_dir: Path, sequence_length: int) -> np.ndarray:
    files = sorted(replay_dir.glob("**/*.npz"))
    if not files:
        raise RuntimeError(f"No .npz sequences found under {replay_dir}")

    batches: list[np.ndarray] = []
    for file_path in files:
        with np.load(file_path) as data:
            sequences = data.get("sequences")
            if sequences is None:
                continue
            seq_arr = sequences.astype(np.float32)
            if seq_arr.ndim != 3 or seq_arr.shape[2] < 2:
                continue
            batches.append(seq_arr)

    if not batches:
        raise RuntimeError(f"No usable sequences found in {replay_dir}")

    concatenated = np.concatenate(batches, axis=0)
    if concatenated.shape[1] != sequence_length:
        raise RuntimeError(
            f"Replay sequences length {concatenated.shape[1]} != expected {sequence_length}"
        )
    return concatenated


def generate_replay_vs_real_plots(
    real_sequences: np.ndarray | torch.Tensor,
    real_features: np.ndarray | torch.Tensor,
    replay_dir: Path | str,
    output_dir: Path | str,
    *,
    seed: int = 1337,
    plot_count: int = 12,
    feature_samples: int = 2000,
    bins: int = 40,
    sequence_length: int | None = None,
) -> dict[str, Any]:
    real_sequences_np = real_sequences.detach().cpu().numpy() if isinstance(real_sequences, torch.Tensor) else np.asarray(real_sequences)
    real_features_np = real_features.detach().cpu().numpy() if isinstance(real_features, torch.Tensor) else np.asarray(real_features)

    if real_sequences_np.ndim != 3:
        raise ValueError("real_sequences must have shape (N, L, C)")
    if real_features_np.ndim != 2:
        raise ValueError("real_features must have shape (N, F)")

    if sequence_length is None:
        sequence_length = real_sequences_np.shape[1]

    replay_dir = Path(replay_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    replay_sequences = _load_replay_sequences(replay_dir, sequence_length)
    rng = np.random.default_rng(seed)

    plot_count = max(1, min(plot_count, len(real_sequences_np), len(replay_sequences)))
    feature_samples = max(1, min(feature_samples, len(real_sequences_np), len(replay_sequences)))

    real_plot_idx = rng.choice(len(real_sequences_np), size=plot_count, replace=False)
    replay_plot_idx = rng.choice(len(replay_sequences), size=plot_count, replace=False)
    real_plot_samples = real_sequences_np[real_plot_idx]
    replay_plot_samples = replay_sequences[replay_plot_idx]

    real_feat_idx = rng.choice(len(real_sequences_np), size=feature_samples, replace=False)
    replay_feat_idx = rng.choice(len(replay_sequences), size=feature_samples, replace=False)
    real_feature_samples = real_features_np[real_feat_idx]
    replay_feature_samples = _compute_sigma_features(replay_sequences[replay_feat_idx])

    sequence_plot_path = output_dir / "trajectory_samples.png"
    feature_hist_path = output_dir / "sigma_feature_histograms.png"
    stats_path = output_dir / "histogram_summary.json"

    _plot_sequence_grid(sequence_plot_path, real_plot_samples, replay_plot_samples)
    _plot_feature_histograms(feature_hist_path, real_feature_samples, replay_feature_samples, bins)

    summary = {
        "real_sequence_count": int(len(real_sequences_np)),
        "replay_sequence_count": int(len(replay_sequences)),
        "feature_dimension": int(real_features_np.shape[1]),
        "feature_hist_samples": int(feature_samples),
        "sequence_plot_samples": int(plot_count),
        "sequence_plot_path": str(sequence_plot_path),
        "feature_hist_path": str(feature_hist_path),
        "replay_dir": str(replay_dir),
    }
    stats_path.write_text(json.dumps(summary, indent=2))
    return summary


__all__ = ["plot_metric_trends", "plot_roc_curve", "generate_replay_vs_real_plots"]
