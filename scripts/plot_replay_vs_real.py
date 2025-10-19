#!/usr/bin/env python3
"""Generate trajectory plots and Sigma-Lognormal feature histograms for replay vs real."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import matplotlib


matplotlib.use("Agg")  # render without display

from data.dataset import GestureDataset, GestureDatasetConfig
from utils.plotting import generate_replay_vs_real_plots


def _load_real_data(
    dataset_id: str,
    sequence_length: int,
    max_gestures: int | None,
    *,
    canonicalize_path: bool,
    canonicalize_duration: bool,
) -> tuple[np.ndarray, np.ndarray]:
    cfg = GestureDatasetConfig(
        dataset_id=dataset_id,
        sequence_length=sequence_length,
        max_gestures=max_gestures,
        use_generated_negatives=False,
        cache_enabled=True,
        normalize_sequences=False,
        normalize_features=False,
        feature_mode="sigma_lognormal",
        canonicalize_path=canonicalize_path,
        canonicalize_duration=canonicalize_duration,
    )
    dataset = GestureDataset(cfg)
    sequences = [seq.numpy() for seq, _, _ in dataset.samples]
    if not sequences:
        raise RuntimeError("No real gestures loaded; check dataset configuration.")
    features = dataset.get_positive_features_tensor().numpy()
    return np.array(sequences, dtype=np.float32), features.astype(np.float32)


def _load_replay_sequences(
    replay_dir: Path,
    sequence_length: int,
) -> np.ndarray:
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
            if seq_arr.ndim != 3 or seq_arr.shape[2] != 3:
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


def _sample_indices(rng: np.random.Generator, total: int, count: int) -> np.ndarray:
    count = min(total, count)
    if count <= 0:
        raise ValueError("Sample count must be positive")
    return rng.choice(total, size=count, replace=False)


def _plot_sequences(ax, sequences: Sequence[np.ndarray], title: str) -> None:
    for seq in sequences:
        deltas = seq[:, :2]
        cumulative = np.vstack([np.zeros(2, dtype=np.float32), np.cumsum(deltas, axis=0)])
        ax.plot(cumulative[:, 0], cumulative[:, 1], alpha=0.5)
    ax.set_title(title)
    ax.set_xlabel("Δx accumulated")
    ax.set_ylabel("Δy accumulated")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.2)


def _compute_sigma_features(sequences: Iterable[np.ndarray]) -> np.ndarray:
    features: list[np.ndarray] = []
    for seq in sequences:
        tensor = torch.from_numpy(seq)
        feat = sigma_lognormal_features_from_sequence(tensor)
        features.append(feat.numpy())
    return np.stack(features)


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay vs real Sigma-Lognormal feature comparison")
    parser.add_argument("--dataset", default="balabit", help="Dataset identifier for real gestures")
    parser.add_argument("--sequence-length", type=int, default=64, help="Gesture sequence length")
    parser.add_argument("--max-gestures", type=int, default=4096, help="Maximum real gestures to load")
    parser.add_argument("--replay-dir", type=Path, required=True, help="Directory containing GAN replay NPZ files")
    parser.add_argument("--output-dir", type=Path, default=Path("analysis/replay_vs_real"), help="Where plots are saved")
    parser.add_argument("--seed", type=int, default=1337, help="Random seed for sampling")
    parser.add_argument("--plot-count", type=int, default=12, help="Number of sequences to visualise per set")
    parser.add_argument("--feature-samples", type=int, default=2000, help="Number of sequences to use for histograms")
    parser.add_argument("--bins", type=int, default=40, help="Histogram bins")
    parser.add_argument(
        "--no-canon-path",
        action="store_false",
        dest="canon_path",
        help="Disable unit-path-length canonicalisation for real gestures",
    )
    parser.add_argument(
        "--no-canon-duration",
        action="store_false",
        dest="canon_duration",
        help="Disable unit-duration canonicalisation for real gestures",
    )
    parser.set_defaults(canon_path=True, canon_duration=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    real_sequences, real_features = _load_real_data(
        dataset_id=args.dataset,
        sequence_length=args.sequence_length,
        max_gestures=None if args.max_gestures <= 0 else args.max_gestures,
        canonicalize_path=args.canon_path,
        canonicalize_duration=args.canon_duration,
    )
    generate_replay_vs_real_plots(
        real_sequences,
        real_features,
        args.replay_dir,
        args.output_dir,
        seed=args.seed,
        plot_count=args.plot_count,
        feature_samples=args.feature_samples if args.feature_samples > 0 else len(real_sequences),
        bins=args.bins,
        sequence_length=args.sequence_length,
    )


if __name__ == "__main__":
    main()
