"""Plot sample gestures from a GestureDataset for sanity checking."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np

from data.dataset import GestureDataset, GestureDatasetConfig


def _accumulate(sequence: np.ndarray) -> np.ndarray:
    """Convert (Δx, Δy, Δt) to absolute (x, y) positions."""
    deltas = sequence[:, :2]
    positions = np.cumsum(deltas, axis=0)
    return positions


def _plot_samples(ax: plt.Axes, sequences: Sequence[np.ndarray], *, title: str) -> None:
    ax.set_title(title)
    for seq in sequences:
        pts = _accumulate(seq)
        ax.plot(pts[:, 0], pts[:, 1], linewidth=1.2)
    ax.set_xlabel(r"Δx accumulated")
    ax.set_ylabel(r"Δy accumulated")
    ax.axis("equal")
    ax.grid(alpha=0.2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot raw gestures from a dataset split.")
    parser.add_argument("--dataset", required=True, help="Dataset identifier (e.g. balabit)")
    parser.add_argument("--sequence-length", type=int, default=200, help="Resampled gesture length")
    parser.add_argument("--sampling-rate", type=float, default=None, help="Optional sampling rate in Hz (e.g. 200)")
    parser.add_argument("--max-gestures", type=int, default=512, help="Maximum gestures to load from dataset")
    parser.add_argument("--split", default="train", help="Dataset split to draw from")
    parser.add_argument("--samples", type=int, default=12, help="Number of gestures to plot")
    parser.add_argument("--output", type=Path, required=True, help="Output PNG path")
    parser.add_argument("--seed", type=int, default=1337, help="Random seed for sample selection")
    parser.add_argument("--canonicalize-path", action="store_true", help="Enable unit path canonicalisation")
    parser.add_argument("--canonicalize-duration", action="store_true", help="Enable unit duration canonicalisation")
    parser.add_argument("--normalize-sequences", action="store_true", help="Apply dataset sequence normalization")
    parser.add_argument("--feature-mode", default="sigma_lognormal", help="Feature mode to load")
    args = parser.parse_args()

    cfg = GestureDatasetConfig(
        dataset_id=args.dataset,
        sequence_length=args.sequence_length,
        sampling_rate=args.sampling_rate,
        max_gestures=args.max_gestures,
        cache_enabled=False,
        split=args.split,
        feature_mode=args.feature_mode,
        canonicalize_path=args.canonicalize_path,
        canonicalize_duration=args.canonicalize_duration,
        normalize_sequences=args.normalize_sequences,
        normalize_features=False,
    )

    dataset = GestureDataset(cfg)
    if len(dataset) == 0:
        raise RuntimeError("Dataset produced zero gestures; check preprocessing parameters.")

    rng = np.random.default_rng(args.seed)
    sample_indices = rng.choice(len(dataset), size=min(args.samples, len(dataset)), replace=False)
    sequences = [dataset[i][0].numpy() for i in sample_indices]

    fig, ax = plt.subplots(figsize=(10, 8))
    _plot_samples(ax, sequences, title=f"{args.dataset} (N={len(sequences)})")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.output, dpi=200)
    plt.close(fig)

    print(f"Saved sample plot to {args.output}")


if __name__ == "__main__":
    main()
