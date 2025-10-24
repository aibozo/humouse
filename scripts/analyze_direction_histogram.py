#!/usr/bin/env python3
"""Compute direction histogram for click-to-click gestures."""
from __future__ import annotations

import argparse
import math
from collections import Counter
from typing import Iterable

import numpy as np

from data.dataset import GestureDataset, GestureDatasetConfig

NUM_BUCKETS = 8


def direction_bucket(dx: float, dy: float, *, epsilon: float = 1e-6) -> int | None:
    magnitude = math.hypot(dx, dy)
    if magnitude <= epsilon:
        return None
    angle = math.atan2(dy, dx)
    return int(((angle + math.pi) / (2 * math.pi)) * NUM_BUCKETS) % NUM_BUCKETS


def bucket_label(idx: int) -> str:
    width = 360.0 / NUM_BUCKETS
    start = -180.0 + idx * width
    end = start + width
    return f"[{start:.1f}°, {end:.1f}°)"


def accumulate(dataset: GestureDataset) -> Counter[int]:
    counter: Counter[int] = Counter()
    for seq, _, _ in dataset.samples:
        arr = seq.cpu().numpy()
        dx_total = float(np.sum(arr[:, 0]))
        dy_total = float(np.sum(arr[:, 1]))
        bucket = direction_bucket(dx_total, dy_total)
        if bucket is not None:
            counter[bucket] += 1
    return counter


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Analyze gesture direction histogram")
    parser.add_argument("--dataset", default="bogazici", help="Dataset identifier")
    parser.add_argument("--sequence-length", type=int, default=200, help="Gesture sequence length")
    parser.add_argument("--sampling-rate", default="auto", help="Sampling rate (float or 'auto')")
    parser.add_argument("--max-gestures", type=int, default=4096, help="Maximum gestures to load")
    parser.add_argument("--use-click-boundaries", action="store_true", help="Segment using click-to-click boundaries")
    parser.add_argument("--min-path-length", type=float, default=0.0, help="Minimum path length to keep (pixels)")
    args = parser.parse_args(list(argv) if argv is not None else None)

    cfg = GestureDatasetConfig(
        dataset_id=args.dataset,
        sequence_length=args.sequence_length,
        sampling_rate=args.sampling_rate,
        max_gestures=args.max_gestures,
        use_generated_negatives=False,
        cache_enabled=False,
        normalize_sequences=False,
        normalize_features=False,
        feature_mode="sigma_lognormal",
        use_click_boundaries=args.use_click_boundaries,
        min_path_length=args.min_path_length,
    )
    dataset = GestureDataset(cfg)
    print(f"Loaded {len(dataset)} gestures (effective sampling rate: {dataset._effective_sampling_rate})")

    counts = accumulate(dataset)
    total = sum(counts.values())
    if total == 0:
        print("No gestures with non-zero displacement found.")
        return

    print("Octant histogram:")
    for idx in range(NUM_BUCKETS):
        count = counts.get(idx, 0)
        pct = (count / total) * 100.0
        print(f"  bucket {idx:02d} {bucket_label(idx):>16} : {count:6d} ({pct:5.2f}%)")


if __name__ == "__main__":
    main()
