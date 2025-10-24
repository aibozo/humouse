#!/usr/bin/env python3
"""CLI entry point for high-resolution local mouse data capture."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from data.local_mouse_collector import MouseGestureCollector  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect high-resolution local mouse gestures.")
    parser.add_argument("--user-id", type=str, default="local_user", help="Identifier for the captured user.")
    parser.add_argument("--dataset-id", type=str, default="local_mouse", help="Dataset identifier to embed in manifests.")
    parser.add_argument("--split", type=str, default="train", help="Dataset split tag (e.g., train/val/test).")
    parser.add_argument("--sampling-hz", type=float, default=200.0, help="Polling frequency for pointer position.")
    parser.add_argument("--gesture-idle", type=float, default=2.0, help="Seconds of inactivity to close a gesture.")
    parser.add_argument("--session-idle", type=float, default=300.0, help="Seconds of inactivity to close a session.")
    parser.add_argument("--sequence-length", type=int, default=64, help="Resampled sequence length for stored gestures.")
    parser.add_argument("--jitter-threshold", type=float, default=8.0, help="Path-length threshold (pixels) to flag jitter gestures.")
    parser.add_argument("--stationary-threshold", type=float, default=1.0, help="Distance threshold to treat a gesture as stationary.")
    parser.add_argument("--min-gesture-events", type=int, default=5, help="Minimum raw events required before persisting a gesture.")
    parser.add_argument("--direction-buckets", type=int, default=8, help="Number of directional buckets (octants).")
    parser.add_argument(
        "--no-stationary-bucket",
        action="store_true",
        help="Disable creation of the stationary (+1) bucket.",
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path("data/raw/local_mouse"),
        help="Directory to store raw session CSV files.",
    )
    parser.add_argument(
        "--processed-root",
        type=Path,
        default=Path("data/processed/local_mouse"),
        help="Directory to store processed gesture NPZ artefacts.",
    )
    parser.add_argument(
        "--no-processed",
        action="store_true",
        help="Disable on-the-fly processed gesture export (raw CSV only).",
    )
    parser.add_argument(
        "--no-features",
        action="store_true",
        help="Skip neuromotor feature computation when persisting gestures.",
    )
    parser.add_argument(
        "--skip-stationary-duplicates",
        action="store_true",
        help="Drop repeated samples with no movement within the polling interval.",
    )
    parser.add_argument(
        "--no-click-split",
        action="store_true",
        help="Disable click-triggered gesture finalisation (idle-only segmentation).",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()))

    collector = MouseGestureCollector(
        user_id=args.user_id,
        dataset_id=args.dataset_id,
        split=args.split,
        sampling_hz=args.sampling_hz,
        gesture_idle_seconds=args.gesture_idle,
        session_idle_seconds=args.session_idle,
        sequence_length=args.sequence_length,
        jitter_path_threshold=args.jitter_threshold,
        stationary_threshold=args.stationary_threshold,
        min_gesture_events=args.min_gesture_events,
        raw_root=args.raw_root,
        processed_root=args.processed_root,
        store_processed=not args.no_processed,
        compute_features=not args.no_features,
        skip_stationary_duplicates=args.skip_stationary_duplicates,
        direction_buckets=args.direction_buckets,
        include_stationary_bucket=not args.no_stationary_bucket,
        split_on_click=not args.no_click_split,
    )
    collector.run()


if __name__ == "__main__":
    main()
