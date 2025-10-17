"""Neuromotor feature extraction utilities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import numpy as np
import torch
from scipy.signal import savgol_filter

from data.segmenter import GestureSequence


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    description: str


FEATURE_SPECS: tuple[FeatureSpec, ...] = (
    FeatureSpec("duration", "Gesture duration in seconds."),
    FeatureSpec("path_length", "Cumulative path length (normalised units)."),
    FeatureSpec("avg_velocity", "Average speed over the gesture."),
    FeatureSpec("peak_velocity", "Peak speed magnitude."),
    FeatureSpec("velocity_std", "Standard deviation of speed."),
    FeatureSpec("acc_mean", "Mean acceleration magnitude."),
    FeatureSpec("acc_std", "Standard deviation of acceleration magnitude."),
    FeatureSpec("jerk_mean", "Mean jerk magnitude."),
    FeatureSpec("jerk_std", "Standard deviation of jerk magnitude."),
    FeatureSpec("curvature_mean", "Mean curvature."),
    FeatureSpec("curvature_std", "Standard deviation of curvature."),
    FeatureSpec("path_efficiency", "Euclidean distance / path length."),
    FeatureSpec("direction_changes", "Count of direction changes based on velocity sign."),
    FeatureSpec("time_to_peak_velocity", "Fraction of duration to reach peak velocity."),
    FeatureSpec("idle_ratio", "Fraction of time with velocity below 5% of peak."),
)


def _compute_velocity(dx: np.ndarray, dy: np.ndarray, dt: np.ndarray) -> np.ndarray:
    dt_safe = np.where(dt <= 1e-6, 1e-6, dt)
    vx = dx / dt_safe
    vy = dy / dt_safe
    return np.sqrt(vx**2 + vy**2)


def _finite_difference(values: np.ndarray, dt: np.ndarray) -> np.ndarray:
    dt_safe = np.where(dt <= 1e-6, 1e-6, dt)
    dv = np.diff(values, prepend=values[0])
    return dv / dt_safe


def _smooth(values: np.ndarray, window_length: int = 7, polyorder: int = 3) -> np.ndarray:
    length = values.shape[0]
    if length < window_length:
        window_length = max(3, length // 2 * 2 + 1)
        if window_length < 3:
            return values
    return savgol_filter(values, window_length=window_length, polyorder=polyorder)


def compute_features(gesture: GestureSequence) -> np.ndarray:
    seq = gesture.sequence
    mask = gesture.mask
    valid_len = int(mask.sum())
    if valid_len < 2:
        return np.zeros(len(FEATURE_SPECS), dtype=np.float32)

    dx = seq[:valid_len, 0]
    dy = seq[:valid_len, 1]
    dt = seq[:valid_len, 2]

    velocity = _compute_velocity(dx, dy, dt)
    velocity = _smooth(velocity)

    acc = _smooth(_finite_difference(velocity, dt))
    jerk = _smooth(_finite_difference(acc, dt))

    x_positions = np.cumsum(np.concatenate([[0.0], dx]))
    y_positions = np.cumsum(np.concatenate([[0.0], dy]))
    x_prime = _smooth(np.gradient(x_positions))
    y_prime = _smooth(np.gradient(y_positions))
    x_double = _smooth(np.gradient(x_prime))
    y_double = _smooth(np.gradient(y_prime))
    curvature = np.abs(x_prime * y_double - y_prime * x_double)
    denom = (x_prime**2 + y_prime**2) ** 1.5
    denom = np.where(denom <= 1e-6, 1e-6, denom)
    curvature = curvature / denom

    euclid_dist = np.linalg.norm([x_positions[-1], y_positions[-1]])
    path_length = float(np.sum(np.sqrt(dx**2 + dy**2)))
    path_efficiency = euclid_dist / path_length if path_length > 1e-6 else 0.0

    peak_idx = int(np.argmax(velocity))
    time_to_peak = np.sum(dt[:peak_idx]) / np.sum(dt) if np.sum(dt) > 1e-6 else 0.0

    idle_threshold = velocity.max() * 0.05
    idle_ratio = float(np.mean(velocity < idle_threshold)) if velocity.size else 0.0

    direction_changes = int(np.sum(np.diff(np.sign(dx)) != 0) + np.sum(np.diff(np.sign(dy)) != 0))

    features = np.array(
        [
            gesture.duration,
            gesture.path_length,
            float(np.mean(velocity)),
            float(np.max(velocity)),
            float(np.std(velocity)),
            float(np.mean(acc)),
            float(np.std(acc)),
            float(np.mean(jerk)),
            float(np.std(jerk)),
            float(np.mean(curvature)),
            float(np.std(curvature)),
            path_efficiency,
            float(direction_changes),
            time_to_peak,
            idle_ratio,
        ],
        dtype=np.float32,
    )
    return features


def compute_feature_matrix(gestures: Iterable[GestureSequence]) -> np.ndarray:
    vectors = [compute_features(gesture) for gesture in gestures]
    return np.vstack(vectors) if vectors else np.empty((0, len(FEATURE_SPECS)), dtype=np.float32)


def tensor_to_gesture(sequence: torch.Tensor, dataset_id: str = "generated", split: str = "train") -> GestureSequence:
    seq_np = sequence.detach().cpu().numpy().astype(np.float32)
    mask = np.ones(seq_np.shape[0], dtype=np.float32)
    dx = seq_np[:, 0]
    dy = seq_np[:, 1]
    dt = seq_np[:, 2]
    duration = float(dt.sum())
    path_length = float(np.sum(np.sqrt(dx**2 + dy**2)))
    cumulative_x = np.cumsum(np.concatenate([[0.0], dx]))
    cumulative_y = np.cumsum(np.concatenate([[0.0], dy]))
    metadata = {
        "start_xy": (0.0, 0.0),
        "end_xy": (float(cumulative_x[-1]), float(cumulative_y[-1])),
        "start_timestamp_ms": 0.0,
        "end_timestamp_ms": float(dt.sum() * 1000.0),
    }
    return GestureSequence(
        dataset_id=dataset_id,
        user_id="generator",
        session_id="sample",
        split=split,
        sequence=seq_np,
        mask=mask,
        duration=duration,
        path_length=path_length,
        original_event_count=seq_np.shape[0] + 1,
        metadata=metadata,
    )


def compute_features_from_sequence(sequence: torch.Tensor) -> torch.Tensor:
    gesture = tensor_to_gesture(sequence)
    features = compute_features(gesture)
    return torch.from_numpy(features)
