"""Sigma-Lognormal decomposition based features for mouse trajectories."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np
from scipy.signal import find_peaks

import torch

from data.segmenter import GestureSequence
from .neuromotor import tensor_to_gesture

_EPS = 1e-6


@dataclass
class StrokeParams:
    distance: float
    t0: float
    mu: float
    sigma: float
    theta_start: float
    theta_end: float


def _stroke_boundaries(speeds: np.ndarray, peak_idx: int, drop_ratio: float = 0.15) -> tuple[int, int]:
    peak_val = speeds[peak_idx]
    threshold = max(peak_val * drop_ratio, 1e-6)
    left = peak_idx
    while left > 0 and speeds[left] > threshold:
        left -= 1
    right = peak_idx
    while right < speeds.size - 1 and speeds[right] > threshold:
        right += 1
    return left, right


def _angle(dx: float, dy: float) -> float:
    return float(np.arctan2(dy, dx))


def _compute_stroke_params(
    times: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    speeds: np.ndarray,
    idx_left: int,
    idx_right: int,
) -> StrokeParams:
    dx = np.diff(x[idx_left : idx_right + 1])
    dy = np.diff(y[idx_left : idx_right + 1])
    dt = np.diff(times[idx_left : idx_right + 1])
    distance = float(np.sum(np.sqrt(dx**2 + dy**2)))

    t0 = float(times[idx_left])
    segment_times = times[idx_left : idx_right + 1] - t0 + _EPS
    log_times = np.log(segment_times + _EPS)
    mu = float(np.mean(log_times))
    sigma = float(np.std(log_times))
    theta_start = _angle(dx[0], dy[0]) if dx.size > 0 else 0.0
    theta_end = _angle(dx[-1], dy[-1]) if dx.size > 0 else 0.0
    return StrokeParams(distance=distance, t0=t0, mu=mu, sigma=sigma, theta_start=theta_start, theta_end=theta_end)


def decompose_sigma_lognormal(
    gesture: GestureSequence,
    *,
    max_strokes: int = 32,
    prominence: float = 0.05,
) -> List[StrokeParams]:
    """Compute Sigma-Lognormal stroke parameters from a gesture sequence."""
    sequence = gesture.sequence
    dx = sequence[:, 0]
    dy = sequence[:, 1]
    dt = np.clip(sequence[:, 2], _EPS, None)

    times = np.concatenate([[0.0], np.cumsum(dt)])
    x = np.concatenate([[0.0], np.cumsum(dx)])
    y = np.concatenate([[0.0], np.cumsum(dy)])

    vx = np.divide(dx, dt, out=np.zeros_like(dx), where=dt > _EPS)
    vy = np.divide(dy, dt, out=np.zeros_like(dy), where=dt > _EPS)
    speeds = np.sqrt(vx**2 + vy**2)

    peaks, _ = find_peaks(speeds, prominence=max(prominence * speeds.max(), 1e-6))
    if peaks.size == 0:
        return []

    strokes: List[StrokeParams] = []
    for peak in peaks[:max_strokes]:
        left, right = _stroke_boundaries(speeds, peak)
        params = _compute_stroke_params(times, x, y, speeds, left, right)
        strokes.append(params)
    return strokes


def _half_stats(values: Sequence[float]) -> List[float]:
    if not values:
        return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    midpoint = max(1, len(values) // 2)
    first = np.array(values[:midpoint], dtype=np.float32)
    second = np.array(values[midpoint:], dtype=np.float32)
    if second.size == 0:
        second = first
    stats = [float(np.max(first)), float(np.min(first)), float(np.mean(first))]
    stats.extend([float(np.max(second)), float(np.min(second)), float(np.mean(second))])
    return stats


def sigma_lognormal_features(gesture: GestureSequence) -> np.ndarray:
    """Return the 37-dimensional Sigma-Lognormal feature vector."""
    strokes = decompose_sigma_lognormal(gesture)
    distances = [s.distance for s in strokes]
    t0s = [s.t0 for s in strokes]
    mus = [s.mu for s in strokes]
    sigmas = [s.sigma for s in strokes]
    theta_start = [s.theta_start for s in strokes]
    theta_end = [s.theta_end for s in strokes]

    feature_vec: List[float] = []
    for values in (distances, t0s, mus, sigmas, theta_start, theta_end):
        feature_vec.extend(_half_stats(values))
    feature_vec.append(float(len(strokes)))
    return np.array(feature_vec, dtype=np.float32)


def sigma_lognormal_features_from_sequence(sequence: torch.Tensor) -> torch.Tensor:
    gesture = tensor_to_gesture(sequence)
    features = sigma_lognormal_features(gesture)
    return torch.from_numpy(features)


__all__ = [
    "sigma_lognormal_features",
    "sigma_lognormal_features_from_sequence",
    "decompose_sigma_lognormal",
    "StrokeParams",
]
