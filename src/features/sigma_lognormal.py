"""Sigma-Lognormal decomposition based features for mouse trajectories."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import math
import numpy as np
import torch
from scipy.signal import find_peaks

try:
    from numba import njit
    from numba.typed import List as NumbaList

    NUMBA_AVAILABLE = True
except Exception:  # pragma: no cover - optional accelerator
    NUMBA_AVAILABLE = False
    njit = None
    NumbaList = None

from data.segmenter import GestureSequence

_EPS = 1e-6
_DEFAULT_MAX_STROKES = 32
_DEFAULT_PROMINENCE = 0.05
_DEFAULT_DROP_RATIO = 0.15


if NUMBA_AVAILABLE:

    @njit(cache=True)
    def _stats_list_numba(strokes, attr_idx):
        count = len(strokes)
        if count == 0:
            return 0.0, 0.0, 0.0
        max_val = strokes[0][attr_idx]
        min_val = strokes[0][attr_idx]
        total = strokes[0][attr_idx]
        for idx in range(1, count):
            val = strokes[idx][attr_idx]
            if val > max_val:
                max_val = val
            if val < min_val:
                min_val = val
            total += val
        mean_val = total / count
        return max_val, min_val, mean_val

    @njit(cache=True)
    def _peak_prominence_numba(speeds, peak_idx):
        peak_height = speeds[peak_idx]
        left_min = peak_height
        i = peak_idx
        while i > 0:
            i -= 1
            val = speeds[i]
            if val > peak_height:
                break
            if val < left_min:
                left_min = val

        right_min = peak_height
        i = peak_idx
        last = speeds.shape[0] - 1
        while i < last:
            i += 1
            val = speeds[i]
            if val > peak_height:
                break
            if val < right_min:
                right_min = val
        if left_min > peak_height:
            left_min = peak_height
        if right_min > peak_height:
            right_min = peak_height
        ref = left_min
        if right_min > ref:
            ref = right_min
        return peak_height - ref

    @njit(cache=True)
    def _sigma_lognormal_features_numba(sequence, max_strokes, prominence, drop_ratio, duration_override):
        length = sequence.shape[0]
        dx = sequence[:, 0]
        dy = sequence[:, 1]
        dt = sequence[:, 2]

        times = np.empty(length + 1, dtype=np.float32)
        times[0] = 0.0
        for i in range(length):
            dt_val = dt[i]
            if dt_val < _EPS:
                dt_val = _EPS
            times[i + 1] = times[i] + dt_val

        x = np.empty(length + 1, dtype=np.float32)
        y = np.empty(length + 1, dtype=np.float32)
        x[0] = 0.0
        y[0] = 0.0
        for i in range(length):
            x[i + 1] = x[i] + dx[i]
            y[i + 1] = y[i] + dy[i]

        speeds = np.empty(length, dtype=np.float32)
        for i in range(length):
            vx = 0.0
            vy = 0.0
            if dt[i] > _EPS:
                vx = dx[i] / dt[i]
                vy = dy[i] / dt[i]
            speeds[i] = math.sqrt(vx * vx + vy * vy)

        max_speed = 0.0
        for i in range(length):
            spd = speeds[i]
            if spd > max_speed:
                max_speed = spd
        threshold = prominence * max_speed
        if threshold < _EPS:
            threshold = _EPS

        peaks = NumbaList()
        if length >= 3:
            for idx in range(1, length - 1):
                center = speeds[idx]
                if center < threshold:
                    continue
                if center > speeds[idx - 1] and center > speeds[idx + 1]:
                    prominence_val = _peak_prominence_numba(speeds, idx)
                    if prominence_val >= threshold:
                        peaks.append(idx)

        strokes = NumbaList()
        peak_count = len(peaks)
        if peak_count > max_strokes:
            peak_count = max_strokes

        for peak_idx_pos in range(peak_count):
            peak_idx = peaks[peak_idx_pos]
            peak_val = speeds[peak_idx]
            drop_threshold = drop_ratio * peak_val
            if drop_threshold < _EPS:
                drop_threshold = _EPS

            left = peak_idx
            while left > 0 and speeds[left] > drop_threshold:
                left -= 1

            right = peak_idx
            max_idx = length - 1
            while right < max_idx and speeds[right] > drop_threshold:
                right += 1

            distance = 0.0
            for idx in range(left, right):
                dx_seg = x[idx + 1] - x[idx]
                dy_seg = y[idx + 1] - y[idx]
                distance += math.sqrt(dx_seg * dx_seg + dy_seg * dy_seg)

            t0 = times[left]
            segment_count = right - left + 1
            log_times = np.empty(segment_count, dtype=np.float32)
            for idx in range(segment_count):
                tau = times[left + idx] - t0
                if tau <= _EPS:
                    tau = _EPS
                log_times[idx] = math.log(tau + _EPS)

            sum_log = 0.0
            sum_sq = 0.0
            for idx in range(segment_count):
                val = log_times[idx]
                sum_log += val
                sum_sq += val * val
            if segment_count > 0:
                mu = sum_log / segment_count
                var = sum_sq / segment_count - mu * mu
                if var < 0.0:
                    var = 0.0
                sigma = math.sqrt(var)
            else:
                mu = 0.0
                sigma = 0.0

            theta_start = 0.0
            theta_end = 0.0
            if right > left:
                dx0 = x[left + 1] - x[left]
                dy0 = y[left + 1] - y[left]
                theta_start = math.atan2(dy0, dx0)
                dx1 = x[right] - x[right - 1]
                dy1 = y[right] - y[right - 1]
                theta_end = math.atan2(dy1, dx1)

            stroke = np.empty(6, dtype=np.float32)
            stroke[0] = distance
            stroke[1] = t0
            stroke[2] = mu
            stroke[3] = sigma
            stroke[4] = theta_start
            stroke[5] = theta_end
            strokes.append(stroke)

        total_duration = 0.0
        for idx in range(length):
            total_duration += dt[idx]
        if duration_override > 0.0:
            total_duration = duration_override
        if total_duration < _EPS:
            total_duration = _EPS
        half_time = 0.5 * total_duration

        first_half = NumbaList()
        second_half = NumbaList()
        for idx in range(len(strokes)):
            stroke = strokes[idx]
            if stroke[1] <= half_time:
                first_half.append(stroke)
            else:
                second_half.append(stroke)

        feature_vec = np.zeros(37, dtype=np.float32)
        pos = 0
        for attr_idx in range(6):
            fmax, fmin, fmean = _stats_list_numba(first_half, attr_idx)
            smax, smin, smean = _stats_list_numba(second_half, attr_idx)
            feature_vec[pos] = fmax
            feature_vec[pos + 1] = fmin
            feature_vec[pos + 2] = fmean
            feature_vec[pos + 3] = smax
            feature_vec[pos + 4] = smin
            feature_vec[pos + 5] = smean
            pos += 6

        feature_vec[-1] = float(len(strokes))
        return feature_vec

else:  # pragma: no cover - fallback placeholder when numba missing

    def _sigma_lognormal_features_numba(*_args, **_kwargs):
        raise RuntimeError("Numba is required for accelerated sigma-lognormal features.")


@dataclass
class StrokeParams:
    distance: float
    t0: float
    mu: float
    sigma: float
    theta_start: float
    theta_end: float


# ---------------------------------------------------------------------------
# Torch implementation
# ---------------------------------------------------------------------------

def _peak_prominence(speeds: torch.Tensor, peak_idx: int) -> float:
    peak_height = float(speeds[peak_idx].item())
    left_min = peak_height
    i = peak_idx
    while i > 0:
        i -= 1
        val = float(speeds[i].item())
        if val > peak_height:
            break
        if val < left_min:
            left_min = val

    right_min = peak_height
    i = peak_idx
    last = speeds.size(0) - 1
    while i < last:
        i += 1
        val = float(speeds[i].item())
        if val > peak_height:
            break
        if val < right_min:
            right_min = val

    return peak_height - max(left_min, right_min)


def _find_local_peaks(speeds: torch.Tensor, threshold: float) -> torch.Tensor:
    if speeds.numel() < 3:
        return torch.empty(0, dtype=torch.long, device=speeds.device)
    candidates = []
    for idx in range(1, speeds.size(0) - 1):
        center = float(speeds[idx].item())
        if center < threshold:
            continue
        if center > float(speeds[idx - 1].item()) and center > float(speeds[idx + 1].item()):
            candidates.append(idx)

    if not candidates:
        return torch.empty(0, dtype=torch.long, device=speeds.device)

    filtered = [idx for idx in candidates if _peak_prominence(speeds, idx) >= threshold]
    if not filtered:
        return torch.empty(0, dtype=torch.long, device=speeds.device)
    return torch.tensor(filtered, dtype=torch.long, device=speeds.device)


def _stroke_boundaries_torch(speeds: torch.Tensor, peak_idx: int, drop_ratio: float) -> tuple[int, int]:
    peak_val = float(speeds[peak_idx].item())
    threshold = max(peak_val * drop_ratio, _EPS)
    left = peak_idx
    while left > 0 and float(speeds[left].item()) > threshold:
        left -= 1
    right = peak_idx
    max_idx = speeds.size(0) - 1
    while right < max_idx and float(speeds[right].item()) > threshold:
        right += 1
    return left, right


def _compute_stroke_params_torch(
    times: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
    idx_left: int,
    idx_right: int,
) -> StrokeParams:
    if idx_right <= idx_left:
        return StrokeParams(
            distance=0.0,
            t0=float(times[idx_left].item()),
            mu=0.0,
            sigma=0.0,
            theta_start=0.0,
            theta_end=0.0,
        )

    dx = x[idx_left + 1 : idx_right + 1] - x[idx_left:idx_right]
    dy = y[idx_left + 1 : idx_right + 1] - y[idx_left:idx_right]

    distance = torch.sqrt(dx**2 + dy**2).sum().item()
    t0 = float(times[idx_left].item())
    segment_times = times[idx_left : idx_right + 1] - t0 + _EPS
    log_times = torch.log(segment_times + _EPS)
    mu = float(log_times.mean().item())
    sigma = float(log_times.std(unbiased=False).item())

    if dx.numel() > 0:
        theta_start = float(torch.atan2(dy[0], dx[0]).item())
        theta_end = float(torch.atan2(dy[-1], dx[-1]).item())
    else:
        theta_start = 0.0
        theta_end = 0.0

    return StrokeParams(
        distance=float(distance),
        t0=t0,
        mu=mu,
        sigma=sigma,
        theta_start=theta_start,
        theta_end=theta_end,
    )


def _split_strokes_by_time(strokes: Sequence[StrokeParams], half_time: float) -> tuple[list[StrokeParams], list[StrokeParams]]:
    if not strokes:
        return [], []
    first: list[StrokeParams] = []
    second: list[StrokeParams] = []
    for stroke in strokes:
        target = first if stroke.t0 <= half_time else second
        target.append(stroke)
    if not first and second:
        first = list(second)
    if not second and first:
        second = list(first)
    return first, second


def _stats(values: Sequence[float]) -> List[float]:
    if not values:
        return [0.0, 0.0, 0.0]
    arr = np.asarray(values, dtype=np.float32)
    return [float(arr.max()), float(arr.min()), float(arr.mean())]


def decompose_sigma_lognormal_torch(
    sequence: torch.Tensor,
    *,
    max_strokes: int = _DEFAULT_MAX_STROKES,
    prominence: float = _DEFAULT_PROMINENCE,
    drop_ratio: float = _DEFAULT_DROP_RATIO,
) -> List[StrokeParams]:
    seq = sequence.detach().to(dtype=torch.float32, device=torch.device("cpu"))
    dx = seq[..., 0]
    dy = seq[..., 1]
    dt = seq[..., 2].clamp_min(_EPS)

    times = torch.cat(
        [torch.zeros(1, dtype=seq.dtype, device=seq.device), torch.cumsum(dt, dim=0)]
    )
    x = torch.cat([torch.zeros(1, dtype=seq.dtype, device=seq.device), torch.cumsum(dx, dim=0)])
    y = torch.cat([torch.zeros(1, dtype=seq.dtype, device=seq.device), torch.cumsum(dy, dim=0)])

    vx = torch.where(dt > 0, dx / dt, torch.zeros_like(dx))
    vy = torch.where(dt > 0, dy / dt, torch.zeros_like(dy))
    speeds = torch.sqrt(vx**2 + vy**2)

    max_speed = float(speeds.max().item()) if speeds.numel() > 0 else 0.0
    threshold = max(prominence * max_speed, _EPS)
    peaks = _find_local_peaks(speeds, threshold)
    if peaks.numel() == 0:
        return []

    strokes: List[StrokeParams] = []
    for peak_idx in peaks.tolist()[:max_strokes]:
        left, right = _stroke_boundaries_torch(speeds, peak_idx, drop_ratio)
        strokes.append(_compute_stroke_params_torch(times, x, y, left, right))
    return strokes


def sigma_lognormal_features_torch(
    sequence: torch.Tensor,
    *,
    max_strokes: int = _DEFAULT_MAX_STROKES,
    prominence: float = _DEFAULT_PROMINENCE,
    drop_ratio: float = _DEFAULT_DROP_RATIO,
    duration: Optional[float] = None,
) -> torch.Tensor:
    original_device = sequence.device
    single = sequence.dim() == 2
    if single:
        sequences = sequence.unsqueeze(0)
    else:
        sequences = sequence

    outputs: list[torch.Tensor] = []
    for seq in sequences:
        if NUMBA_AVAILABLE:
            seq_np = seq.detach().cpu().numpy()
            duration_override = float(duration) if duration is not None else -1.0
            feature_np = _sigma_lognormal_features_numba(
                seq_np,
                max_strokes,
                prominence,
                drop_ratio,
                duration_override,
            )
            outputs.append(torch.from_numpy(feature_np))
            continue

        strokes = decompose_sigma_lognormal_torch(
            seq,
            max_strokes=max_strokes,
            prominence=prominence,
            drop_ratio=drop_ratio,
        )
        dt_sum = float(seq[..., 2].clamp_min(_EPS).sum().item())
        total_duration = max(duration if duration is not None else dt_sum, _EPS)
        half_time = 0.5 * total_duration
        first_half, second_half = _split_strokes_by_time(strokes, half_time)

        feature_vec: List[float] = []
        for attr in ("distance", "t0", "mu", "sigma", "theta_start", "theta_end"):
            first_vals = [getattr(stroke, attr) for stroke in first_half]
            second_vals = [getattr(stroke, attr) for stroke in second_half]
            feature_vec.extend(_stats(first_vals))
            feature_vec.extend(_stats(second_vals))
        feature_vec.append(float(len(strokes)))

        outputs.append(torch.tensor(feature_vec, dtype=torch.float32))

    result = torch.stack(outputs, dim=0).to(original_device)
    if single:
        return result[0]
    return result


# ---------------------------------------------------------------------------
# Legacy NumPy/SciPy implementation (used for validation)
# ---------------------------------------------------------------------------

def _stroke_boundaries_numpy(speeds: np.ndarray, peak_idx: int, drop_ratio: float = _DEFAULT_DROP_RATIO) -> tuple[int, int]:
    peak_val = speeds[peak_idx]
    threshold = max(peak_val * drop_ratio, _EPS)
    left = peak_idx
    while left > 0 and speeds[left] > threshold:
        left -= 1
    right = peak_idx
    while right < speeds.size - 1 and speeds[right] > threshold:
        right += 1
    return left, right


def _compute_stroke_params_numpy(
    times: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    idx_left: int,
    idx_right: int,
) -> StrokeParams:
    if idx_right <= idx_left:
        return StrokeParams(0.0, float(times[idx_left]), 0.0, 0.0, 0.0, 0.0)

    dx = np.diff(x[idx_left : idx_right + 1])
    dy = np.diff(y[idx_left : idx_right + 1])
    dt = np.diff(times[idx_left : idx_right + 1])
    distance = float(np.sum(np.sqrt(dx**2 + dy**2)))

    t0 = float(times[idx_left])
    segment_times = times[idx_left : idx_right + 1] - t0 + _EPS
    log_times = np.log(segment_times + _EPS)
    mu = float(np.mean(log_times))
    sigma = float(np.std(log_times))
    theta_start = float(np.arctan2(dy[0], dx[0])) if dx.size > 0 else 0.0
    theta_end = float(np.arctan2(dy[-1], dx[-1])) if dx.size > 0 else 0.0
    return StrokeParams(distance=distance, t0=t0, mu=mu, sigma=sigma, theta_start=theta_start, theta_end=theta_end)


def decompose_sigma_lognormal_numpy(
    gesture: GestureSequence,
    *,
    max_strokes: int = _DEFAULT_MAX_STROKES,
    prominence: float = _DEFAULT_PROMINENCE,
) -> List[StrokeParams]:
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
        left, right = _stroke_boundaries_numpy(speeds, peak)
        params = _compute_stroke_params_numpy(times, x, y, left, right)
        strokes.append(params)
    return strokes


def sigma_lognormal_features_numpy(gesture: GestureSequence) -> np.ndarray:
    strokes = decompose_sigma_lognormal_numpy(gesture)
    duration = max(float(gesture.duration), _EPS)
    half_time = 0.5 * duration
    first_half, second_half = _split_strokes_by_time(strokes, half_time)

    feature_vec: List[float] = []
    for attr in ("distance", "t0", "mu", "sigma", "theta_start", "theta_end"):
        first_values = [getattr(stroke, attr) for stroke in first_half]
        second_values = [getattr(stroke, attr) for stroke in second_half]
        feature_vec.extend(_stats(first_values))
        feature_vec.extend(_stats(second_values))

    feature_vec.append(float(len(strokes)))
    return np.array(feature_vec, dtype=np.float32)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def decompose_sigma_lognormal(
    gesture: GestureSequence,
    *,
    max_strokes: int = _DEFAULT_MAX_STROKES,
    prominence: float = _DEFAULT_PROMINENCE,
    drop_ratio: float = _DEFAULT_DROP_RATIO,
) -> List[StrokeParams]:
    seq_tensor = torch.from_numpy(gesture.sequence.astype(np.float32, copy=False))
    return decompose_sigma_lognormal_torch(
        seq_tensor,
        max_strokes=max_strokes,
        prominence=prominence,
        drop_ratio=drop_ratio,
    )


def sigma_lognormal_features(gesture: GestureSequence) -> np.ndarray:
    seq_tensor = torch.from_numpy(gesture.sequence.astype(np.float32, copy=False))
    features = sigma_lognormal_features_torch(
        seq_tensor,
        duration=float(gesture.duration),
    )
    return features.cpu().numpy()


def sigma_lognormal_features_from_sequence(sequence: torch.Tensor) -> torch.Tensor:
    features = sigma_lognormal_features_torch(sequence)
    return features


__all__ = [
    "sigma_lognormal_features",
    "sigma_lognormal_features_from_sequence",
    "sigma_lognormal_features_torch",
    "decompose_sigma_lognormal",
    "decompose_sigma_lognormal_torch",
    "StrokeParams",
]
