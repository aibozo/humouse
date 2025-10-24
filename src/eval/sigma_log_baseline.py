"""Evaluate Sigma-Lognormal classifier baselines against synthetic trajectories."""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from data.dataset import GestureDataset, GestureDatasetConfig
from features import sigma_lognormal_features_from_sequence, tensor_to_gesture, sigma_lognormal_features_torch
from features.sigma_lognormal import decompose_sigma_lognormal, StrokeParams

_BASELINE_CACHE: Dict[tuple, tuple["RealGestureStats", np.ndarray]] = {}


@dataclass
class RealGestureStats:
    lengths: np.ndarray
    durations: np.ndarray
    displacements: np.ndarray
    sequences: List[np.ndarray]
    templates: List[List[StrokeParams]]
    template_groups: dict[int, List[List[StrokeParams]]] = field(default_factory=dict)


class FunctionBasedTrajectoryGenerator:
    """Replicates BeCAPTCHA function-driven synthetic trajectories."""

    def __init__(
        self,
        stats: RealGestureStats,
        sequence_length: int | None = 64,
        seed: int | None = None,
        sampling_rate: float | None = 200.0,
    ):
        self.stats = stats
        self.sequence_length = sequence_length if sequence_length and sequence_length > 0 else None
        self.sampling_rate = sampling_rate if sampling_rate is not None else 200.0
        self.rng = np.random.default_rng(seed)

    def _sample_length(self) -> float:
        return float(self.rng.choice(self.stats.lengths))

    def _sample_duration(self) -> float:
        return float(self.rng.choice(self.stats.durations))

    def _sample_displacement(self) -> np.ndarray:
        return self.stats.displacements[self.rng.integers(0, len(self.stats.displacements))]

    def _shape_points(self, u: np.ndarray, shape: str) -> np.ndarray:
        if shape == "linear":
            x = u
            y = np.zeros_like(u)
        elif shape == "quadratic":
            a = self.rng.uniform(-1.0, 1.0)
            x = u
            y = a * (u - 0.5) ** 2
        elif shape == "exponential":
            k = self.rng.uniform(1.0, 3.0)
            x = u
            y = np.exp(k * u) - 1.0
        else:
            raise ValueError(f"Unsupported shape: {shape}")
        points = np.stack([x, y], axis=1)
        points -= points[0]
        return points

    def _velocity_weights(self, profile: str, n: int) -> np.ndarray:
        if profile == "constant":
            weights = np.ones(n, dtype=np.float32)
        elif profile == "logarithmic":
            weights = np.linspace(0.2, 1.0, n, dtype=np.float32)
        elif profile == "gaussian":
            positions = np.linspace(-2.0, 2.0, n, dtype=np.float32)
            weights = np.exp(-0.5 * positions**2) + 0.1
        else:
            raise ValueError(f"Unsupported velocity profile: {profile}")
        return weights.astype(np.float32)

    def generate_sequence(self, shape: str, velocity_profile: str) -> np.ndarray:
        duration = max(self._sample_duration(), 1e-3)
        if self.sequence_length is not None:
            n = self.sequence_length
        else:
            n = max(16, int(round(duration * self.sampling_rate)))
        weights = self._velocity_weights(velocity_profile, n)
        cumulative = np.concatenate([[0.0], np.cumsum(weights)])
        cumulative /= cumulative[-1]

        points = self._shape_points(cumulative, shape)
        target_length = max(self._sample_length(), 1e-3)
        diffs = np.diff(points, axis=0)
        current_len = np.sum(np.linalg.norm(diffs, axis=1))
        scale = target_length / max(current_len, 1e-6)
        points *= scale

        displacement = self._sample_displacement()
        vec_norm = np.linalg.norm(points[-1]) + 1e-9
        target_vec = displacement
        tgt_norm = np.linalg.norm(target_vec) + 1e-9
        if vec_norm > 0 and tgt_norm > 0:
            base_dir = points[-1] / vec_norm
            target_dir = target_vec / tgt_norm
            angle = math.atan2(target_dir[1], target_dir[0]) - math.atan2(base_dir[1], base_dir[0])
            rot = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
            points = points @ rot.T
            points *= tgt_norm / vec_norm

        if self.rng.random() < 0.6:
            base_len = n
            tail_start = max(1, int(base_len * self.rng.uniform(0.6, 0.85)))
            tail_count = points.shape[0] - (tail_start + 1)
            if tail_count > 0:
                correction = self.rng.normal(scale=target_length * 0.05, size=(tail_count, 2))
                points[tail_start + 1 :] += correction

        dx_dy = np.diff(points, axis=0)
        dt = weights / weights.sum() * duration
        sequence = np.stack([dx_dy[:, 0], dx_dy[:, 1], dt], axis=1)
        return sequence.astype(np.float32)

    def generate_batch(self, num_samples: int, shape: str, velocity_profile: str) -> List[np.ndarray]:
        return [self.generate_sequence(shape, velocity_profile) for _ in range(num_samples)]


class SigmaLognormalTrajectoryGenerator:
    """Generates trajectories by sampling Sigma-Lognormal stroke parameters."""

    def __init__(
        self,
        stats: RealGestureStats,
        sequence_length: int | None = 64,
        seed: int | None = None,
        sampling_rate: float | None = 200.0,
    ):
        self.stats = stats
        self.sequence_length = sequence_length if sequence_length and sequence_length > 0 else None
        self.sampling_rate = sampling_rate if sampling_rate is not None else 200.0
        self.rng = np.random.default_rng(seed)
        self.templates = [tpl for tpl in stats.templates if tpl]
        if not self.templates:
            raise RuntimeError("No Sigma-Lognormal templates available for synthesis.")

    def _jitter_stroke(self, stroke: StrokeParams) -> StrokeParams:
        return StrokeParams(
            distance=stroke.distance * float(self.rng.uniform(0.85, 1.15)),
            t0=max(1e-3, stroke.t0 + float(self.rng.normal(0.0, max(0.01, 0.05 * stroke.t0)))),
            mu=stroke.mu + float(self.rng.normal(0.0, 0.1)),
            sigma=max(0.05, stroke.sigma + float(self.rng.normal(0.0, 0.05))),
            theta_start=stroke.theta_start + float(self.rng.normal(0.0, 0.2)),
            theta_end=stroke.theta_end + float(self.rng.normal(0.0, 0.2)),
        )

    def _stroke_velocity(self, stroke: StrokeParams, t: float) -> float:
        tau = t - stroke.t0
        if tau <= 1e-6:
            return 0.0
        sigma = max(0.05, stroke.sigma)
        mu = stroke.mu
        distance = max(1e-6, stroke.distance)
        coeff = distance / (math.sqrt(2 * math.pi) * sigma * tau)
        exponent = -0.5 * ((math.log(tau) - mu) / sigma) ** 2
        return coeff * math.exp(exponent)

    def _stroke_angle(self, stroke: StrokeParams, tau: float) -> float:
        sigma = max(0.05, stroke.sigma)
        logistics = 1.0 / (1.0 + math.exp(-(math.log(max(tau, 1e-6)) - stroke.mu) / (3.0 * sigma)))
        return stroke.theta_start + (stroke.theta_end - stroke.theta_start) * logistics

    def generate_sequence(self) -> np.ndarray:
        disp = self.stats.displacements[self.rng.integers(0, len(self.stats.displacements))]
        target_length = float(self.rng.choice(self.stats.lengths)) * float(self.rng.uniform(0.9, 1.1))

        angle = math.atan2(disp[1], disp[0])
        bin_idx = int(((angle + math.pi) / (2 * math.pi)) * 8) % 8
        candidate_group = self.stats.template_groups.get(bin_idx, [])
        if candidate_group:
            template = candidate_group[self.rng.integers(0, len(candidate_group))]
        else:
            template = self.templates[self.rng.integers(0, len(self.templates))]
        strokes = [self._jitter_stroke(stroke) for stroke in template]
        max_time = max(stroke.t0 + math.exp(stroke.mu + 2.0 * stroke.sigma) for stroke in strokes)
        max_time = max_time * float(self.rng.uniform(0.9, 1.2))
        duration = float(self.rng.choice(self.stats.durations)) * float(self.rng.uniform(0.9, 1.1))
        if self.sequence_length is not None:
            n = self.sequence_length
        else:
            n = max(32, int(round(duration * self.sampling_rate)))

        times = np.linspace(1e-3, max_time, n + 1, dtype=np.float32)

        points = np.zeros((n + 1, 2), dtype=np.float32)
        for idx in range(n):
            t0 = float(times[idx])
            t1 = float(times[idx + 1])
            dt = t1 - t0
            if dt <= 0:
                continue
            t_mid = 0.5 * (t0 + t1)
            vx = 0.0
            vy = 0.0
            for stroke in strokes:
                tau = t_mid - stroke.t0
                if tau <= 1e-6:
                    continue
                speed = self._stroke_velocity(stroke, t_mid)
                if speed <= 0:
                    continue
                angle = self._stroke_angle(stroke, tau)
                vx += speed * math.cos(angle)
                vy += speed * math.sin(angle)
            points[idx + 1, 0] = points[idx, 0] + vx * dt
            points[idx + 1, 1] = points[idx, 1] + vy * dt

        dx = np.diff(points[:, 0])
        dy = np.diff(points[:, 1])
        traj_length = np.sum(np.sqrt(dx**2 + dy**2)) + 1e-9
        scale = target_length / traj_length
        dx *= scale
        dy *= scale

        if np.linalg.norm(disp) > 1e-6:
            base_end = np.array([np.sum(dx), np.sum(dy)])
            base_norm = np.linalg.norm(base_end) + 1e-9
            angle = math.atan2(disp[1], disp[0]) - math.atan2(base_end[1], base_end[0])
            rot = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
            stacked = np.stack([dx, dy], axis=1) @ rot.T
            dx = stacked[:, 0]
            dy = stacked[:, 1]

        duration = float(self.rng.choice(self.stats.durations)) * float(self.rng.uniform(0.9, 1.1))
        dt = np.diff(times)
        dt = np.clip(dt, 1e-4, None)
        dt = dt / dt.sum() * duration

        noise_scale = target_length * 0.015
        dx += self.rng.normal(scale=noise_scale, size=dx.shape)
        dy += self.rng.normal(scale=noise_scale, size=dy.shape)

        sequence = np.stack([dx.astype(np.float32), dy.astype(np.float32), dt.astype(np.float32)], axis=1)
        return sequence

    def generate_batch(self, num_samples: int) -> List[np.ndarray]:
        samples: List[np.ndarray] = []
        for _ in range(num_samples):
            try:
                samples.append(self.generate_sequence())
            except Exception:
                continue
        if not samples:
            raise RuntimeError("SigmaLognormalTrajectoryGenerator failed to generate samples")
        return samples


def _collect_real_stats(dataset: GestureDataset) -> RealGestureStats:
    lengths: List[float] = []
    durations: List[float] = []
    displacements: List[np.ndarray] = []
    sequences: List[np.ndarray] = []
    templates: List[List[StrokeParams]] = []
    template_groups: dict[int, List[List[StrokeParams]]] = {i: [] for i in range(8)}

    for sequence, _, label in dataset.samples:
        if label.item() != 1.0:
            continue
        seq_np = sequence.cpu().numpy()
        dx = seq_np[:, 0]
        dy = seq_np[:, 1]
        dt = seq_np[:, 2]
        lengths.append(float(np.sum(np.sqrt(dx**2 + dy**2))))
        durations.append(float(np.sum(dt)))
        disp_vec = np.array([np.sum(dx), np.sum(dy)], dtype=np.float32)
        displacements.append(disp_vec)
        sequences.append(seq_np)
        gesture = tensor_to_gesture(sequence)
        strokes = decompose_sigma_lognormal(gesture)
        if strokes:
            templates.append(strokes)
            angle = math.atan2(disp_vec[1], disp_vec[0])
            bin_idx = int(((angle + math.pi) / (2 * math.pi)) * 8) % 8
            template_groups.setdefault(bin_idx, []).append(strokes)

    return RealGestureStats(
        lengths=np.array(lengths),
        durations=np.array(durations),
        displacements=np.stack(displacements),
        sequences=sequences,
        templates=templates,
        template_groups=template_groups,
    )


def _features_from_sequences(sequences: Iterable[np.ndarray]) -> np.ndarray:
    seq_list = list(sequences)
    if not seq_list:
        return np.empty((0, 37), dtype=np.float32)
    tensor = torch.from_numpy(np.stack(seq_list)).to(dtype=torch.float32)
    features = sigma_lognormal_features_torch(tensor).cpu().numpy()
    return features


def _train_classifier(X_real: np.ndarray, X_fake: np.ndarray, seed: int = 1337) -> Tuple[float, dict]:
    X = np.concatenate([X_real, X_fake])
    y = np.concatenate([np.ones(len(X_real)), np.zeros(len(X_fake))])
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=seed,
        stratify=y,
    )
    clf = RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=seed)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    report = classification_report(y_test, preds, output_dict=False)
    return acc, {"report": report}


def run_baseline(
    dataset_id: str,
    sequence_length: int,
    max_gestures: int,
    seed: int,
    samples_per_case: int,
    generator_type: str = "function",
    gan_dir: Path | None = None,
    output: Path | None = None,
    canonicalize_path: bool = True,
    canonicalize_duration: bool = True,
    sampling_rate: float | None = 200.0,
    *,
    make_plots: bool = True,
) -> dict:
    cache_key = (
        dataset_id,
        sequence_length,
        max_gestures,
        canonicalize_path,
        canonicalize_duration,
        sampling_rate,
    )

    cached = _BASELINE_CACHE.get(cache_key)
    if cached is None:
        dataset_cfg = GestureDatasetConfig(
            dataset_id=dataset_id,
            sequence_length=sequence_length,
            max_gestures=max_gestures,
            use_generated_negatives=False,
            cache_enabled=False,
            normalize_sequences=False,
            normalize_features=False,
            feature_mode="sigma_lognormal",
            canonicalize_path=canonicalize_path,
            canonicalize_duration=canonicalize_duration,
            sampling_rate=sampling_rate,
            feature_reservoir_size=None,
        )
        dataset = GestureDataset(dataset_cfg)
        stats = _collect_real_stats(dataset)
        real_features = dataset.get_positive_features_tensor(use_full=True).numpy()
        _BASELINE_CACHE[cache_key] = (stats, real_features)
    else:
        stats, real_features = cached

    results = []
    sampling_rate = sampling_rate if sampling_rate is not None else 200.0

    if generator_type == "function":
        generator = FunctionBasedTrajectoryGenerator(
            stats,
            sequence_length=sequence_length,
            seed=seed,
            sampling_rate=sampling_rate,
        )
        shapes = ["linear", "quadratic", "exponential"]
        velocities = ["constant", "logarithmic", "gaussian"]
        for shape in shapes:
            for velocity in velocities:
                synthetic_sequences = generator.generate_batch(samples_per_case, shape, velocity)
                synthetic_features = _features_from_sequences(synthetic_sequences)
                acc, extras = _train_classifier(real_features, synthetic_features, seed=seed)
                results.append(
                    {
                        "shape": shape,
                        "velocity": velocity,
                        "accuracy": acc,
                        "error_rate": 1.0 - acc,
                        "details": extras,
                    }
                )
    elif generator_type == "sigma":
        generator = SigmaLognormalTrajectoryGenerator(
            stats,
            sequence_length=sequence_length,
            seed=seed,
            sampling_rate=sampling_rate,
        )
        replicates = 5
        for idx in range(replicates):
            synthetic_sequences = generator.generate_batch(samples_per_case)
            synthetic_features = _features_from_sequences(synthetic_sequences)
            acc, extras = _train_classifier(real_features, synthetic_features, seed=seed + idx)
            results.append(
                {
                    "shape": "sigma_lognormal",
                    "velocity": f"replicate_{idx}",
                    "accuracy": acc,
                    "error_rate": 1.0 - acc,
                    "details": extras,
                }
            )
    elif generator_type == "gan":
        if gan_dir is None:
            raise ValueError("gan_dir must be provided for GAN evaluation")
        files = sorted(Path(gan_dir).glob("**/*.npz"))
        if not files:
            raise RuntimeError(f"No .npz sequences found under {gan_dir}")
        rng = np.random.default_rng(seed)
        for file_path in files:
            with np.load(file_path) as data:
                sequences = data.get("sequences")
                if sequences is None:
                    continue
                if sequences.ndim == 2:
                    sequences = sequences[None, ...]
                idxs = rng.choice(
                    sequences.shape[0],
                    size=min(samples_per_case, sequences.shape[0]),
                    replace=sequences.shape[0] < samples_per_case,
                )
                selected = [sequences[i] for i in idxs]
            synthetic_features = _features_from_sequences(selected)
            acc, extras = _train_classifier(real_features, synthetic_features, seed=seed)
            results.append(
                {
                    "shape": "gan",
                    "velocity": file_path.name,
                    "accuracy": acc,
                    "error_rate": 1.0 - acc,
                    "details": extras,
                }
            )
    else:
        raise ValueError(f"Unsupported generator type: {generator_type}")

    summary = {
        "dataset_id": dataset_id,
        "sequence_length": sequence_length,
        "sampling_rate": sampling_rate,
        "samples_per_case": samples_per_case,
        "generator": generator_type,
        "results": results,
        "average_accuracy": float(np.mean([item["accuracy"] for item in results])),
    }

    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Sigma-Lognormal baseline evaluation.")
    parser.add_argument("--dataset", default="balabit", help="Dataset identifier")
    parser.add_argument("--sequence-length", type=int, default=64, help="Gesture sequence length")
    parser.add_argument("--max-gestures", type=int, default=4096, help="Maximum real gestures to load")
    parser.add_argument("--samples", type=int, default=2000, help="Synthetic samples per shape/velocity case")
    parser.add_argument("--seed", type=int, default=1337, help="Random seed")
    parser.add_argument("--sampling-rate", type=float, default=200.0, help="Sampling rate (Hz) for synthetic generators when dynamic length is enabled")
    parser.add_argument(
        "--generator",
        choices=["function", "sigma", "gan"],
        default="function",
        help="Synthetic trajectory generator type",
    )
    parser.add_argument("--gan-dir", type=Path, default=None, help="Directory containing GAN sample NPZ files")
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON output path")
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
    args = parser.parse_args()

    summary = run_baseline(
        dataset_id=args.dataset,
        sequence_length=args.sequence_length,
        max_gestures=args.max_gestures,
        seed=args.seed,
        samples_per_case=args.samples,
        generator_type=args.generator,
        gan_dir=args.gan_dir,
        output=args.output,
        canonicalize_path=args.canon_path,
        canonicalize_duration=args.canon_duration,
        sampling_rate=args.sampling_rate,
        make_plots=True,
    )
    for item in summary["results"]:
        print(
            f"[{summary['generator']}] shape={item['shape']:<15} velocity={item['velocity']:<12}"
            f" accuracy={item['accuracy']*100:6.2f}%  error={item['error_rate']*100:5.2f}%"
        )
    print(f"Average accuracy: {summary['average_accuracy']*100:.2f}% | Error: {(1.0-summary['average_accuracy'])*100:.2f}%")


if __name__ == "__main__":
    main()
