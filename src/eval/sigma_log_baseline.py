"""Evaluate Sigma-Lognormal classifier baselines against synthetic trajectories."""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from data.dataset import GestureDataset, GestureDatasetConfig
from features import sigma_lognormal_features_from_sequence


@dataclass
class RealGestureStats:
    lengths: np.ndarray
    durations: np.ndarray


class FunctionBasedTrajectoryGenerator:
    """Replicates BeCAPTCHA function-driven synthetic trajectories."""

    def __init__(self, stats: RealGestureStats, sequence_length: int = 64, seed: int | None = None):
        self.stats = stats
        self.sequence_length = sequence_length
        self.rng = np.random.default_rng(seed)

    def _sample_length(self) -> float:
        return float(self.rng.choice(self.stats.lengths))

    def _sample_duration(self) -> float:
        return float(self.rng.choice(self.stats.durations))

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

    def _velocity_weights(self, profile: str) -> np.ndarray:
        n = self.sequence_length
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
        weights = self._velocity_weights(velocity_profile)
        cumulative = np.concatenate([[0.0], np.cumsum(weights)])
        cumulative /= cumulative[-1]

        points = self._shape_points(cumulative, shape)
        target_length = max(self._sample_length(), 1e-3)
        diffs = np.diff(points, axis=0)
        current_len = np.sum(np.linalg.norm(diffs, axis=1))
        scale = target_length / max(current_len, 1e-6)
        points *= scale

        theta = self.rng.uniform(-math.pi, math.pi)
        rot = np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])
        points = points @ rot.T

        dx_dy = np.diff(points, axis=0)
        duration = max(self._sample_duration(), 1e-3)
        dt = weights / weights.sum() * duration
        sequence = np.stack([dx_dy[:, 0], dx_dy[:, 1], dt], axis=1)
        return sequence.astype(np.float32)

    def generate_batch(self, num_samples: int, shape: str, velocity_profile: str) -> List[np.ndarray]:
        return [self.generate_sequence(shape, velocity_profile) for _ in range(num_samples)]


def _collect_real_stats(dataset: GestureDataset) -> RealGestureStats:
    lengths: List[float] = []
    durations: List[float] = []
    for sequence, _, label in dataset.samples:
        if label.item() != 1.0:
            continue
        seq_np = sequence.cpu().numpy()
        dx = seq_np[:, 0]
        dy = seq_np[:, 1]
        dt = seq_np[:, 2]
        lengths.append(float(np.sum(np.sqrt(dx**2 + dy**2))))
        durations.append(float(np.sum(dt)))
    return RealGestureStats(lengths=np.array(lengths), durations=np.array(durations))


def _features_from_sequences(sequences: Iterable[np.ndarray]) -> np.ndarray:
    feature_list: List[np.ndarray] = []
    for seq in sequences:
        tensor = torch.from_numpy(seq)
        feature = sigma_lognormal_features_from_sequence(tensor)
        feature_list.append(feature.numpy())
    return np.stack(feature_list)


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
    output: Path | None = None,
) -> dict:
    dataset_cfg = GestureDatasetConfig(
        dataset_id=dataset_id,
        sequence_length=sequence_length,
        max_gestures=max_gestures,
        use_generated_negatives=False,
        cache_enabled=False,
        normalize_sequences=False,
        normalize_features=False,
        feature_mode="sigma_lognormal",
    )
    dataset = GestureDataset(dataset_cfg)
    stats = _collect_real_stats(dataset)
    generator = FunctionBasedTrajectoryGenerator(stats, sequence_length=sequence_length, seed=seed)

    real_features = dataset.get_positive_features_tensor().numpy()

    shapes = ["linear", "quadratic", "exponential"]
    velocities = ["constant", "logarithmic", "gaussian"]
    results = []
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

    summary = {
        "dataset_id": dataset_id,
        "sequence_length": sequence_length,
        "samples_per_case": samples_per_case,
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
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON output path")
    args = parser.parse_args()

    summary = run_baseline(
        dataset_id=args.dataset,
        sequence_length=args.sequence_length,
        max_gestures=args.max_gestures,
        seed=args.seed,
        samples_per_case=args.samples,
        output=args.output,
    )
    for item in summary["results"]:
        print(
            f"shape={item['shape']:<10} velocity={item['velocity']:<12} accuracy={item['accuracy']*100:6.2f}%"
            f"  error={item['error_rate']*100:5.2f}%"
        )
    print(f"Average accuracy: {summary['average_accuracy']*100:.2f}% | Error: {(1.0-summary['average_accuracy'])*100:.2f}%")


if __name__ == "__main__":
    main()
