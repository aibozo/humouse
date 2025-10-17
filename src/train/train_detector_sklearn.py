"""Train classical detectors (Random Forest, SVM) on gesture features."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Tuple

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from data.dataset import GestureDataset, GestureDatasetConfig
from train.config_schemas import DataConfig, LoggingConfig


def _load_dataset(data_cfg: DataConfig) -> Tuple[np.ndarray, np.ndarray]:
    dataset_cfg = GestureDatasetConfig(
        dataset_id=data_cfg.dataset_id,
        sequence_length=data_cfg.sequence_length,
        max_gestures=data_cfg.max_gestures,
        min_events=5,
        use_generated_negatives=data_cfg.use_generated_negatives,
        cache_enabled=data_cfg.cache_enabled,
        cache_dir=data_cfg.cache_dir,
        replay_path=data_cfg.replay_path,
        replay_sample_ratio=data_cfg.replay_sample_ratio,
        normalize_sequences=data_cfg.normalize_sequences,
        normalize_features=data_cfg.normalize_features,
        feature_mode=data_cfg.feature_mode,
    )
    dataset = GestureDataset(dataset_cfg)
    features = []
    labels = []
    for sequence, feature, label in dataset.samples:
        features.append(feature.cpu().numpy())
        labels.append(int(label.item()))
    return np.stack(features), np.array(labels)


def _build_model(model_cfg: Dict[str, Any]):
    model_type = model_cfg.get("type", "random_forest").lower()
    params = model_cfg.get("params", {})
    if model_type == "random_forest":
        return RandomForestClassifier(**params)
    if model_type == "svm":
        return SVC(**params)
    raise ValueError(f"Unsupported sklearn model type: {model_type}")


def _train_and_evaluate(
    X: np.ndarray,
    y: np.ndarray,
    model_cfg: Dict[str, Any],
    seed: int,
) -> Dict[str, Any]:
    test_size = float(model_cfg.get("test_size", 0.2))
    stratify = y if np.unique(y).size > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=stratify,
    )
    model = _build_model(model_cfg)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    metrics = {
        "accuracy": float(accuracy_score(y_test, preds)),
        "f1_macro": float(f1_score(y_test, preds, average="macro")),
        "report": classification_report(y_test, preds, output_dict=False),
    }
    return {
        "metrics": metrics,
        "model": model,
    }


def _write_summary(output_dir: Path, metrics: Dict[str, Any], cfg_dict: Dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "config": cfg_dict,
        "metrics": metrics,
    }
    summary_path = output_dir / "sklearn_summary.json"
    with summary_path.open("w") as fp:
        json.dump(summary, fp, indent=2)
    (output_dir / "classification_report.txt").write_text(metrics["report"])


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    if "experiment" in cfg_dict:
        cfg_dict = cfg_dict["experiment"]

    data_cfg = DataConfig(**cfg_dict["data"])
    logging_cfg = LoggingConfig(**cfg_dict["logging"])
    model_cfg = cfg_dict.get("model", {})

    X, y = _load_dataset(data_cfg)
    result = _train_and_evaluate(X, y, model_cfg, seed=cfg_dict.get("seed", 1337))

    output_dir = Path(logging_cfg.checkpoint_dir) / "sklearn"
    _write_summary(output_dir, result["metrics"], cfg_dict)


if __name__ == "__main__":
    main()
