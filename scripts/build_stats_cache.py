#!/usr/bin/env python3
"""Pre-compute GestureDataset statistics and optional feature reservoirs."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from data.dataset import GestureDataset, GestureDatasetConfig
from dataclasses import fields

from train.config_schemas import DataConfig

logger = logging.getLogger(__name__)


def _load_experiment_dict(path: Path) -> dict[str, Any]:
    cfg = OmegaConf.load(path)
    if isinstance(cfg, DictConfig):
        cfg = OmegaConf.to_container(cfg, resolve=True)
    if "experiment" in cfg:
        cfg = cfg["experiment"]
    if "data" not in cfg:
        raise ValueError(f"Configuration at {path} does not contain a 'data' section.")
    return cfg


def _build_data_config(base: dict[str, Any], args: argparse.Namespace) -> DataConfig:
    data_dict = dict(base["data"])
    if args.dataset_id:
        data_dict["dataset_id"] = args.dataset_id
    if args.sequence_length is not None:
        data_dict["sequence_length"] = args.sequence_length
    if args.max_gestures is not None:
        data_dict["max_gestures"] = args.max_gestures
    if args.split is not None:
        data_dict["split"] = args.split
    if args.sampling_rate is not None:
        data_dict["sampling_rate"] = args.sampling_rate
    if args.feature_reservoir_size is not None:
        data_dict["feature_reservoir_size"] = args.feature_reservoir_size
    valid_fields = {f.name for f in fields(DataConfig)}
    data_dict = {k: v for k, v in data_dict.items() if k in valid_fields}
    return DataConfig(**data_dict)


def _dataset_config_from_data(data_cfg: DataConfig) -> GestureDatasetConfig:
    return GestureDatasetConfig(
        dataset_id=data_cfg.dataset_id,
        sequence_length=data_cfg.sequence_length,
        max_gestures=data_cfg.max_gestures,
        sampling_rate=data_cfg.sampling_rate,
        min_events=data_cfg.min_events,
        split=data_cfg.split,
        user_filter=data_cfg.user_filter,
        use_generated_negatives=data_cfg.use_generated_negatives,
        cache_enabled=data_cfg.cache_enabled,
        cache_dir=data_cfg.cache_dir,
        replay_path=data_cfg.replay_path,
        replay_sample_ratio=data_cfg.replay_sample_ratio,
        normalize_sequences=data_cfg.normalize_sequences,
        normalize_features=data_cfg.normalize_features,
        feature_mode=data_cfg.feature_mode,
        canonicalize_path=data_cfg.canonicalize_path,
        canonicalize_duration=data_cfg.canonicalize_duration,
        include_goal_geometry=data_cfg.include_goal_geometry,
        use_click_boundaries=data_cfg.use_click_boundaries,
        click_button=data_cfg.click_button,
        direction_buckets=data_cfg.direction_buckets,
        rotate_to_buckets=data_cfg.rotate_to_buckets,
        min_path_length=data_cfg.min_path_length,
        feature_reservoir_size=data_cfg.feature_reservoir_size,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build cached statistics/reservoirs for GestureDataset.")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to experiment YAML/JSON containing a 'data' section (e.g., conf/experiment/train_gan.yaml).",
    )
    parser.add_argument("--dataset-id", type=str, help="Override dataset_id from config.")
    parser.add_argument("--sequence-length", type=int, help="Override sequence_length.")
    parser.add_argument("--max-gestures", type=int, help="Override max_gestures.")
    parser.add_argument("--split", type=str, help="Override dataset split.")
    parser.add_argument("--sampling-rate", type=float, help="Override sampling_rate.")
    parser.add_argument(
        "--feature-reservoir-size",
        type=int,
        help="Override feature_reservoir_size (set to 0 to disable reservoir sampling).",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper()))
    experiment_dict = _load_experiment_dict(args.config)
    data_cfg = _build_data_config(experiment_dict, args)
    dataset_cfg = _dataset_config_from_data(data_cfg)

    logger.info("Building GestureDataset cache for %s (reservoir_size=%s)", dataset_cfg.dataset_id, dataset_cfg.feature_reservoir_size)
    dataset = GestureDataset(dataset_cfg)
    if len(dataset) == 0:
        raise RuntimeError("Gesture dataset produced zero samples; nothing to cache.")

    meta_path, reservoir_path = dataset.metadata_paths()
    conditioning = dataset.get_positive_features_tensor()
    full_features = dataset.get_positive_features_tensor(use_full=True)

    logger.info("Total samples: %d | Conditioning features: %d | Full features: %d", len(dataset), conditioning.size(0), full_features.size(0))
    if meta_path is not None:
        if meta_path.exists():
            logger.info("Metadata saved to %s", meta_path)
        else:
            logger.warning("Metadata path %s does not exist; caching may be disabled.", meta_path)
    if reservoir_path is not None:
        if dataset_cfg.feature_reservoir_size:
            logger.info("Reservoir tensor saved to %s", reservoir_path)
        elif reservoir_path.exists():
            logger.info("Reservoir tensor present at %s (feature_reservoir_size disabled in config)", reservoir_path)


if __name__ == "__main__":
    main()
