"""Hydra-driven training entry point for detector experiments."""
from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import hydra
import torch
import pandas as pd
import numpy as np
from omegaconf import DictConfig, OmegaConf
from hydra.utils import to_absolute_path
from torch import nn, optim
from torch.utils.data import DataLoader, random_split

from data.dataset import GestureDataset, GestureDatasetConfig
from models.detector import DetectorConfig, GestureDetector
from train.config_schemas import (
    DataConfig,
    DetectorExperimentConfig,
    DetectorModelConfig,
    DetectorTrainingConfig,
)
from utils.eval import compute_roc_metrics, roc_curve_points
from utils.housekeeping import tidy_checkpoint_artifacts
from utils.logging import CSVMetricLogger, LoggingConfig, experiment_logger, write_summary_json, log_wandb_artifact
from utils.plotting import plot_metric_trends, plot_roc_curve

logger = logging.getLogger(__name__)


@dataclass
class CrossEvalEntry:
    key: str
    dataset_id: str
    split: Optional[str] = None
    user_filter: Optional[List[str]] = None
    description: Optional[str] = None

    def detail(self) -> Dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "split": self.split,
            "user_filter": self.user_filter,
            "description": self.description,
        }


def _load_cross_eval_manifest(manifest_path: str) -> list[CrossEvalEntry]:
    path = Path(to_absolute_path(manifest_path))
    if not path.exists():
        raise FileNotFoundError(f"Cross-eval manifest not found at {path}")

    cfg = OmegaConf.load(path)
    data = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(data, dict) or "datasets" not in data:
        raise ValueError("Cross-eval manifest must define a 'datasets' list")

    entries: list[CrossEvalEntry] = []
    for item in data["datasets"]:
        if not isinstance(item, dict):
            continue
        dataset_id = item.get("dataset_id")
        if not dataset_id:
            logger.warning("Skipping manifest entry without dataset_id: %s", item)
            continue
        split = item.get("split")
        key = item.get("name") or item.get("key")
        if not key:
            key = f"{dataset_id}_{split}" if split else dataset_id
        user_filter = item.get("user_filter")
        if user_filter is not None and not isinstance(user_filter, list):
            user_filter = [str(user_filter)]
        description = item.get("description")
        entries.append(
            CrossEvalEntry(
                key=str(key),
                dataset_id=str(dataset_id),
                split=str(split) if split else None,
                user_filter=[str(u) for u in user_filter] if user_filter else None,
                description=str(description) if description else None,
            )
        )
    return entries


def _resolve_cross_eval_entries(experiment_cfg: DetectorExperimentConfig) -> list[CrossEvalEntry]:
    entries: list[CrossEvalEntry] = []
    if experiment_cfg.data.eval_dataset_ids:
        for dataset_id in experiment_cfg.data.eval_dataset_ids:
            entries.append(
                CrossEvalEntry(
                    key=str(dataset_id),
                    dataset_id=str(dataset_id),
                    split=experiment_cfg.data.split,
                    user_filter=experiment_cfg.data.user_filter,
                )
            )

    if experiment_cfg.data.cross_eval_manifest:
        try:
            manifest_entries = _load_cross_eval_manifest(experiment_cfg.data.cross_eval_manifest)
            if manifest_entries:
                entries = manifest_entries  # manifest overrides default list
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.error("Failed to load cross-eval manifest: %s", exc)
            raise
    return entries


def _build_experiment_config(cfg: DictConfig) -> tuple[DetectorExperimentConfig, Dict[str, Any]]:
    cfg_dict: Dict[str, Any] = OmegaConf.to_container(cfg, resolve=True)  # type: ignore[assignment]
    if "experiment" in cfg_dict:
        cfg_dict = cfg_dict["experiment"]

    data_cfg = DataConfig(**cfg_dict["data"])
    model_cfg = DetectorModelConfig(detector=DetectorConfig(**cfg_dict["model"]["detector"]))
    training_cfg = DetectorTrainingConfig(
        epochs=cfg_dict["training"]["epochs"],
        lr=cfg_dict["training"]["lr"],
        weight_decay=cfg_dict["training"]["weight_decay"],
        log_interval=cfg_dict["training"]["log_interval"],
        eval_interval=cfg_dict["training"]["eval_interval"],
    )
    logging_cfg = LoggingConfig(**cfg_dict["logging"])

    experiment_cfg = DetectorExperimentConfig(
        experiment_name=cfg_dict["experiment_name"],
        seed=cfg_dict["seed"],
        data=data_cfg,
        model=model_cfg,
        training=training_cfg,
        logging=logging_cfg,
    )
    return experiment_cfg, cfg_dict


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():  # pragma: no cover
        torch.cuda.manual_seed_all(seed)


def _prepare_dataloaders(experiment_cfg: DetectorExperimentConfig) -> tuple[DataLoader, DataLoader]:
    dataset_cfg = GestureDatasetConfig(
        dataset_id=experiment_cfg.data.dataset_id,
        sequence_length=experiment_cfg.data.sequence_length,
        max_gestures=experiment_cfg.data.max_gestures,
        use_generated_negatives=experiment_cfg.data.use_generated_negatives,
        cache_enabled=experiment_cfg.data.cache_enabled,
        cache_dir=experiment_cfg.data.cache_dir,
        replay_path=experiment_cfg.data.replay_path,
        replay_sample_ratio=experiment_cfg.data.replay_sample_ratio,
        split=experiment_cfg.data.split,
        user_filter=experiment_cfg.data.user_filter,
        normalize_sequences=experiment_cfg.data.normalize_sequences,
        normalize_features=experiment_cfg.data.normalize_features,
        feature_mode=experiment_cfg.data.feature_mode,
    )
    dataset = GestureDataset(dataset_cfg)
    if len(dataset) == 0:
        raise RuntimeError("Gesture dataset produced zero samples; check preprocessing pipeline.")

    val_size = max(1, int(0.2 * len(dataset)))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset,
        batch_size=experiment_cfg.data.batch_size,
        shuffle=True,
        num_workers=experiment_cfg.data.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=experiment_cfg.data.batch_size,
        shuffle=False,
        num_workers=experiment_cfg.data.num_workers,
    )
    return train_loader, val_loader


def _evaluate_on_dataset(
    entry: CrossEvalEntry,
    experiment_cfg: DetectorExperimentConfig,
    model: GestureDetector,
    device: torch.device,
    wandb_run=None,
) -> Dict[str, float]:
    eval_cfg = GestureDatasetConfig(
        dataset_id=entry.dataset_id,
        sequence_length=experiment_cfg.data.sequence_length,
        max_gestures=experiment_cfg.data.max_gestures,
        use_generated_negatives=False,
        cache_enabled=experiment_cfg.data.cache_enabled,
        cache_dir=experiment_cfg.data.cache_dir,
        split=entry.split or experiment_cfg.data.split,
        user_filter=entry.user_filter,
        normalize_sequences=experiment_cfg.data.normalize_sequences,
        normalize_features=experiment_cfg.data.normalize_features,
        feature_mode=experiment_cfg.data.feature_mode,
    )
    dataset = GestureDataset(eval_cfg)
    if len(dataset) == 0:
        logger.warning("Cross-eval dataset %s yielded zero gestures", entry.key)
        return {}

    loader = DataLoader(
        dataset,
        batch_size=experiment_cfg.data.batch_size,
        shuffle=False,
        num_workers=experiment_cfg.data.num_workers,
    )
    criterion = nn.BCEWithLogitsLoss()
    _, _, logits, labels = _run_epoch(model, loader, criterion, None, device)
    if logits is None or labels is None:
        return {}

    logits_np = logits.numpy()
    labels_np = labels.numpy()
    metrics = compute_roc_metrics(labels_np, logits_np)

    roc_path = Path(experiment_cfg.logging.checkpoint_dir) / "plots" / f"detector_{entry.key}_roc.png"
    fpr, tpr = roc_curve_points(labels_np, logits_np)
    plot_roc_curve(fpr, tpr, roc_path, title=f"ROC ({entry.key})")

    if wandb_run is not None:
        try:
            import wandb  # type: ignore

            table = wandb.Table(columns=["fpr", "tpr"])
            for f, t in zip(fpr, tpr):
                table.add_data(float(f), float(t))
            wandb_run.log({f"detector/cross_eval/{entry.key}_roc": table})
        except ImportError:  # pragma: no cover
            pass

    return metrics


def _run_epoch(
    model: GestureDetector,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optional[optim.Optimizer],
    device: torch.device,
) -> Tuple[float, float, Optional[torch.Tensor], Optional[torch.Tensor]]:
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    logits_list: list[torch.Tensor] = []
    labels_list: list[torch.Tensor] = []

    for sequences, features, labels in loader:
        sequences = sequences.to(device)
        features = features.to(device)
        labels = labels.to(device)

        logits = model(sequences, features)
        loss = criterion(logits, labels)

        if is_train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * labels.size(0)
        preds = (torch.sigmoid(logits) >= 0.5).float()
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)

        if not is_train:
            logits_list.append(logits.detach().cpu())
            labels_list.append(labels.detach().cpu())

    avg_loss = total_loss / max(1, total_samples)
    accuracy = total_correct / max(1, total_samples)
    if not is_train and logits_list:
        logits_tensor = torch.cat(logits_list)
        labels_tensor = torch.cat(labels_list)
    else:
        logits_tensor = None
        labels_tensor = None
    return avg_loss, accuracy, logits_tensor, labels_tensor


def run_detector_training(
    experiment_cfg: DetectorExperimentConfig,
    wandb_run=None,
    metrics_logger: Optional[CSVMetricLogger] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, float]:
    _set_seed(experiment_cfg.seed)
    train_loader, val_loader = _prepare_dataloaders(experiment_cfg)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = GestureDetector(experiment_cfg.model.detector).to(device)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=experiment_cfg.training.lr,
        weight_decay=experiment_cfg.training.weight_decay,
    )
    criterion = nn.BCEWithLogitsLoss()

    metrics_csv_path = Path(experiment_cfg.logging.checkpoint_dir) / "detector_metrics.csv"
    if metrics_logger is None:
        metrics_logger = CSVMetricLogger(
            metrics_csv_path,
            fieldnames=[
                "epoch",
                "train_loss",
                "train_acc",
                "val_loss",
                "val_acc",
                "roc_auc",
                "pr_auc",
                "fpr_at_95_tpr",
            ],
        )

    last_metrics: Dict[str, float] = {}
    final_logits_np: Optional[np.ndarray] = None
    final_labels_np: Optional[np.ndarray] = None

    best_val_metrics: Dict[str, float] = {}
    best_epoch = 0
    cross_results: Dict[str, Dict[str, float]] = {}
    cross_details: Dict[str, Dict[str, Any]] = {}
    cross_entries = _resolve_cross_eval_entries(experiment_cfg)

    for epoch in range(1, experiment_cfg.training.epochs + 1):
        train_loss, train_acc, _, _ = _run_epoch(model, train_loader, criterion, optimizer, device)
        if wandb_run is not None and epoch % max(1, experiment_cfg.logging.log_interval) == 0:
            wandb_run.log({
                "detector_epoch": epoch,
                "detector/train_loss": train_loss,
                "detector/train_acc": train_acc,
            })

        val_loss, val_acc, val_logits, val_labels = _run_epoch(model, val_loader, criterion, None, device)
        eval_metrics: Dict[str, float] = {}
        if val_logits is not None and val_labels is not None:
            logits_np = val_logits.numpy()
            labels_np = val_labels.numpy()
            eval_metrics = compute_roc_metrics(labels_np, logits_np)
            final_logits_np = logits_np
            final_labels_np = labels_np
            if wandb_run is not None:
                try:
                    import wandb  # type: ignore

                    fpr, tpr = roc_curve_points(labels_np, logits_np)
                    table = wandb.Table(columns=["fpr", "tpr"])
                    for f, t in zip(fpr, tpr):
                        table.add_data(float(f), float(t))
                    wandb_run.log({"detector/val_roc_curve": table})
                except ImportError:  # pragma: no cover
                    pass

        log_payload = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
        }
        log_payload.update(eval_metrics)
        metrics_logger.log(log_payload)

        if wandb_run is not None:
            wandb_run.log({f"detector/{k}": v for k, v in log_payload.items() if k != "epoch"})

        logger.info(
            "Detector epoch %d | train_loss=%.4f acc=%.3f | val_loss=%.4f acc=%.3f | roc_auc=%.3f",
            epoch,
            train_loss,
            train_acc,
            val_loss,
            val_acc,
            log_payload.get("roc_auc", float("nan")),
        )
        last_metrics = log_payload
        if "roc_auc" in log_payload:
            if not best_val_metrics or log_payload["roc_auc"] > best_val_metrics.get("roc_auc", float("-inf")):
                best_val_metrics = log_payload.copy()
                best_epoch = epoch

    checkpoint_path = Path(experiment_cfg.logging.checkpoint_dir) / f"detector_{experiment_cfg.experiment_name}.pt"
    torch.save(model.state_dict(), checkpoint_path)
    logger.info("Saved checkpoint to %s", checkpoint_path)

    summary = {
        "final_metrics": last_metrics,
        "best_val_metrics": best_val_metrics,
        "best_val_epoch": best_epoch,
        "metrics_csv": str(metrics_csv_path),
    }

    if final_logits_np is not None and final_labels_np is not None:
        predictions_path = Path(experiment_cfg.logging.checkpoint_dir) / "detector_val_predictions.csv"
        predictions_path.parent.mkdir(parents=True, exist_ok=True)
        with predictions_path.open("w", newline="") as fp:
            writer = csv.writer(fp)
            writer.writerow(["label", "logit"])
            for label, logit in zip(final_labels_np, final_logits_np):
                writer.writerow([float(label), float(logit)])
        summary["predictions_csv"] = str(predictions_path)

        fpr, tpr = roc_curve_points(final_labels_np, final_logits_np)
        plot_roc_curve(fpr, tpr, Path(experiment_cfg.logging.checkpoint_dir) / "plots" / "detector_val_roc.png")

    plot_metric_trends(
        metrics_csv_path,
        Path(experiment_cfg.logging.checkpoint_dir) / "plots" / "detector_losses.png",
        "epoch",
        ["train_loss", "val_loss"],
        "Detector Losses",
    )
    plot_metric_trends(
        metrics_csv_path,
        Path(experiment_cfg.logging.checkpoint_dir) / "plots" / "detector_auc.png",
        "epoch",
        ["roc_auc", "pr_auc"],
        "Detector AUC Metrics",
    )

    if cross_entries:
        for entry in cross_entries:
            metrics = _evaluate_on_dataset(
                entry,
                experiment_cfg,
                model,
                device,
                wandb_run,
            )
            if metrics:
                cross_results[entry.key] = metrics
                cross_details[entry.key] = entry.detail()
        if cross_results:
            summary["cross_dataset"] = cross_results
            if cross_details:
                summary["cross_dataset_details"] = cross_details
            cross_records: list[Dict[str, Any]] = []
            for entry in cross_entries:
                entry_metrics = cross_results.get(entry.key)
                if not entry_metrics:
                    continue
                record = {
                    "name": entry.key,
                    "dataset_id": entry.dataset_id,
                    "split": entry.split,
                    "user_filter": ",".join(entry.user_filter) if entry.user_filter else None,
                }
                record.update(entry_metrics)
                cross_records.append(record)

            cross_csv = Path(experiment_cfg.logging.checkpoint_dir) / "detector_cross_eval.csv"
            default_columns = [
                "name",
                "dataset_id",
                "split",
                "user_filter",
                "roc_auc",
                "pr_auc",
                "fpr_at_95_tpr",
            ]
            if cross_records:
                cross_df = pd.DataFrame(cross_records)
            else:
                cross_df = pd.DataFrame(columns=default_columns)
            cross_df.to_csv(cross_csv, index=False)
            summary["cross_dataset_csv"] = str(cross_csv)
            if wandb_run is not None:
                try:
                    import wandb  # type: ignore

                    columns = list(cross_df.columns) or default_columns
                    table = wandb.Table(columns=columns)
                    for row in cross_df.itertuples(index=False):
                        table.add_data(*row)
                    wandb_run.log({"detector/cross_eval_table": table})
                except ImportError:  # pragma: no cover
                    pass

    summary_path = Path(experiment_cfg.logging.checkpoint_dir) / "detector_summary.json"
    write_summary_json(summary_path, summary)

    result = last_metrics.copy()
    result.update(
        {
            "summary_path": str(summary_path),
            "metrics_csv": str(metrics_csv_path),
            "plots_dir": str(Path(experiment_cfg.logging.checkpoint_dir) / "plots"),
            "checkpoint_path": str(checkpoint_path),
        }
    )
    if "cross_dataset_csv" in summary:
        result["cross_dataset_csv"] = summary["cross_dataset_csv"]
    if "predictions_csv" in summary:
        result["predictions_csv"] = summary["predictions_csv"]
    if "cross_dataset_details" in summary:
        result["cross_dataset_details"] = summary["cross_dataset_details"]
    return result


def load_detector_config(config_path: str) -> DetectorExperimentConfig:
    cfg = OmegaConf.load(config_path)
    experiment_cfg, _ = _build_experiment_config(cfg)
    return experiment_cfg


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    experiment_cfg, cfg_dict = _build_experiment_config(cfg)
    logger.info("Starting detector experiment: %s", experiment_cfg.experiment_name)
    Path(experiment_cfg.logging.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    metrics_logger = CSVMetricLogger(
        Path(experiment_cfg.logging.checkpoint_dir) / "detector_metrics.csv",
        fieldnames=[
            "epoch",
            "train_loss",
            "train_acc",
            "val_loss",
            "val_acc",
            "roc_auc",
            "pr_auc",
            "fpr_at_95_tpr",
        ],
    )

    with experiment_logger(experiment_cfg.logging, cfg_dict) as run:
        result = run_detector_training(experiment_cfg, wandb_run=run, metrics_logger=metrics_logger)

        checkpoint_dir = Path(experiment_cfg.logging.checkpoint_dir)
        artifact_files = [
            checkpoint_dir / f"detector_{experiment_cfg.experiment_name}.pt",
            checkpoint_dir / "detector_metrics.csv",
            checkpoint_dir / "detector_val_predictions.csv",
        ]
        if "metrics_csv" in result:
            artifact_files.append(Path(result["metrics_csv"]))
        if "summary_path" in result:
            artifact_files.append(Path(result["summary_path"]))
        if "cross_dataset_csv" in result:
            artifact_files.append(Path(result["cross_dataset_csv"]))

        artifact_dirs = [checkpoint_dir / "plots"]

        log_wandb_artifact(
            run,
            f"detector_run_{experiment_cfg.experiment_name}",
            "detector-run",
            artifact_files,
            artifact_dirs,
        )

        if experiment_cfg.logging.target == "wandb":
            tidy_checkpoint_artifacts(experiment_cfg.logging.checkpoint_dir)


if __name__ == "__main__":
    main()
