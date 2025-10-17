"""Post-run checkpoint aggregation CLI.

This script scans checkpoint directories for summary artifacts and prints
concise tables covering GAN and detector runs. Results may optionally be
exported to CSV for downstream analysis.
"""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class AggregationResult:
    """Structured aggregation output for checkpoint summaries."""

    gan_rows: list[dict[str, Any]] = field(default_factory=list)
    detector_rows: list[dict[str, Any]] = field(default_factory=list)
    cross_rows: list[dict[str, Any]] = field(default_factory=list)

    def as_records(self) -> list[dict[str, Any]]:
        """Flatten the aggregated rows with run type markers."""
        records: list[dict[str, Any]] = []
        records.extend({"run_type": "gan", **row} for row in self.gan_rows)
        records.extend({"run_type": "detector", **row} for row in self.detector_rows)
        records.extend({"run_type": "detector_cross", **row} for row in self.cross_rows)
        return records


def _load_json(path: Path) -> Dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as fp:
            return json.load(fp)
    except json.JSONDecodeError as exc:  # pragma: no cover - invalid files are rare
        logger.warning("Failed to parse JSON from %s: %s", path, exc)
    except FileNotFoundError:  # pragma: no cover - defensive
        logger.debug("Summary file %s disappeared before reading", path)
    return {}


def _run_label(summary_path: Path, root: Path) -> str:
    try:
        relative = summary_path.parent.relative_to(root)
        label = str(relative)
        if label == ".":
            return summary_path.parent.name
        return label
    except ValueError:  # pragma: no cover - root mismatch
        return summary_path.parent.name


def _aggregate_gan(summary: Dict[str, Any], run_label: str) -> dict[str, Any]:
    return {
        "run": run_label,
        "final_epoch": summary.get("final_epoch"),
        "final_step": summary.get("final_step"),
        "g_loss": summary.get("g_loss"),
        "d_loss": summary.get("d_loss"),
        "feature_l1": summary.get("feature_l1"),
        "feature_cov_diff": summary.get("feature_cov_diff"),
        "diversity_xy": summary.get("diversity_xy"),
        "replay_buffer_size": summary.get("replay_buffer_size"),
    }


def _extract_val_metric(summary: Dict[str, Any], key: str) -> Any:
    final_metrics = summary.get("final_metrics", {}) or {}
    best_metrics = summary.get("best_val_metrics", {}) or {}
    if key in final_metrics and final_metrics.get(key) is not None:
        return final_metrics.get(key)
    return best_metrics.get(key)


def _aggregate_detector(summary: Dict[str, Any], run_label: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    row = {
        "run": run_label,
        "best_val_epoch": summary.get("best_val_epoch"),
        "train_loss": _extract_val_metric(summary, "train_loss"),
        "val_loss": _extract_val_metric(summary, "val_loss"),
        "roc_auc": _extract_val_metric(summary, "roc_auc"),
        "pr_auc": _extract_val_metric(summary, "pr_auc"),
        "fpr_at_95_tpr": _extract_val_metric(summary, "fpr_at_95_tpr"),
    }
    cross_rows: list[dict[str, Any]] = []
    cross_section = summary.get("cross_dataset") or {}
    cross_details = summary.get("cross_dataset_details") or {}

    for dataset_id, metrics in cross_section.items():
        entry = {
            "run": run_label,
            "dataset": dataset_id,
        }
        if dataset_id in cross_details:
            details = cross_details[dataset_id] or {}
            entry.update({
                "dataset_id": details.get("dataset_id"),
                "split": details.get("split"),
                "user_filter": ",".join(details.get("user_filter", [])) if details.get("user_filter") else None,
                "description": details.get("description"),
            })
        entry.update({k: metrics.get(k) for k in ("roc_auc", "pr_auc", "fpr_at_95_tpr") if k in metrics})
        cross_rows.append(entry)
    return row, cross_rows


def aggregate_checkpoints(root: Path | str) -> AggregationResult:
    """Scan the provided root directory for summary artifacts."""
    root_path = Path(root).expanduser().resolve()
    result = AggregationResult()

    if not root_path.exists():
        logger.info("Checkpoint root %s does not exist", root_path)
        return result

    summary_paths = sorted(root_path.rglob("*_summary.json"))
    if not summary_paths:
        logger.info("No summary JSON files found under %s", root_path)
        return result

    for summary_path in summary_paths:
        summary = _load_json(summary_path)
        if not summary:
            continue
        run_label = _run_label(summary_path, root_path)
        name = summary_path.name
        if name == "gan_summary.json":
            result.gan_rows.append(_aggregate_gan(summary, run_label))
        elif name == "detector_summary.json":
            detector_row, cross_rows = _aggregate_detector(summary, run_label)
            result.detector_rows.append(detector_row)
            result.cross_rows.extend(cross_rows)
        else:
            logger.debug("Skipping unknown summary file %s", summary_path)

    return result


def _print_table(title: str, rows: Iterable[dict[str, Any]], columns: List[str]) -> None:
    rows_list = list(rows)
    if not rows_list:
        return
    df = pd.DataFrame(rows_list)
    missing_cols = [col for col in columns if col not in df.columns]
    for col in missing_cols:
        df[col] = None
    df = df[columns]
    numeric_cols = df.select_dtypes(include=["float", "int"]).columns
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    df[numeric_cols] = df[numeric_cols].round(4)
    print(f"\n{title}")
    print(df.to_string(index=False))


def _save_records(path: Path, records: list[dict[str, Any]]) -> None:
    if not records:
        logger.info("No records to persist to %s", path)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(records)
    numeric_cols = df.select_dtypes(include=["float", "int"]).columns
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    df.to_csv(path, index=False)
    logger.info("Wrote aggregated summary to %s", path)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Summarise checkpoint metrics across runs.")
    parser.add_argument(
        "--root",
        type=str,
        default="checkpoints",
        help="Root directory containing run subfolders and summary artifacts.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional CSV path to persist combined run metrics.",
    )
    args = parser.parse_args(argv)

    result = aggregate_checkpoints(args.root)
    if not result.gan_rows and not result.detector_rows:
        print(f"No checkpoint summaries found under {args.root}.")
        return

    _print_table(
        "GAN Runs",
        result.gan_rows,
        [
            "run",
            "final_epoch",
            "final_step",
            "g_loss",
            "d_loss",
            "feature_l1",
            "feature_cov_diff",
            "diversity_xy",
        ],
    )
    _print_table(
        "Detector Runs",
        result.detector_rows,
        [
            "run",
            "best_val_epoch",
            "val_loss",
            "roc_auc",
            "pr_auc",
            "fpr_at_95_tpr",
        ],
    )
    _print_table(
        "Cross-Dataset Metrics",
        result.cross_rows,
        [
            "run",
            "dataset",
            "dataset_id",
            "split",
            "user_filter",
            "description",
            "roc_auc",
            "pr_auc",
            "fpr_at_95_tpr",
        ],
    )

    if args.output:
        output_path = Path(args.output)
        _save_records(output_path, result.as_records())


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
