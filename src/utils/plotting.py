"""Plotting utilities for experiment metrics."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import pandas as pd


def plot_metric_trends(
    csv_path: Path | str,
    output_path: Path | str,
    x_column: str,
    metric_columns: Sequence[str],
    title: str,
) -> Path:
    csv_path = Path(csv_path)
    if not csv_path.exists():
        return Path(output_path)
    df = pd.read_csv(csv_path)
    if x_column not in df.columns:
        raise ValueError(f"{x_column} not present in {csv_path}")

    plt.figure(figsize=(10, 6))
    for metric in metric_columns:
        if metric in df.columns:
            plt.plot(df[x_column], df[metric], label=metric)
    plt.title(title)
    plt.xlabel(x_column)
    plt.ylabel("value")
    plt.legend()
    plt.grid(True, alpha=0.3)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    return output_path


def plot_roc_curve(
    fpr: Iterable[float],
    tpr: Iterable[float],
    output_path: Path | str,
    title: str = "ROC Curve",
) -> Path:
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label="ROC")
    plt.plot([0, 1], [0, 1], "--", color="gray", label="random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    return output_path


__all__ = ["plot_metric_trends", "plot_roc_curve"]

