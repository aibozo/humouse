"""Evaluation utilities for detector metrics."""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score, roc_curve


def compute_roc_metrics(labels: np.ndarray, logits: np.ndarray) -> Dict[str, float]:
    if labels.shape != logits.shape:
        raise ValueError("labels and logits must have the same shape")
    probs = 1 / (1 + np.exp(-logits))
    roc_auc = roc_auc_score(labels, probs)
    precision, recall, _ = precision_recall_curve(labels, probs)
    pr_auc = auc(recall, precision)

    fpr, tpr, thresholds = roc_curve(labels, probs)
    target_tpr = 0.95
    idx = np.searchsorted(tpr, target_tpr)
    if idx >= len(fpr):
        fpr_at_95 = fpr[-1]
        threshold_at_95 = thresholds[-1]
    else:
        fpr_at_95 = fpr[idx]
        threshold_at_95 = thresholds[idx]

    return {
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "fpr_at_95_tpr": float(fpr_at_95),
        "threshold_at_95_tpr": float(threshold_at_95),
    }


def feature_distribution_metrics(real_features: np.ndarray, fake_features: np.ndarray) -> Dict[str, float]:
    if real_features.ndim != 2 or fake_features.ndim != 2:
        raise ValueError("feature arrays must be rank-2")
    if real_features.shape[1] != fake_features.shape[1]:
        raise ValueError("feature dimensionality mismatch")

    real_mean = real_features.mean(axis=0)
    fake_mean = fake_features.mean(axis=0)
    mean_l1 = np.mean(np.abs(fake_mean - real_mean))
    mean_l2 = np.linalg.norm(fake_mean - real_mean)

    # simple covariance trace comparison
    real_cov = np.cov(real_features, rowvar=False)
    fake_cov = np.cov(fake_features, rowvar=False)
    cov_diff = np.trace(np.abs(real_cov - fake_cov)) / real_features.shape[1]

    return {
        "mean_l1": float(mean_l1),
        "mean_l2": float(mean_l2),
        "cov_trace_diff": float(cov_diff),
    }


def sequence_diversity_metric(sequences: np.ndarray) -> float:
    if sequences.ndim != 3:
        raise ValueError("sequences must be (N, L, C)")
    # compute mean std over spatial dims as quick proxy
    spatial = sequences[..., :2]
    return float(np.std(spatial, axis=1).mean())


def roc_curve_points(labels: np.ndarray, logits: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if labels.shape != logits.shape:
        raise ValueError("labels and logits must have the same shape")
    probs = 1 / (1 + np.exp(-logits))
    fpr, tpr, _ = roc_curve(labels, probs)
    return fpr, tpr
