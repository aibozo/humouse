# 1-Layer Mouse GAN Overview

This document condenses the baseline specification for the conditional "1-layer mouse GAN" project. Use it as the reference point before modifying any component of the stack.

## Problem Statement
Generate realistic human-like mouse gestures via a conditional GAN while simultaneously training a detector that distinguishes genuine human traces from synthetic ones. The baseline must act as a strong foundation for later multi-layer, session-level adversarial systems.

## Key Objectives
- Build a reproducible ingestion + preprocessing pipeline that standardises gestures into model-ready tensors and feature vectors.
- Train a conditional GAN (generator + discriminator) that produces gesture sequences conditioned on contextual cues (distance, duration, target properties, style noise).
- Train and evaluate a detector (feature-based + sequence-based fusion) against both real human data and synthetic traces.
- Establish metrics, monitoring, and experiment tracking practices to guide iterative improvements.

## Data Sources
Start with publicly accessible human mouse-dynamics datasets:
- Balabit Mouse Dynamics Challenge (benchmark sessions)
- Boğaziçi Mouse Dynamics Dataset (long-form usage)
- SapiMouse (open dataset with reference code)
- Attentive Cursor Dataset (web-focused short traces)

Store raw downloads under `data/raw/<dataset>` and processed gesture tensors under `data/processed/<dataset>`.

## Processing Pipeline
1. Ingest dataset-specific formats, normalise coordinates to `[0,1]` or ego-centred deltas, and compute `Δt` in seconds.
2. Segment continuous traces into gestures using pause thresholds or velocity-based start/stop rules.
3. Canonicalise sequence length (e.g., resample to 64 points) or pad to `L_max`; retain metadata such as duration and contextual targets.
4. Compute neuromotor feature vectors (duration, velocity, acceleration/jerk stats, curvature, path efficiency, spectral entropy, Fitts' law residuals, etc.).
5. Split train/val/test by user and reserve cross-dataset test folds to evaluate generalisation.

## Model Components
- **Generator (G):** Conditional sequence model (GRU-based or TCN decoder) that maps latent noise + condition context to gesture sequences. Includes smoothness and biomechanical regularisers.
- **Discriminator (D):** Sequence encoder (TCN or BiLSTM) with optional auxiliary regression heads predicting neuromotor stats; trained using WGAN-GP or hinge losses.
- **Detector (C):** Ensemble of feature-based (XGBoost/MLP) and sequence-based (TCN/BiLSTM + attention) branches, fused via an MLP classifier.

## Training Strategy
1. Pretrain detector on real vs simple synthetic (e.g., WindMouse) gestures.
2. Pretrain generator with supervised reconstruction to stabilise GAN training.
3. Run adversarial GAN updates (k discriminator steps per generator step) with gradient penalty and auxiliary feature regression.
4. Periodically refresh detector negatives with current generator outputs and continue alternating GAN and detector updates.

## Evaluation Checklist
- Generator realism: discriminator scores, neuromotor feature distribution distances, velocity profile correlation, Fitts' law residuals, diversity metrics.
- Detector robustness: ROC/PR AUC, FP@TPR thresholds, cross-dataset generalisation, calibration (ECE), ablations per branch.
- Experiment tracking: log runs via Weights & Biases or TensorBoard, save checkpoints and generated sample buffers.

## Ethical Note
This project is intended for benign research and red-teaming. Respect dataset licences, user privacy, and platform terms of service; do not deploy adversarial traces for malicious evasion.

