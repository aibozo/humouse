# Task Scratchpad

Use this document to track actionable items, progress, and open questions. Update sections when starting or finishing work to keep continuity across sessions.

## Backlog
- Flesh out GAN training loop with replay buffer and detector co-training.
- Persist processed gesture datasets to disk (parquet/npz) for faster reloads.
- Expand neuromotor feature set (spectral metrics, Fitts residuals) and update models/tests accordingly.
- Evaluate replay features vs real distribution (plots + Sigma-Lognormal histograms).
- Schedule longer steady-state GAN training runs with BeCAPTCHA hyperparameters and logging.
- Add inline GAN evaluation hook invoking sigma_log_baseline during training and logging to W&B.
- Plan hyperparameter sweeps (TTUR vs symmetric LR, supervised reconstruction weight) aligned with BeCAPTCHA study.
- Run sanity-check training cycles with `experiment=train_gan_paper` (≥10 epochs) and compare sigma error against function/sigma baselines.
- Reconcile dataset resampling with 200 Hz requirement (variable-length sequences or dedicated loader profile).
- Warm-start GAN on Δx/Δy (fixed Δt=1/200) and ensure generator head matches delta statistics.
- Add goal-geometry conditioning (`cosθ`, `sinθ`, distance, target width/style) to generator input.
- Replace jerk penalty with curvature/lateral deviation losses in canonical frame; optionally add high-frequency band penalty.
- Track theta_start/path-efficiency/jerk metrics post-change to confirm jitter matches real data.

## In Progress
- *(empty — update when work begins)*

## Blocked
- *(empty — add blockers as they arise)*

## Done / Notes
- 2025-10-16: Added Sigma-Lognormal baseline evaluator (`src/eval/sigma_log_baseline.py`) generating function-based synthetic trajectories and RandomForest metrics per shape/velocity profile.
- 2025-10-17: Added replay vs real plotting utility (`scripts/plot_replay_vs_real.py`); canonicalised gesture preprocessing (unit path/time) with sigma-feature GAN training and inline evaluation updates (`conf/experiment/train_gan.yaml`, `src/data/dataset.py`, `src/train/train_gan.py`).
- 2025-10-17: Reconciled Sigma-Lognormal feature vector with BeCAPTCHA spec and refreshed baseline generators; introduced paper-aligned LSTM/BCE GAN config (`conf/experiment/train_gan_paper.yaml`) and documentation updates.
- 2025-10-17: Added fixed-rate (200 Hz) resampling path in `GestureDataset` plus absolute-coordinate GAN training option; paper config now emits `{x, y}` positions with linear heads and converts back to deltas for metrics/exports.
- 2025-02-14: Created `.venv` (Python 3.12) and installed dependencies from `requirements.txt`; generated `requirements.lock` for reproducibility.
- 2025-02-14: Inventoried local datasets and documented schema/integration notes in `docs/data_inventory.md`, including citations for all datasets.
- 2025-02-14: Added dataset registry and validation utilities (`src/data/registry.py`, `src/data/loaders.py`) with pytest coverage (`tests/test_dataset_registry.py`).
- 2025-02-14: Implemented per-dataset loaders (`src/data/balabit_loader.py`, `src/data/bogazici_loader.py`, `src/data/attentive_loader.py`, `src/data/sapimouse_loader.py`) and tests (`tests/test_data_loaders.py`).
- 2025-02-15: Added gesture segmentation & resampling utilities with velocity/padding options (`src/data/segmenter.py`, `tests/test_segmenter.py`).
- 2025-02-15: Implemented neuromotor feature extraction (`src/features/neuromotor.py`, `tests/test_neuromotor_features.py`).
- 2025-02-15: Scaffolded generator, discriminator, and detector models with component utilities and tests (`src/models/*.py`, `tests/test_models.py`).
- 2025-02-15: Recorded Boğaziçi raw archive checksum (`data/raw/bogazici/checksums.txt`) and confirmed Balabit/SapiMouse licence status (no explicit licence; cite repos).
- 2025-02-15: Added Hydra configs, logging helpers, training entrypoints (`conf/`, `src/utils/logging.py`, `src/train/train_*.py`) with tests (`tests/test_configs.py`).
- 2025-02-15: Added gesture dataset -> dataloader utilities and integrated detector training loop with logging/checkpointing (`src/data/dataset.py`, `src/train/train_detector.py`, `tests/test_gesture_dataset.py`).
- 2025-02-15: Implemented WGAN-GP training loop (`src/train/train_gan.py`) using GestureDataset with caching, replay buffer export, sample artifacts, metric CSV logging, summary JSON, optional detector co-training, and W&B artifact packaging.
- 2025-02-15: Extended detector training (`src/train/train_detector.py`) with ROC/PR evaluation tables, CSV metrics, summary JSON, cross-dataset evaluation support, validation prediction exports, plotting utilities, and W&B artifact packaging; expanded evaluation/plotting utilities (`src/utils/eval.py`, `src/utils/plotting.py`).

## Quick Links
- Project overview: [project_overview.md](project_overview.md)
- Timeline: [timeline.md](timeline.md)
- Architecture: [architecture_gan.md](architecture_gan.md)
- Data pipeline: [data_pipeline.md](data_pipeline.md)
- Neuromotor features: [neuromotor_features.md](neuromotor_features.md)
- Data inventory: [data_inventory.md](data_inventory.md)
- Experiment tracking: [experiment_tracking.md](experiment_tracking.md)
