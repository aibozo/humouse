# Task Scratchpad

Use this document to track actionable items, progress, and open questions. Update sections when starting or finishing work to keep continuity across sessions.

## Backlog
- [ ] Profile GAN training loop after copy/memory tweaks (torch profiler + `nvidia-smi dmon`) – confirm GPU util after profiler run
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
- Recreate paper-faithful data regime (click-to-click segmentation or Shen benchmark) to compare against gap-based gestures.
- Configure unconditioned LSTM GAN run (no feature conditioning, shared LR) matching BeCAPTCHA training schedule.
- Audit sigma baseline pipeline to mirror paper’s per-direction RF evaluation before reintroducing advanced constraints.
- Switch gesture resampling to data-driven Δt (derive sampling rate from raw timestamp deltas per dataset).
- Build octant bucketing pipeline (per-gesture atan2 binning + balanced loaders).
- Prototype rotation-based augmentation to populate sparse octants while preserving original gestures.

## In Progress
- 2025-10-24: Plan & implement diffusion generator + D-eval hooks (diffusion package, training loop, metrics integration) — diffusion trainer + DDIM sampler scaffolded; GAN now calls diffusion eval (feature/density metrics) each epoch when configured; diffusion training now records C2ST metrics + CSV logger.

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
- 2025-02-16: Cached sigma-lognormal features inside `GestureDataset` for reuse during GAN training, updated `_feature_tensor_from_sequences` to avoid recomputing real batches, and raised GAN dataloader `num_workers` to 32 for faster pipeline throughput.
- 2025-02-16: Parallelised sigma feature extraction for synthetic batches, moved cached tensors into shared memory, added configurable feature worker pool + persistent dataloader workers, tuned configs to curb memory spikes, and ported sigma features to torch with optional GPU execution.
- 2025-10-22: Added cached dataset statistics/reservoir metadata pipeline (`src/data/dataset.py`), configurable feature reservoirs via `feature_reservoir_size`, and helper script `scripts/build_stats_cache.py` for regenerating stats.
- 2025-10-22: Added `scripts/profile_gan_training.py` torch-profiler harness and updated GAN configs to consume conditioning reservoirs; training loop now uses reservoir-backed feature pools.

## Quick Links
- Project overview: [project_overview.md](project_overview.md)
- Timeline: [timeline.md](timeline.md)
- Architecture: [architecture_gan.md](architecture_gan.md)
- Data pipeline: [data_pipeline.md](data_pipeline.md)
- Neuromotor features: [neuromotor_features.md](neuromotor_features.md)
- Data inventory: [data_inventory.md](data_inventory.md)
- Experiment tracking: [experiment_tracking.md](experiment_tracking.md)
