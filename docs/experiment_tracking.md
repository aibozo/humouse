# Experiment Tracking & Configuration

## Hydra Configuration
- Base config: `conf/config.yaml` (sets defaults and Hydra output directory).
- Experiment profiles:
  - `conf/experiment/train_gan.yaml`
  - `conf/experiment/train_detector.yaml`
- Launch examples:
  ```bash
  python src/train/train_gan.py
  python src/train/train_detector.py experiment=train_detector
  ```

## Dataset & Dataloaders
- Gesture dataset wrapper: `src/data/dataset.py` (generates gestures + neuromotor features, optional synthetic/replay negatives).
- Positive gestures are cached under `data/processed/<dataset>/gestures_*.npz` to avoid reprocessing; toggle via `cache_enabled` in the data config.
- Replay buffer outputs from GAN training can be sampled by the detector via `replay_path` / `replay_sample_ratio` config keys.
- `train_detector.py` uses `GestureDataset` with automatic train/val split via `torch.utils.data.random_split`.

## Logging & Artifacts
- Logging configuration is defined in each experiment under the `logging` section and mapped to `utils.logging.LoggingConfig`.
- Default target uses Weights & Biases (`target: wandb`); adjust entity/project/mode as needed.
- GAN loop logs discriminator/generator losses, feature distance (`feature_l1`), covariance drift (`feature_cov_diff`), and diversity metrics every `log_interval` steps; metrics are also persisted to `checkpoints/gan/metrics.csv` via `CSVMetricLogger`, with trend plots under `plots/` and summary metadata saved to `gan_summary.json`.
- Generated gesture samples are exported to `checkpoints/gan/samples/` as compressed NPZ files on each `sample_interval`.
- Detector training logs ROC/PR metrics (including `roc_auc`, `pr_auc`, `fpr@95%TPR`) and, when W&B is enabled, uploads ROC curve tables; metrics are mirrored to `checkpoints/detector/detector_metrics.csv` and summary stats saved to `detector_summary.json`. Final validation predictions stored in `detector_val_predictions.csv`.
- When W&B logging is active, both GAN and detector scripts package checkpoints, metrics CSVs, plots, samples, and summaries into artifacts (`gan_run_*`, `detector_run_*`).
- Detector training logs ROC/PR metrics (including `roc_auc`, `pr_auc`, `fpr@95%TPR`) and, when W&B is enabled, uploads ROC curve tables; metrics are mirrored to `checkpoints/detector/detector_metrics.csv`.
- Checkpoints stored under `logging.checkpoint_dir` (created automatically).

## Training Scripts
- GAN training (`src/train/train_gan.py`): full WGAN-GP loop with gradient penalty, critic steps, replay-buffer exports, CSV metrics, sample artifacts, and optional detector co-training.
- Detector training (`src/train/train_detector.py`): supervised training with AdamW + BCE loss, leveraging cached gestures, replay negatives, ROC/PR logging, trend plots, CSV/JSON summaries, cross-dataset evaluation (`data.eval_dataset_ids`), and validation prediction exports.

## Tests
- `tests/test_configs.py` validates experiment configs.
- `tests/test_gesture_dataset.py` covers dataset caching, replay buffer sampling, and tensor outputs; `tests/test_configs.py` exercises evaluation/plotting utilities.

- ## Optional Detector Co-Training
- Controlled via `training.co_train_detector` in `conf/experiment/train_gan.yaml`.
- `detector_update_every` defines the GAN epoch cadence; each trigger runs `detector_epochs_per_update` epochs using the replay buffer as additional negatives.
- Co-training reuses `train_detector` utilities with W&B/table logging (if enabled) and writes metrics into the shared `detector_metrics.csv` / `detector_summary.json` artifacts.
