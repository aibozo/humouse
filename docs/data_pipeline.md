# Data Pipeline Specification

Follow this specification when implementing dataset ingestion, segmentation, feature extraction, preprocessing, and model integration.

## 1. Ingestion
- Dataset registry lives under `src/data/registry.py`; validation helpers in `src/data/loaders.py`.
- Dataset-specific readers currently implemented: Balabit, Boğaziçi, Attentive Cursor, SapiMouse (provisional).
- Standardise raw columns to `(timestamp_ms, x, y, button_state, viewport_width, viewport_height, target_x, target_y, target_width)` where available.

## 2. Normalisation
- Choose coordinate frame per experiment (viewport-normalised vs ego-centric).
- Convert timestamps to seconds, compute `Δt` with minimum clamp, drop duplicate timestamps.

## 3. Gesture Segmentation
- `segment_event_stream` (`src/data/segmenter.py`) segments event streams using gap threshold and optional velocity triggers.
- `events_to_gesture` resamples or pads to fixed length sequences of `(Δx, Δy, Δt)`.

## 4. Feature Extraction
- `src/features/neuromotor.py` implements per-gesture feature vectors (`compute_features`, `compute_feature_matrix`).

## 5. Model Integration
- Gesture dataset/dataloader scaffolding in `src/data/dataset.py` converts segmented gestures + features into PyTorch-ready batches (with optional synthetic + replay negatives and on-disk caching). Cached positives feed GAN/detector training, while replay outputs are persisted for co-training, metrics, and cross-evaluation.
- Detector training loop scaffold lives in `src/train/train_detector.py`; GAN scaffold in `src/train/train_gan.py`.
- Hydra configuration + logging documented in `docs/experiment_tracking.md`.

## 6. Splitting Strategy
- User holdout; cross-dataset evaluation; manifest storage under `data/processed/<dataset>/splits/`.

## 7. Data Versioning
- Metadata + checksums under `data/raw/` and `data/processed/`.

## 8. Validation
- Tests cover loaders (`tests/test_data_loaders.py`), registry (`tests/test_dataset_registry.py`), segmentation (`tests/test_segmenter.py`), features (`tests/test_neuromotor_features.py`), gesture dataset (`tests/test_gesture_dataset.py`), and model scaffolding (`tests/test_models.py`).
