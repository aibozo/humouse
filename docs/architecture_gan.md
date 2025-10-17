# Architecture Reference — 1-Layer Mouse GAN

Use this document when implementing or modifying the generator, discriminator, and detector components. It captures interface expectations, tensor shapes, and implementation details.

## Generator (G)
- Implementation: `src/models/generator.py` (`ConditionalGenerator`).
- Inputs: latent `z` shape `(batch, latent_dim)` and condition vector `(batch, condition_dim)` (currently neuromotor feature vector length 15).
- Backbone: MLP expansion + residual TCN decoder; outputs `(batch, target_len, 3)` representing `(Δx, Δy, Δt)` with `tanh` on spatial deltas and `softplus` on `Δt`.

## Discriminator / Critic (D)
- Implementation: `src/models/discriminator.py` (`GestureDiscriminator`).
- Inputs: sequence `(batch, length, 3)` and condition vector `(batch, condition_dim)`.
- Architecture: Residual TCN encoder + condition MLP fusion; outputs critic scalar and auxiliary regression head (placeholder size 4).

## Detector (C)
- Implementation: `src/models/detector.py` (`GestureDetector`).
- Feature branch: MLP on neuromotor feature vector.
- Sequence branch: Residual TCN + adaptive pooling.
- Fusion head outputs logits `(batch,)`.

## Shared Components
- `src/models/components.py` exposes `mlp` helper and residual TCN blocks.

## Training Scripts
- GAN training loop (`src/train/train_gan.py`) implements WGAN-GP with gradient penalty, configurable critic steps, replay-buffer exports, CSV metric logging, sample artifact dumps, and optional detector co-training.
- Detector training loop (`src/train/train_detector.py`) performs supervised training with AdamW, optional replay negatives, cached gesture loading, ROC/PR metrics via `utils.eval`, W&B ROC tables, CSV/JSON summaries, validation prediction exports, and cross-dataset evaluation support.

## Data Flow
1. Load dataset via event loaders (`src/data/`) and segment gestures (`src/data/segmenter.py`).
2. Build neuromotor features (`src/features/neuromotor.py`).
3. Create batches using `GestureDataset` (`src/data/dataset.py`).
4. Feed batches into training scripts configured via Hydra.

## Next Steps
- Expand auxiliary outputs and losses (e.g., neuromotor regression) for discriminator stability.
- Evaluate additional feature metrics (spectral, temporal) to guide GAN tuning.
