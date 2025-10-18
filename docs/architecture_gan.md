# Architecture Reference — 1-Layer Mouse GAN

Use this document when implementing or modifying the generator, discriminator, and detector components. It captures interface expectations, tensor shapes, and implementation details.

## Generator (G)
- Default conditional pathway: `src/models/generator.py` (`ConditionalGenerator`). Inputs: latent `z` and condition vector `(batch, condition_dim)` (neuromotor features by default). Backbone: MLP expansion + residual TCN decoder. Output dimensionality/activations now come from `GeneratorConfig` (`output_dim`, `activation_xy`, `activation_dt`); the baseline keeps `(Δx, Δy, Δt)` with `tanh`/`softplus` heads.
- Paper-aligned pathway: `src/models/gan_lstm.py::LSTMGenerator`, activated via `model.architecture: lstm` in Hydra (`experiment=train_gan_paper`). In this mode the head is linear (`activation_xy: linear`, `output_dim: 2`) so the network predicts cumulative `{x̂, ŷ}` coordinates directly, matching the BeCAPTCHA description.

## Discriminator / Critic (D)
- Default critic: `src/models/discriminator.py` (`GestureDiscriminator`) — residual TCN encoder + condition fusion, trained with WGAN-GP/hinge losses.
- Paper-aligned critic: `src/models/gan_lstm.py::LSTMDiscriminator`, mirroring the BeCAPTCHA architecture. When `absolute_coordinates` is enabled the discriminator consumes absolute `{x, y}` trajectories (input dim = 2) and uses vanilla GAN/BCE loss.

## Detector (C)
- Implementation: `src/models/detector.py` (`GestureDetector`).
- Feature branch: MLP on neuromotor feature vector.
- Sequence branch: Residual TCN + adaptive pooling.
- Fusion head outputs logits `(batch,)`.

## Shared Components
- `src/models/components.py` exposes `mlp` helper and residual TCN blocks.

- GAN training loop (`src/train/train_gan.py`) supports both WGAN-GP (default) and vanilla GAN (BCE) objectives. Select via `training.adversarial_type` (`wgan` vs `vanilla`). When running the BeCAPTCHA reproduction (`experiment=train_gan_paper`), the loop instantiates the LSTM generator/discriminator pair, skips reconstruction warm-up, and logs sigma-lognormal error metrics each epoch.
- Detector training loop (`src/train/train_detector.py`) performs supervised training with AdamW, optional replay negatives, cached gesture loading, ROC/PR metrics via `utils.eval`, W&B ROC tables, CSV/JSON summaries, validation prediction exports, and cross-dataset evaluation support.

## Data Flow
1. Load dataset via event loaders (`src/data/`) and segment gestures (`src/data/segmenter.py`).
   - For BeCAPTCHA experiments set `data.sampling_rate: 200.0` so `events_to_gesture` interpolates raw timestamps onto a 200 Hz grid before padding/truncation.
2. Build neuromotor features (`src/features/neuromotor.py`).
3. Create batches using `GestureDataset` (`src/data/dataset.py`).
4. Feed batches into training scripts configured via Hydra.

## Next Steps
- Expand auxiliary outputs and losses (e.g., neuromotor regression) for discriminator stability.
- Evaluate additional feature metrics (spectral, temporal) to guide GAN tuning.
