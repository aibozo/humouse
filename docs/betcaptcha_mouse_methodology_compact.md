# BeCAPTCHA-Mouse Methodology (Condensed)

## System Intent
- Augment traditional CAPTCHAs with a behavioral layer that labels individual mouse trajectories as *human* or *bot* within one movement between clicks.
- Leverage neuromotor modeling plus realistic synthetic trajectories to both harden detectors and evaluate them against worst-case attacks.

## Benchmark & Data Prep
- Source data: Shen et al. mouse benchmark (58 users, >200k trajectories, each session contains two passes over 8 click targets). Paper samples 35 random trajectories per user ⇒ >5k real tracks.
- Trajectory unit: movement between consecutive clicks; raw logs include `{x, y}` (pixels), event type, timestamp.
- Resampling: real and synthetic signals are represented at 200 Hz; GAN outputs can be downsampled or retrained for other rates.
- Directional modelling: 8 canonical directions (8→1, 1→2, …, 7→8). For synthesis the number of samples `M` per trajectory is drawn from a normal distribution that matches the mean/std of each direction in real data.
- Benchmark released as 15k trajectories: 5k real + 5k function-based synthetics + 5k GAN synthetics covering varied lengths, directions, and velocities.

## Neuromotor Feature Pipeline
- Velocity decomposition: apply Sigma-Lognormal model to the speed profile `|v(t)|`, automatically estimating `N` strokes and parameters per stroke using [Fischer & Plamondon, 2017].
- Stroke parameters (`i`-th stroke):
  - `Di`: path length contribution (input pulse).
  - `t0i`: onset time shift.
  - `µi`: log-temporal delay.
  - `σi`: neuromotor response time.
  - `θsi`, `θei`: start/end angles.
- Velocity reconstruction equals the sum of lognormals, capturing agonist/antagonist bursts and end-of-path corrections (Fig. 3.b in paper).
- Feature vector (37 dims per trajectory):
  - For each of the six parameters, compute `{max, min, mean}` separately on the first and second half of the trajectory (6×6 stats).
  - Append the stroke count `N` (complexity indicator).
- Optional augmentation: concatenate six global statistics from Chu et al. (2018)—duration, distance, displacement, average angle, average velocity, movement efficiency (distance/displacement).

## Synthetic Trajectory Generators

### Function-Based Synthesizer
- Shapes: linear, quadratic (`y = ax^2 + bx + c`), exponential; coefficients sampled from ranges fitted to human data.
- Velocity profiles (`VP`):
  1. Constant spacing (uniform velocity).
  2. Logarithmic spacing (initial acceleration).
  3. Gaussian spacing (accel + decel).
- Generation steps per sample:
  1. Select start/end points consistent with target pairing.
  2. Sample `M` from direction-specific Gaussian, then allocate `x̂` samples according to chosen velocity profile.
  3. Fit shape to `x̂` and endpoints to obtain `ŷ`.
- Outputs cover 9 shape/velocity combinations; combined with direction sampling yields 5k trajectories. VP=3 most closely mimics human kinematics and proved hardest to classify.

### GAN-Based Synthesizer
- Latent seed `z ∈ ℝ¹⁰⁰ ~ N(0, I)`; generator outputs `{x̂, ŷ}` (length `M`) at 200 Hz.
- Architecture:
  - Generator: LSTM(128, ReLU) → LSTM(64, ReLU) → TimeDistributed Dense(2) (linear outputs).
  - Discriminator: LSTM(128, LeakyReLU) → LSTM(64, LeakyReLU) → Dense(1, Sigmoid).
- Training regimen:
  - Use 60% of real trajectories; batch size 128.
  - Adam with `α=2×10⁻⁴`, `β₁=0.5`, `β₂=0.999`, `ε=1×10⁻⁸`, 50 epochs.
  - Alternate updates: freeze `w_D` while updating `w_G` to fool the discriminator, then unfreeze and train `w_D` on mixed real/fake batches.
- Resulting samples reproduce acceleration/deceleration and terminal corrections; however, RF-based detectors still reach >99% accuracy against them, indicating detectable artifacts.

## Detection & Training Framework
- Per-trajectory processing: extract neuromotor (± global) features, then classify.
- Individual classifiers evaluated:
  - Traditional ML: SVM (RBF), Random Forest (RF), KNN (k=10), MLP.
  - Sequence models: LSTM and GRU using raw `{x, y}` series (architecturally matched to GAN discriminator).
  - GAN discriminator reused as a classifier (fine-tuned with various hidden sizes).
- Training splits: stratified 70/30 train/test with equal human vs synthetic samples (`L = L_h + L_s`, `L_h = L_s`); experiments repeated 5× (σ ≈ 0.1%).
- Directional binary tasks: eight models (one per trajectory direction) trained with RFs to gauge sensitivity to direction/length.
- Multi-attack scenario: classifiers trained on all 10 attack types (9 function-based + GAN) across all directions.
- One-class vs multi-class study: One-Class SVM trained only on human traces versus multiclass SVM trained on human+synthetic data.
- Sample-efficiency study: evaluate accuracy as training size `L` varies from 100 to 7,000 (balanced real/fake).

## Key Findings
- **Direction sensitivity** (RF, per-trajectory):
  - Long trajectories (e.g., 1→2, 2→3) achieve ≥95% accuracy even against VP=3.
  - Short paths (8→1, 6→7, 7→8) are hardest; VP=3 fakes slip through up to 17% of the time.
  - GAN fakes are slightly harder than function-based ones, yet still detected at ≥88% accuracy per direction.
  - Combining neuromotor with legacy global features yields ≈99% accuracy across all synthetic types.
- **Impact of synthetic data**:
  - Training only on real data (One-Class SVM) caps accuracy at 59.9–66.3% depending on features.
  - Mixing real+fake (Multiclass SVM) boosts accuracy to 89.8% (neuromotor) and 98.2% (neuromotor+global), a 95.4% relative error drop vs the Chu et al. baseline.
- **Classifier ablation (multi-attack)**:
  - RF dominates (Function-based: 98.5% acc, 99.8% AUC; GAN: 99.7% acc, 99.9% AUC; Combined: 98.7% acc).
  - SVM is close behind (~98% acc). KNN/MLP trail (~92–94% acc).
  - LSTM/GRU on raw sequences reach ≈98% acc on GAN attacks but underperform RF on combined scenarios.
- **Sample efficiency**:
  - Statistical models (RF/SVM/MLP/KNN) saturate near `L=500`.
  - LSTM/GRU need ≥2,000 samples to match RF, highlighting the usefulness of feature engineering when data are scarce.
- **GAN discriminator reuse**:
  - Large discriminator (LSTM 128/64) attains 99.9% accuracy on GAN attacks and ~90% on unseen function-based fakes despite never training on them.
  - Smaller discriminators (LSTM 16/8) collapse on function-based attacks (<65% acc), emphasizing capacity needs for cross-generator generalization.

## Deployment & Integration Notes
- The detector works on single trajectories; multiple movements can be fused with standard biometric score combination techniques.
- Can be injected transparently alongside image-based CAPTCHAs (monitor cursor path while user solves visual challenge).
- Fusion concept (Fig. 7): combine BeCAPTCHA mouse score with cognitive CAPTCHA result, contextual signals (IP, MAC), and other behavioral channels (keystroke, touchscreen).

## Limitations & Future Directions (from paper)
- Improve neuromotor feature set with derived metrics.
- Hybrid generator: feed function-based trajectories into GAN as priors; deepen generator to increase diversity.
- Extend analysis to touchpads and other interaction devices; leverage larger HCI datasets via transfer learning.
- Behavioral CAPTCHAs are complementary to cognitive schemes, not replacements.

