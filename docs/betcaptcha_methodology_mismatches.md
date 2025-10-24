# BeCAPTCHA-Mouse Methodology Gaps

This note records where the current implementation diverges from the methodology captured in `docs/betcaptcha_mouse_methodology_compact.md` (itself a condensed summary of the BeCAPTCHA-Mouse paper). Each item lists the paper expectation, the observed repository behaviour, and a status flag.

## System Intent
- **Overall goal** — *Status: match*  
  Paper and repository both seek to generate human-like mouse trajectories and train detectors that spot bots. No action needed.

## Benchmark & Data Preparation
- **Data source / task format** — *Status: mismatch*  
  Paper uses the Shen et al. dataset (58 users performing an 8-target click task; trajectories are between consecutive clicks). The project instead defaults to `bogazici` data with velocity/gap-based segmentation (`conf/experiment/train_gan_paper.yaml:5-11`, `src/data/segmenter.py:200-286`), so gestures are not tied to fixed click targets or balanced direction bins.
- **Sample counts & splits** — *Status: mismatch*  
  The BeCAPTCHA benchmark releases 5 k real + 10 k synthetic trajectories with 35 real samples per user. Our configs cap real gestures at 4 096 without ensuring per-user balancing (`conf/experiment/train_gan_paper.yaml:10`), and no public synthetic bundle is produced.
- **Resampling cadence** — *Status: mismatch*  
  Paper resamples/generates at 200 Hz. Repo configurations run at 100 Hz (`conf/experiment/train_gan_paper.yaml:7`), and the segmenter inherits that rate, yielding 10 ms spacing instead of the paper’s 5 ms.
- **Sequence length policy** — *Status: mismatch*  
  Paper samples the number of points `M` per trajectory from a Gaussian per direction. We fix `sequence_length: 200` for all samples (`conf/experiment/train_gan_paper.yaml:6`), so variability in `M` is lost.
- **Trajectory unit definition** — *Status: mismatch*  
  Paper defines gestures strictly as the movement between two clicks. Our segmentation is based on temporal gaps or sustained low velocity (`src/data/segmenter.py:200-286`), which can combine multiple intent segments or break mid-gesture.
- **Conditioning features** — *Status: mismatch*  
  Paper trains GANs on raw trajectories without conditioning. Implementation appends neuromotor + goal geometry features (37 + 5 dims) and feeds them into both generator and discriminator (`conf/experiment/train_gan_paper.yaml:24-35`, `src/data/dataset.py:62-125`).

## Neuromotor Feature Pipeline
- **Feature vector contents** — *Status: partial match*  
  We reproduce the 37 sigma-lognormal statistics (`src/features/sigma_lognormal.py:59-138`), but we also append five geometry features when `include_goal_geometry` is true (`src/data/dataset.py:62-125`), which deviates from the paper’s pure neuromotor vector.
- **Baseline feature fusion** — *Status: mismatch*  
  Paper reports best results when combining neuromotor and global features (duration, distance, etc.). Our GAN configuration uses sigma-lognormal features only; the Chu et al. global features are not concatenated in this path.

## Synthetic Trajectory Generators
- **Function-based synthesiser** — *Status: partial match*  
  Shapes (linear/quadratic/exponential) and velocity profiles (constant/logarithmic/Gaussian) are implemented (`src/eval/sigma_log_baseline.py:34-210`), but the generator pulls displacement statistics from the Boğaziçi data and enforces a fixed sequence length unless manually disabled. The paper samples duration/length per direction from the Shen benchmark and varies `M`.
- **GAN architecture inputs** — *Status: mismatch*  
  Paper’s GAN uses latent noise only. Our generator consumes latent noise + 42-dim condition vectors (`conf/experiment/train_gan_paper.yaml:23-35`, `src/train/train_gan.py:480-508`), altering the learning problem.
- **Generator topology** — *Status: mismatch*  
  Paper: two LSTM layers (128 + 64 units) with direct `{x̂, ŷ}` output. Repo: autoregressive seq2seq decoder with uniform 128-unit layers and an encoder-decoder warm-up (`src/models/seq2seq.py:23-112`, `src/train/train_gan.py:700-832`). There is no 64-unit layer.
- **Discriminator topology** — *Status: mismatch*  
  Paper: LSTM(128) → LSTM(64) → dense sigmoid, no conditioning. Repo: LSTM stack with identical 128-unit layers plus feature conditioning (`src/models/gan_lstm.py:33-82`, `conf/experiment/train_gan_paper.yaml:33-36`).
- **Training warm-up** — *Status: mismatch*  
  We run a 10‑epoch reconstruction warm-up with an auxiliary encoder and KL term before adversarial updates (`conf/experiment/train_gan_paper.yaml:56`, `src/train/train_gan.py:700-832`). Paper trains the GAN directly without such pretraining.
- **Learning rates & tricks** — *Status: mismatch*  
  Paper uses Adam with α = 2×10⁻⁴ for both networks (β₁=0.5, β₂=0.999). Repo sets generator lr = 2×10⁻⁴ and discriminator lr = 2×10⁻⁵ plus label smoothing and R1 regularisation (`conf/experiment/train_gan_paper.yaml:41-44`, `76-81`), which diverges materially.
- **Absolute coordinates & dt** — *Status: partial match*  
  Both approaches model absolute `{x, y}` at a fixed dt; however, our dt is 0.01 s (100 Hz) instead of the paper’s 0.005 s (200 Hz).

## Detection & Training Framework
- **Classifier family** — *Status: mismatch*  
  Paper evaluates RF, SVM, KNN, MLP, LSTM, GRU on per-trajectory features (including per-direction models). Repo’s primary detector is a fusion network with TCN + MLP branches (`src/models/detector.py`, `conf/experiment/train_detector.yaml:12-24`), and we lack scripted per-direction RF experiments.
- **Synthetic data usage in detector** — *Status: mismatch*  
  Paper alternates training with real + synthetic samples generated on the fly. Detector training here relies on a replay buffer path (`conf/experiment/train_detector.yaml:12-17`) but does not orchestrate the specific ten attack types or directional splits described in the paper.
- **One-class vs multi-class SVM study** — *Status: missing*  
  No implementation reproduces the One-Class vs Multi-Class SVM comparison from Table 4.
- **Sample-efficiency curves** — *Status: missing*  
  There is no automation for the L vs accuracy sweeps reported (Fig. 6). Scripts focus on GAN sigma evaluations instead.

## Key Findings & Evaluation
- **Directional accuracy table** — *Status: missing*  
  No notebook or script reproduces the per-direction accuracy table (Table 3). Sigma evaluation hooks aggregate accuracy across files without isolating directions (`src/train/train_gan.py:556-640`).
- **Average accuracy numbers** — *Status: missing*  
  We lack recorded metrics showing the 98–99 % accuracy figures or 95.4 % relative error reduction cited. Current evaluation logs (`analysis/` plots, `checkpoints/*/metrics.csv`) do not document such results.
- **GAN discriminator reuse study** — *Status: missing*  
  Paper benchmarks multiple discriminator layer sizes (Table 6). Repo does not expose configuration sweeps for discriminator depth or reuse-as-classifier experiments.

## Deployment & Integration
- **CAPTCHA fusion diagram** — *Status: N/A*  
  Paper illustrates integration with image CAPTCHAs. Repo has logging scaffolding (W&B, CSV) but no end-to-end CAPTCHA fusion prototype; this is outside current scope.

## Future Work Hooks
- **Paper-suggested improvements** — *Status: partially tracked*  
  Some items (e.g., hybrid generators, richer features) appear in `docs/task_scratchpad.md`, but others (touchpad extension, transfer learning) are not planned explicitly.

---
Use this checklist when aligning the implementation with the BeCAPTCHA methodology; clear items once the repository matches the cited behaviour.

