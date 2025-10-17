# Project Timeline & Milestones

This timeline assumes an initial 12-week planning horizon. Adjust the pace based on dataset availability and compute resources. Weeks refer to consecutive calendar weeks from project kick-off.

## Phase 0 — Repository & Access (Week 0–1)
- Finalise repo scaffolding, documentation, and environment management.
- Secure dataset access and verify licences for Balabit, Boğaziçi, SapiMouse, Attentive Cursor.
- Stand up experiment tracking (Weights & Biases or TensorBoard) and seed configuration templates.

## Phase 1 — Data Pipeline (Week 2–3)
- Implement dataset loaders and gesture segmentation heuristics.
- Build canonical preprocessing (normalisation, resampling/padding) and metadata extraction.
- Implement neuromotor feature computation library and unit tests on sampled sessions.
- Produce baseline processed datasets saved to `data/processed` and document statistics.

## Phase 2 — Detector Baseline (Week 4–5)
- Engineer feature-based detector (XGBoost/MLP) and sequence-based detector (TCN/BiLSTM).
- Create fusion head, training scripts, and evaluation metrics (ROC/PR, calibration).
- Run cross-validation on real-vs-real baselines (cross-user splits) and record performance on holdout sets.

## Phase 3 — Generator & Discriminator (Week 6–8)
- Implement conditional generator (latent + context) with pretraining objective.
- Implement discriminator with auxiliary feature regression and WGAN-GP training loop.
- Validate generator pretrain quality (reconstruction losses, smoothness metrics).

## Phase 4 — Adversarial Coupling (Week 9–10)
- Alternate GAN training with detector hardening; manage generated negative buffers.
- Integrate monitoring dashboards for GAN losses, gradient penalties, detector metrics.
- Conduct ablations on condition inputs and regularisation weights.

## Phase 5 — Evaluation & Reporting (Week 11–12)
- Run comprehensive evaluation suite (realism, robustness, cross-dataset tests).
- Summarise experiment findings, update documentation, and prepare next-step recommendations.
- Identify gaps for stage-2 (session-level) adversarial system.

## Continuous Tasks
- Maintain reproducibility: deterministic seeds, config versioning, dataset hashes.
- Track open issues, blocked items, and risk mitigation in `docs/task_scratchpad.md`.
- Review ethical implications and dataset compliance for any new additions.

