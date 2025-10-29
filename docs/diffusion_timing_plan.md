# Diffusion + Timing Factorization Plan

> **Purpose**: Define the data flow, model components, and integration steps for splitting spatial diffusion (Δx, Δy) from the timing process (Δt). This prevents normalization drift, keeps evaluation honest, and gives us a roadmap before coding.

## 1. Objectives & Constraints

- **Geometry diffusion** should stay identical to the current setup (normalized Δx/Δy, fixed length, masked loss), so existing checkpoints remain usable.
- **Timing model** must reproduce the empirical distribution of gesture durations and per-step Δt while staying differentiable (for future joint training) and lightweight (a single 3090 box).
- **Evaluation** must never mix representations. All metrics requiring Δt (vel/acc/jerk) must use the timing model outputs; geometry-only metrics should ignore Δt entirely.
- **Deterministic reproducibility**: cache stats, random seeds, and config knobs so short tests (≤5 epochs) and sweeps give consistent timing behavior.

## 2. Data Processing Pipeline

### 2.1 Geometry stream (existing)

1. Load gestures via `GestureDataset` with `normalize_sequences=true`, `canonicalize_path=false`, `canonicalize_duration=false` (matches current exp_short_cosine).
2. Keep only the first two channels (Δx, Δy) for diffusion; enforce fixed sequence length `T` via padding.
3. Normalization: per-channel mean/std from **train positives only** (already cached). Clamp std floor at `1e-3` to prevent tiny scales.
4. Dataloader emits `(Δx,Δy)` + mask + conditioning features; Δt is dropped at this stage.

### 2.2 Timing stream (new)

1. For every sample in the same dataset splits, cache:
   - **Total duration** `D = Σ Δt` (seconds).
   - **Normalized profile** `w = Δt / D` (shape `[T]`, padded with zeros, mask-aware).
   - Optional conditioning features (path length, curvature, octant, user/device id hashes).
2. Store summary stats:
   - Log-duration mean/std (`μ_D`, `σ_D`).
   - Mean normalized profile `E[w]` and concentration proxy (variance per step) for later Dirichlet fits.
3. Persist timing cache alongside existing dataset metadata (`datasets/<id>/timing_cache.pt`) so every run reuses the same stats.

## 3. Model Components

### 3.1 Geometry Diffusion (unchanged)

- `UNet1D(in_channels=2, out_channels=2)` predicting ε or v.
- Objective, schedule, EMA, diagnostics remain untouched.
- Output: normalized deltas `Δx̂, Δŷ` of shape `[B,T,2]`.

### 3.2 Timing Head (new)

Two lightweight subnetworks sharing the same conditioner (e.g., 2-layer MLP on neuromotor + simple metadata):

1. **Duration head**
   - Outputs `μ_D`, `logσ_D` for lognormal total duration.
   - Loss: negative log-likelihood on real `D` (masked to positives only).

2. **Profile head**
   - Outputs Dirichlet concentration vector `α ∈ ℝ^{T}` via `softplus` (add ε).
   - Optional low-rank parameterization: predict base template of length `K` (e.g., 16) and upsample to `T` to keep params small.
   - Loss: Dirichlet NLL against real normalized `w` (ignore padded steps via mask).

Both heads train from the same dataloader (can share batches with diffusion to keep caches hot) but are optimized separately (e.g., distinct AdamW). Later we can multi-task them with the UNet if desired.

## 4. Sampling & Integration Flow

1. **Sample geometry**: run diffusion sampler → `Δx̂, Δŷ` (normalized). Denormalize with dataset stats.
2. **Sample timing**:
   - Feed conditioning vector into duration/profile heads (EMA weights) → `(μ_D, σ_D, α)`.
   - Draw `D ~ LogNormal(μ_D, σ_D)`.
   - Draw `w ~ Dirichlet(α)` (or use mean profile if deterministic run desired) → `Δt = D · w`.
3. **Assemble sequence**: concat `[Δx̂, Δŷ, Δt]`, re-normalize if downstream code expects normalized tensors (store separate mean/std for Δt).
4. **Evaluation hooks** (C2ST, sample stats, GAN diffusion eval) **must call** this assembly helper before computing any Δt-aware feature. Geometry-only metrics can operate on `[Δx,Δy]` directly.

## 5. Configuration & Experiments

- Create `conf/diffusion/exp_factored_timing.yaml`:
  - `model.in_channels=2`, `training.objective=epsilon` (or `v`).
  - New `timing` block: `enabled`, `head_hidden`, `dirichlet_rank`, `loss_weights`, `cache_path`.
  - Add CLI flag `+timing.eval_assign=true` to control whether eval attaches Δt (useful for ablations).
- Update `scripts/run_diffusion_short.py` to accept `--timing-enabled` override for quick smoke tests.

## 6. Testing Strategy

1. **Unit tests** (`tests/test_timing_model.py`):
   - Dirichlet head reproduces constant profiles when fed uniform α.
   - Duration head recovers known μ/σ on synthetic lognormal samples.
2. **Integration test**:
   - Feed identical geometry samples through the assembly helper with cached timing → C2ST between “real” and “fake” (same tensors) ≈0.5, even for Δt-aware features.
3. **Regression script**: extend `scripts/diffusion_x0_diagnostics.py` to report duration/profile stats of generated samples vs real cache.

## 7. Migration Steps

1. Implement timing cache builder (`scripts/build_timing_cache.py`) to precompute `D`, `w`, and template stats for every dataset split.
2. Add timing module (`src/timing/model.py`) + trainer (`src/timing/train.py`) sharing Hydra config with diffusion.
3. Hook timing assembly into diffusion evaluation pipeline via a single helper (`compose_spatiotemporal_sequence(geom, cond, sampler, timing_cfg)`). Replace all ad-hoc Δt padding/clamping with this call.
4. Run short experiment (`exp_factored_timing`, 5 epochs) logging:
   - Geometry stats (existing normalized stds).
   - Timing stats (mean duration, Δt median, |vel| p95) to confirm match.
5. Once validated, update `docs/diffusion_sanity_plan.md` to mark timing symmetry as verified and describe the new pipeline.

## 8. Open Questions

- Do we condition timing on the same vector as diffusion (neuromotor stats, start→goal), or do we need extra context (device, user)?
- Should the Dirichlet head share embeddings with the UNet (joint training) or remain independent for now?
- How do we handle sequences shorter than `T` when sampling `w`? (Proposal: sample Dirichlet of length `T`, then zero out masked tail and renormalize.)

Answer these before implementation to avoid another normalization mismatch.

