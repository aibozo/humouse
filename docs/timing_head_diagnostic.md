# Timing Head End-to-End Diagnostic Plan

Purpose: isolate why the duration/profile head fails to match real Δt distributions before touching loss weights. Follow these steps sequentially; log findings in the scratchpad after each stage.

## 1. Cache Integrity
- **Action:** rerun `scripts/build_timing_cache.py` and verify `datasets/<id>/<split>_timing.pt` stats match raw dataset summaries (duration mean/std, profile mean).
- **Checks:** no negatives; mask sums align with sequence length; profile mean integrates to ~1.0.

## 2. Dataloader / Dataset
- **Action:** instantiate `TimingDataset` and inspect a minibatch (`torch.mean(batch['profile'], dim=0)`).
- **Checks:** features tensor matches conditioning dimension; masks correctly zero padded steps; log-duration equals `log(duration)`.

## 3. Model Architecture Review
- Confirm backbone receives the same conditioning vector used in diffusion (neuromotor features). If base features are only Δt stats, consider augmenting with path length/direction metadata.
- Evaluate whether direct per-step linear logits can represent sharply decaying profiles; consider low-rank templates or monotonic parameterizations if not.
- Ensure duration head outputs log-parameters (μ, log σ) without clipping true variance.

## 4. Training Loop Verification
- **Action:** enable the new `_log_stats` path (already logs duration μ/σ and first/mid/last profile means each epoch).
- **Checks:** training vs validation loss trending; logged stats converging toward cache stats; detect collapse (flat profiles) early.

## 5. Sampler Diagnostics
- Use `scripts/compare_timing_sampler.py --checkpoint ... --cache ... --samples 2048` after each training run.
- Compare duration quantiles and profile mean (first/mid/last) against cache output; expect first bin ≈0.12, mid ≈5e-4, tail ≈2e-5 for Balabit.

## 6. Integration Hooks
- In diffusion eval, assert that when timing sampler is enabled, FAKE Δt stats (median, |vel| p95) stay within ±10% of real stats before C2ST is computed.

## 7. Decision Points Before Tuning
- If cache/dataloader look correct but sampler remains flat, revisit architecture (e.g., predict logit increments + cumulative normalization).
- If stats match in the standalone sampler but drift during diffusion eval, inspect normalization (are sequences re-normalized after assigning Δt?).
- Only adjust loss weights/temperatures after structural issues above are resolved.

