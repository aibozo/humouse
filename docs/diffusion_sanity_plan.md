# Diffusion Sanity Checklist

> **Purpose.** Track every data → model → data transformation the diffusion pipeline performs and record how we verify (or plan to verify) each step. Work across sessions by checking boxes and adding notes as verifications land. Always keep real/fake symmetry in mind.

## Legend
- ✅ Verified recently
- ⚠️ Needs verification
- ❓ Unknown / not started

---
## 1. Dataset ingestion & caching
| Step | Status | Verification method | Notes |
| --- | --- | --- | --- |
| Raw loaders (Balabit, etc.) feed GestureDataset correctly (split, user filters, max gestures). | ✅ | Spot-check `len(dataset.samples)` vs expected counts; ensure `max_*` overrides honored. | 2025‑10‑26: instantiated diffusion dataset via `DiffusionDataConfig` → train=84 462 samples, val capped at 256; confirms split/max routing post-cache purge. |
| Normalization stats computed from **positives only** (train split). | ✅ | Inspect `dataset._sequence_mean/std` and verify no negatives in stats. | `build_diffusion_dataset` forces `use_generated_negatives=False`, `_compute_statistics` iterates only `_positive_sequences`; script confirmed all labels==1.0 so stats exclude synthetics. |
| Canonicalization flags (path/duration) match GAN baseline (currently disabled). | ✅ | Config diff vs GAN; cache purge ensured new settings. | `data.canonicalize_* = false` in base config and caches regenerated. |
| Replay / generated negatives disabled for diffusion training datasets. | ✅ | Confirm `use_generated_negatives=False` and replay path unset. | Diffusion dataset builder hard-codes `use_generated_negatives=False`, observed labels are all ones and no replay path configured. |

## 2. Batch construction & augmentations
| Step | Status | Verification method | Notes |
| --- | --- | --- | --- |
| Collator produces tensors [B,T,C], masks consistent with padding. | ✅ | Add unit test dumping a batch, check mask sums. | `diffusion_collate_fn` (`src/diffusion/data.py`) stacks tensors + `infer_mask_from_deltas`; runtime logs show mask_sum=8192 for 32×64 batches under exp_short_cosine. |
| Augmentations (time_stretch, jitter, mirror) applied symmetrically to Δx/Δy and inverted before metrics. | ⚠️ | For each augmentation, confirm inverse transform exists or disable for eval runs. | Recently disabled for exp_short_cosine. Need doc for when they’re enabled. |
| Δt channel handling: if `C=2`, Δt inserted consistently for metrics, plots, C2ST. | ✅ | Audit all code paths that expect Δt (C2ST, plotting) and ensure fake uses same Δt stats as real. | `_compute_sample_stats` + diffusion eval now borrow Δt via `match_time_channel`, so summaries/plots consume real timing traces. |

## 3. Diffusion objective & schedule
| Step | Status | Verification method | Notes |
| --- | --- | --- | --- |
| `q_sample` uses √ᾱ and √(1-ᾱ) consistently. | ✅ | schedule.py audit + math review (2025‑10‑26). |  |
| Target computation (ε or v) matches training objective config. | ✅ | Logging of `target_std≈1`; objective toggle working. |  |
| Min-SNR weighting (if enabled) matches log-snr formula. | ✅ | Unit test `_min_snr_weight` vs reference formula. | `tests/test_diffusion_training.py::test_min_snr_weight_matches_reference` covers both γ>0 and γ=0. |
| Self-conditioning pipeline (teacher forward, detach) works and toggled intentionally. | ✅ | Add smoke test verifying self_cond flag changes input dim. | `tests/test_diffusion_sampling.py::test_unet_self_conditioning_changes_input` asserts doubled input channels + different outputs when conditioning changes. |

## 4. Training loop
| Step | Status | Verification method | Notes |
| --- | --- | --- | --- |
| Loss masking excludes padded steps only (no leakage). | ✅ | `masked_mse` audit + log mask sums. | Logging shows mask_sum=8192 (expected). |
| Optimizer/EMA updates (grad scaler, clip) happen every step. | ✅ | code audit; grads clip when enabled. |  |
| Validation uses EMA shadow, same normalization stats as training. | ✅ | evaluate() renormalizes via training dataset. |  |
| Logging/summary (val loss, C2ST, feature deltas, sample stds) recorded per run. | ✅ | `training.summary_path` writing JSON. |  |

## 5. Sampling path (DDIM)
| Step | Status | Verification method | Notes |
| --- | --- | --- | --- |
| Sampler respects objective (ε vs v) with `x0_from_eps` / `x0_from_v`. | ✅ | Added objective flag; tests run (2025‑10‑26). |  |
| `load_sampler_from_checkpoint` pulls objective from checkpoint (or override). | ✅ | Config payload includes `training.objective`. |  |
| Δt reinsertion for downstream metrics: C2ST uses val Δt; sample stats/plots still pad constant. | ✅ | Update `_compute_sample_stats` and plotting helper to borrow real Δt. | Shared `match_time_channel` helper now feeds both sample stats and GAN diffusion eval outputs, so downstream PNGs inherit realistic Δt. |

## 6. Evaluation metrics
| Step | Status | Verification method | Notes |
| --- | --- | --- | --- |
| C2ST real/fake pipelines fully symmetric (same denorm, Δt, feature normalization). | ✅ | Need unit test comparing features when feeding identical tensors. | `tests/test_diffusion_training.py::test_diffusion_classifier_metrics_real_equals_fake` drives identical batches and observes accuracy/AUC ≈0.5. |
| Feature library (neuromotor) expects Δt channel; confirm when C=2 we inject constant. | ⚠️ | Document at call sites; add helper to synthesize Δt. | |
| Sample variance logging compares to real stats to detect scale drift. | ✅ | `_compute_sample_stats` records norm/denorm stds. | |
| Plots / analytics (analysis/diffusion_fake_vs_real.png) rely on cumulative sum of Δx/Δy. | ✅ | Added script; visually confirms discrepancy. | |

## 7. Search & diagnostics
| Step | Status | Verification method | Notes |
| --- | --- | --- | --- |
| Optuna harness runs trials, saves params + summary, prunes failures. | ✅ | `scripts/search_diffusion.py` executed 8 trials. | Many trials still fail early; need more robust camp. |
| Summary JSON schema stable (val loss, scale gap, C2ST, sample std). | ✅ | `outputs/search_trials/trial_XXX/summary.json`. | |
| Additional diagnostics (x̂0 recon accuracy, per-step stats) | ✅ | Add script to run EMA on val batch, log x̂0 vs x0 stats. | `scripts/diffusion_x0_diagnostics.py` loads a checkpoint and reports avg MSE/MAE + std ratios across validation batches. |

## 8. Open questions / TODOs
- [x] Recompute/verify normalization stats use only positive real samples (exclude generated negs) and confirm denorm numbers (2025‑10‑26 script run, labels all ones).
- [x] Enforce Δt symmetry everywhere (plots, sample stats, evaluation). Document expected behavior when `in_channels=2` vs 3.
- [x] Add unit test for sampler + evaluator verifying that real input → C2ST≈0.5 (i.e., feeding real as fake should confuse classifier).
- [ ] Investigate x̂0 variance directly: capture model outputs mid-training to see if network inherently explodes scale.
- [ ] Consider scale regularizer or explicit constraint if model keeps drifting (tie into Optuna search).

## References & quick links
- [Diffusion D-Eval spec](docs/diffusion-d-eval-spec.md)
- [Task scratchpad](docs/task_scratchpad.md)
- [Search harness](scripts/search_diffusion.py)
