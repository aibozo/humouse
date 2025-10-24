# Trajectory Diffusion for Mouse Movements — Build & D‑Eval Integration

**Audience:** coding agents implementing a diffusion generator for 2D cursor trajectories and wiring it into the existing GAN/Detector pipeline for evaluation and hard‑negative generation.

**Why diffusion here?** GANs can look sharp but often **miss rare modes** (mode collapse). Diffusion learns to **denoise** corrupted data at many noise levels, which supplies gradients **across the whole distribution**, typically yielding **better coverage** (more of the real data variety). We’ll keep the model small so it trains on a single 3090.

---

## 0) Deliverables (high level)

1. **`diffusion/` package** with:
   - `models/unet1d.py` — 1D UNet over time with time‑step embeddings, optional self‑conditioning, optional conditioning FiLM.
   - `noise/schedule.py` — cosine/log‑SNR schedules + utilities to go `x0 ↔ xt` and sample ε or v.
   - `train.py` — DDPM/DDIM training loop (v‑prediction; masked loss).
   - `sample.py` — fast samplers (DDIM 20 steps; DPM‑Solver optional).
   - `latents/autoencoder.py` (optional) — tiny temporal autoencoder if we do latent diffusion.
2. **Eval hooks** callable from existing code:
   - `generate_diffusion_samples(n, seq_len, cond=None) -> torch.Tensor[B,T,C]` (normalized deltas).
   - Metrics: `c2st(features_real, features_fake)`, `mmd(features_real, features_fake)`, `prd(features_real, features_fake)`.
   - Kinematic & spectral summaries: jerk, path‑efficiency, heading‑change, lateral RMS, velocity spectrum.
3. **Hydra configs** under `configs/diffusion/*.yaml` and CLI entrypoints.
4. **Integration points** in the current detector/GAN eval harness to use diffusion samples as negatives and for distribution matching checks.

---

## 1) Data representation & invariances

**Input tensor**: trajectories as **deltas** `Δx, Δy[, Δt]` with shape **[B, T, C]**, `C=2` (or 3 if we keep Δt).

**Invariances (recommended):**
- **Recenter**: start at (0,0) by subtracting cumulative position at t=0.
- **Canonical rotation**: rotate each trajectory so **net displacement** points along **+x**; store the rotation angle, and rotate back for visualization.
- **Timing**: Prefer **fixed Δt** during modeling (`Δt = 1/T`); keep a separate variant if you truly need variable timing.

**Normalization**: per‑channel standardization over the **train split**: `z = (x - mean) / std` (store `mean/std` for de/normalization).

**Masking**: Pad to `T_max`; keep `mask[B,T]` (1=valid, 0=pad). Losses must ignore padded steps.

**Augmentations (data‑efficient):**
- time‑stretch ±10–20% (resample in time), tiny local time‑warp
- micro‑jitter on deltas (Gaussian noise with small σ in normalized units)
- left–right mirror (flip y)
- sub‑trajectory crop (random contiguous segment → then rescale duration/path to unit)
- small start/goal perturbation if conditioning on goal

---

## 2) Model spec (small 1D UNet over time)

**UNet1D(T,Cin→Cout)**
- **Down path** (4 stages): 2×(Conv1d → GN → SiLU → Dropout 0.1), stride‑2 downsample between stages
  - widths: 128 → 192 → 256 → 256
- **Bottleneck**: same block + **self‑attention** over time (multi‑head, d_model=256, 4 heads)
- **Up path**: mirror with skip connections; ConvTranspose1d or nearest‑upsample+conv
- **Time‑step embedding**: sinusoidal/Fourier `emb(t)` → MLP → FiLM (scale/shift) each block
- **Self‑conditioning (optional)**: concat previous prediction `x̂0` channel‑wise to input during training with prob 0.5
- **Conditioning vector (optional)**: project cond→width and add via FiLM to each block (can be neuromotor/sigma features, start→goal, total path length)
- **Output channels**: `Cout = C` (predict ε or v with the same dimensionality as input)
- **Param budget**: ~1–3M params (fits easily on 3090)

**Notes**
- Use **GroupNorm(32)** (or 16 if channels < 32).  
- Enable **mixed precision** (`amp`) and keep **EMA** of weights (decay 0.999).

---

## 3) Diffusion objective (DDPM, v‑prediction)

At random timestep `t` (0..T−1):
1. Sample noise ε ~ N(0, I).
2. Corrupt clean data `x0` to `xt = ᾱ_t * x0 + σ_t * ε` (use cosine schedule for ᾱ_t, σ_t).
3. Network predicts **v** (velocity) where `v = α_t * ε − σ_t * x0` (better‑conditioned than ε).
4. **Loss**: MSE between predicted `v̂θ(xt, t, cond)` and true `v`, averaged **only over valid steps** (`mask==1`).

**Min‑SNR weighting (optional):** weight each timestep’s loss to emphasize mid‑noise region; reduces overfit at small/high noise.

**Self‑conditioning:** with 50% prob, pass previous `x̂0` from a teacher‑forward as an extra input; improves quality/data‑efficiency.

---

## 4) Training hyperparams (3090‑friendly)

- Optim: **AdamW** (lr **2e‑4**, betas (0.9, 0.999), wd **1e‑2**)
- Batch: 64 (or 32 if T is large), grad‑accum if needed
- **Grad clip**: 1.0
- **EMA**: decay 0.999
- **Epochs**: 50–100 (monitor early‑stop via validation)
- **cudnn.benchmark**: True (shapes stable)
- **amp autocast** + **GradScaler**

**Logging**: recon loss vs steps, validation MMD/PRD/C2ST (see §7), spectra and kinematics overlays every N epochs.

---

## 5) Sampling (fast)

Use **DDIM** (η=0) or **DPM‑Solver** with **~20 steps**:
1. Sample `x_T ~ N(0, I)`
2. For `t = T..1`: compute `x_{t-1}` using `v̂θ` (or ε̂θ) with your schedule
3. At end: de‑normalize, rotate back, **cumsum(Δx, Δy)** to positions for plotting; set `Δt` constant if used.

**Classifier‑free guidance (for conditioning):** duplicate batch with null cond; blend `cond` and `null` predictions with scale `γ≈1.0–2.0` if you need stronger adherence to cond features.

---

## 6) Optional: Latent diffusion

- Train a tiny **temporal autoencoder** (Conv1d↓, stride‑2, 2–3 levels) to compress from T to T/4 with latent channels 8–16.
- Train diffusion **in latent space** (same objective), then decode.
- Benefits: fewer steps/time; costs: extra pretraining stage.

---

## 7) Metrics & how to compute (plain‑English)

> Implement these in `eval/metrics.py`. All operate on **feature vectors per sequence**, not on raw deltas.

- **C2ST (Classifier Two‑Sample Test)**: Fit a tiny probe (logistic regression or 2‑layer MLP) to classify **real vs generated** on a held‑out set of *features* (velocity, acceleration, **jerk** = change of acceleration, heading‑change, path‑efficiency).  
  **Readout**: Accuracy near **50–60%** on **validation** ⇒ distributions are hard to tell apart (good).

- **MMD (Maximum Mean Discrepancy)**: A scalar distance between two sample sets using an RBF kernel.  
  **Readout**: **Lower is better**. Use the median pairwise distance as kernel bandwidth (median heuristic).

- **PRD (Precision–Recall for Distributions)**:  
  - **Precision** ≈ realism (do fakes look valid?)  
  - **Recall** ≈ coverage (do fakes cover the variety?)  
  **Readout**: We want **both high**; GANs often have high precision/low recall. Diffusion should lift recall.

- **Spectral power**: FFT of velocity over time → power at each frequency.  
  **Readout**: Fakes should match the **high‑freq** tail (human micro‑corrections) and mid‑band energy.

- **Kinematic overlays**: histograms/means for **jerk**, **path‑efficiency** (net displacement / path length), **mean absolute heading‑change**, **lateral RMS** in canonical frame.  
  **Readout**: Curves for real vs fake should overlap on **validation**.

Implementation hint: you already have helpers akin to `_path_efficiency`, `_mean_abs_heading_change`, `_lateral_rms`, `_jerk_magnitude` in the GAN code; reuse/port them.

---

## 8) Integration with existing Detector (D‑Eval)

### A) As distribution‑matching evaluator
Add a `--use_diffusion_eval` switch in your current `sigma_eval` / eval harness to **generate N diffusion samples** on the **validation split conditions** and compute:
- `c2st_acc_val`, `mmd_val`, `prd_precision_val`, `prd_recall_val`
- spectra + kinematic overlays (saved plots and numeric summaries)

### B) As hard‑negative generator for detector training
Pipeline:
1. **Generate** M diffusion samples (`M≈#real_val`) with the same length/conditioning distribution as real val.
2. **Assemble negatives** = diffusion fakes **+** GAN fakes **+** simple bots (splines/PID/noise).  
3. **Train detector** (or a small head on D’s penultimate features) on **real vs mixture** negatives.  
4. **Evaluate on unseen bots** (held‑out generator styles) → report ROC‑AUC/F1.

### Function signatures (stable API)
```python
# diffusion/api.py
def generate_diffusion_samples(
    n: int,
    seq_len: int,
    cond: torch.Tensor | None,
    device: torch.device = torch.device("cuda"),
    steps: int = 20,
    seed: int | None = None,
) -> torch.Tensor:
    """Return [n, seq_len, C] normalized deltas (float32)."""

# eval/metrics.py
def c2st(real_feats: torch.Tensor, fake_feats: torch.Tensor) -> float: ...
def mmd_rbf(real_feats: torch.Tensor, fake_feats: torch.Tensor, sigma: float | None = None) -> float: ...
def prd(real_feats: torch.Tensor, fake_feats: torch.Tensor) -> tuple[float, float]:  # precision, recall
```

Wire these into your existing `train_gan.py`/detector eval loop similarly to how `sigma_eval` is triggered.

---

## 9) Training & sampling CLI (Hydra examples)

`configs/diffusion/train.yaml`:
```yaml
experiment_name: "diff_mouse_v1"
seed: 1337

data:
  dataset_id: "humouse_default"
  sequence_length: 200
  batch_size: 64
  num_workers: 8
  canonicalize_path: true
  canonicalize_duration: true
  rotate_to_plus_x: true
  normalize_sequences: true
  feature_mode: "neuromotor"  # for metrics only

model:
  base_channels: 128
  channel_mults: [1, 1.5, 2, 2]
  attn_bottleneck: true
  dropout: 0.1
  self_condition: true
  cond_dim: 0  # set >0 to enable FiLM conditioning

diffusion:
  objective: "v"         # v-pred
  schedule: "cosine"
  timesteps: 1000
  min_snr_weight: true

optim:
  lr: 2e-4
  weight_decay: 1e-2
  betas: [0.9, 0.999]
  grad_clip: 1.0
  ema_decay: 0.999
  epochs: 80
  amp: true

logging:
  log_interval: 50
  eval_interval: 1
  sample_interval: 5
  num_eval_samples: 512
```

**Train:**
```bash
python -m diffusion.train experiment=train
```

**Sample 2k sequences (20 DDIM steps):**
```bash
python -m diffusion.sample   ckpt=checkpoints/diff_mouse_v1.pt   n=2000 steps=20 seq_len=200 out_dir=gen/diff_v1
```

**Eval (plug into existing harness):**
```bash
python -m eval.diffusion_eval   ckpt=checkpoints/diff_mouse_v1.pt   val_split=data/val   metrics=[c2st,mmd,prd,kinematics,spectra]
```

---

## 10) Overfitting controls (small data = 4k trajectories)

- Keep the UNet small (≤3M params), dropout 0.1, weight decay 1e‑2.
- Strong but plausible augmentations (see §1).
- **Self‑conditioning** + **EMA**.
- **User/device‑held‑out splits** for validation/test to catch memorization.
- Watch **C2ST/MMD/PRD on validation**; overfit signals: train improves but val degrades.

---

## 11) Acceptance checklist

- [ ] Training stable (loss decreases; no NaNs)
- [ ] Validation **C2ST ≤ 60%**, **MMD↓** across runs
- [ ] **PRD precision & recall both high** (recall not lagging badly)
- [ ] Kinematic & spectral overlays match validation
- [ ] `generate_diffusion_samples(...)` returns tensors that denormalize and visualize correctly
- [ ] Detector trained on **mixed negatives** reaches strong **ROC‑AUC** on **unseen** bot styles

---

## 12) Minimal training loop (pseudocode)

```python
for step, (x0, mask, cond) in loader:
    x0 = x0.to(device)            # [B,T,C] normalized deltas
    mask = mask.to(device)        # [B,T]
    t = randint(0, T-1, [B]).to(device)

    eps = randn_like(x0)
    alpha_bar_t, sigma_t = schedule(t)
    x_t = alpha_bar_t * x0 + sigma_t * eps

    v_true = alpha_t * eps - sigma_t * x0    # v-objective
    v_pred = model(x_t, t_embed(t), cond, mask, x0_prev=maybe_selfcond)

    loss = mse(v_pred[mask==1], v_true[mask==1])
    if min_snr: loss *= snr_weight(t)

    scaler.scale(loss).backward()
    clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optim); scaler.update(); optim.zero_grad(set_to_none=True)
    ema.update(model)
```

---

### Notes for Integrators
- Keep denorm/rotation logic **identical** to the rest of the codebase, so metrics are apples‑to‑apples.
- When using diffusion samples in **D‑Eval**, make sure real/fake go through the **same canonicalization and normalization** before entering the discriminator or metric features.
- Prefer **GPU‑vectorized** feature computations to avoid CPU sync in the training loop; only move small subsamples to CPU for plots.

---

*End of spec.*
