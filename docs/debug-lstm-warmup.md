# Debug Review: LSTM Warm-up Collapse in BeCAPTCHA GAN

This document summarizes the detailed investigation of your LSTM-based generator warm-up collapse in the `humouse` project.

---

## 🧠 Root Cause Summary

### 1. **Loss space mismatch (absolute vs normalized)**
The reconstruction warm-up computes loss in **absolute XY space**, but your generator outputs are passed through a `tanh` activation.  
→ The cumulative positions (x, y) can exceed [-1, 1], causing **gradient saturation** and making the network appear to "not learn".

### 2. **Repeated input across time**
Your LSTM generator repeats the same latent vector for each timestep, so without time conditioning, it struggles to form complex sequences.  
→ The network often learns a constant flat output.

### 3. **Latent whitening + noise during warm-up**
The whitening and Gaussian noise in warm-up further destabilize learning. They can be re-enabled later but should be off initially.

---

## 🧩 Evidence in Code

- `_prepare_targets()` converts deltas → cumulative positions when `absolute_coordinates=True`  
- Generator XY head uses `tanh`  
- Warm-up loss: `L1Loss(reconstructed, target_sequences)`  
- Pretraining script uses **normalized positions** and **linear XY head**, not `tanh`

---

## ✅ Fix A — Train in normalized position space (recommended)

### Config adjustments
```yaml
model:
  architecture: lstm
  generator:
    output_dim: 2
    activation_xy: linear
training:
  absolute_coordinates: true
```

### Warm-up logic patch
Normalize positions before L1 loss:
```python
if absolute_coordinates:
    pos_mean, pos_std = _compute_position_stats_from_loader(dataloader)
    target_norm = (target_sequences - pos_mean) / (pos_std + 1e-6)
    preds = generator(latent, condition)
    loss = criterion(preds, target_norm) * loss_weight
else:
    preds = generator(latent, condition)
    loss = criterion(preds, target_sequences) * loss_weight
```

Helper for position stats:
```python
def _compute_position_stats_from_loader(dl, limit_batches=200):
    sum_vec, sum_sq, count = torch.zeros(2), torch.zeros(2), 0
    for bidx, (_, seq, _) in enumerate(dl):
        if bidx >= limit_batches: break
        pos = torch.cumsum(seq[..., :2], dim=1).double()
        sum_vec += pos.sum((0,1)); sum_sq += (pos**2).sum((0,1))
        count += pos.shape[0] * pos.shape[1]
    mean = (sum_vec / count).float().view(1,1,2)
    std = ((sum_sq / count - mean.double().view(2)**2).clamp_min(1e-6).sqrt().float().view(1,1,2))
    return mean, std
```

Disable whitening + noise until loss stabilizes.

---

## 🧮 Fix B — Stay in delta space
If keeping `tanh`, compute L1 in **delta space** instead of positions:
```python
target_deltas = real_sequences[..., :2]
pred_deltas = generator(latent, condition)[..., :2]
loss = criterion(pred_deltas, target_deltas) * loss_weight
```

---

## 🔧 Verification Steps
1. Add quick printouts:
```python
print("target std", target_sequences.std())
print("pred std", reconstructed.std())
```
→ If targets >> 1 and preds << 1 → gradient saturation.

2. Train for 1–2 epochs — loss should now visibly decrease.

---

## 🧭 Optional Enhancements
- Add `[0..1]` time ramp to condition vector
- Warm-up using embedding-table latent pretraining (like in pretrainer script)
- Skip warm-up entirely by loading a pretrainer checkpoint

---

## 📄 References
- `src/train/train_gan.py` – warm-up logic
- `src/models/gan_lstm.py` – LSTM generator implementation
- `scripts/pretrain_lstm_generator.py` – normalized training routine
- `docs/architecture_gan.md` – paper-aligned config recommendations

---

This version is ready to drop in as `docs/debug-lstm-warmup.md`.
