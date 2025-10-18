# Neuromotor Feature Reference

This document lists the neuromotor features to compute per gesture along with definitions and implementation notes. Use it to maintain consistency between preprocessing, detector training, and evaluation.

## Feature List (implemented)
- **duration** — gesture duration in seconds.
- **path_length** — cumulative path length (normalised units).
- **avg_velocity** — mean speed magnitude.
- **peak_velocity** — maximum speed magnitude.
- **velocity_std** — speed standard deviation.
- **acc_mean / acc_std** — mean and standard deviation of acceleration magnitude (first derivative of speed).
- **jerk_mean / jerk_std** — mean and standard deviation of jerk magnitude (second derivative of speed).
- **curvature_mean / curvature_std** — mean and standard deviation of curvature (see implementation for details).
- **path_efficiency** — euclidean displacement divided by path length.
- **direction_changes** — number of sign flips in x/y differentials.
- **time_to_peak_velocity** — fraction of total duration needed to reach peak velocity.
- **idle_ratio** — fraction of samples with speed below 5% of peak speed.

Implemented in `src/features/neuromotor.py` with smoothing via Savitzky–Golay filters (`savgol_filter`).

## Additional Candidates (not yet implemented)
- Overshoot count relative to target region.
- Spectral features (centroid, entropy, dominant frequency of Δx/Δy).
- Session-level aggregates (mean/std of per-gesture metrics).
- Fitts’ law residuals when target metadata available.

Contributions welcome—extend `FEATURE_SPECS` and add unit tests.

## Sigma-Lognormal Feature Vector (BeCAPTCHA-Mouse)
- Each gesture is decomposed into lognormal strokes as described in [Plamondon 1995](https://doi.org/10.1007/BF00201464) and Appendix B of BeCAPTCHA-Mouse (Table 2).
- For every stroke we retain the six core parameters: path distance, onset time `t₀`, lognormal location `μ`, lognormal spread `σ`, initial angle, and final angle.
- Gestures are split at half the total duration. For each parameter we record the maximum, minimum, and mean over the first half and repeat for the second half (6 statistics × 6 parameters = 36 values).
- The final feature appends the stroke count `N`, yielding the 37-dimensional vector the paper reports.
- Implemented in `src/features/sigma_lognormal.py` via `_split_strokes_by_time` and `_stats`; make sure gestures are analysed in raw pixel/time space before any unit-path or unit-duration canonicalisation when reproducing BeCAPTCHA experiments.
- The paper-faithful pipeline resamples raw events to a fixed 200 Hz grid (`data.sampling_rate=200`) before feature extraction so Δt is constant (5 ms) across real and generated gestures.

## Implementation Notes
- Input `GestureSequence` expected from `src/data/segmenter.py` (resampled sequences with mask).
- Finite differences use safe denominators (`dt` clamped at 1e-6).
- Smoothing automatically adjusts window length for short gestures.
- Direction change count currently sums sign flips on Δx and Δy; refine if using polar direction bins.
- Feature vectors returned as `float32`; batch helper `compute_feature_matrix` stacks vectors for detector training.
