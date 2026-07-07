# Argus Panoptes — Validation Fixes

**Date:** 2026-07-06 (evening, follow-up)
**Trigger:** Post-implementation physics / edge-case test pass (52 pytest + manual validation).

---

## Changes

### 1. `models/baseline.py` — classification on small / imbalanced splits

**Problem:** `classification_report(..., target_names=le.classes_)` crashed when the
test fold omitted a `health_state` class (`ValueError: Number of classes, 3, does not
match size of target_names, 4`). Reproduced on a 20-sample smoke dataset.

**Fix:** Pass explicit `labels=np.arange(n_classes)` to both `classification_report`
and `f1_score(..., average="macro")` so absent classes are reported with
`zero_division=0` instead of raising.

### 2. `dsp/signal_processor.py` — wire `frequency.n_harmonics`

**Problem:** `processor_config.yaml` exposed `n_harmonics` but feature extraction
always integrated 1×, 2×, and 3× TPF bands regardless of config.

**Fix:** Gate 2× and 3× band integration on `self.freq.n_harmonics` (≥2 / ≥3).
Feature column names are unchanged; harmonics above the limit return `0.0`.

### 3. Tests

- `tests/test_baseline.py` — regression test for the classification-report edge case.
- `tests/test_signal_processor.py` — `test_n_harmonics_limits_harmonic_bands`.

---

## Not changed (reviewed, acceptable as-is)

- Vibration clipping at extreme cut params (flagged in metadata).
- Wear applied via both specific-energy and force multiplier (documented design).
- `per_tooth_force_n` naming (effective impulse scale, not literal per-tooth).
- Friction heat term linear in wear only (lumped-model simplification).

---

## Verification

```
pytest tests/ -q          # 54 passed
python models/baseline.py --data-dir data/_test_smoke   # completes (classification OK)
```
