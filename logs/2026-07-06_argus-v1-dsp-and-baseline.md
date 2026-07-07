# Argus Panoptes — Session Change Log

**Date:** 2026-07-06 (evening)
**Scope:** DSP `SignalProcessor` v1, dataset integration, XGBoost baseline +
ablations, new tests, verification. Documentation-only edits are noted in §6.
**Environment:** Python 3.13.9 (Windows / PowerShell). Core deps unchanged
(NumPy, SciPy, Pandas, PyArrow, Pydantic, PyYAML, Matplotlib, tqdm, pytest). New
ML packages installed via the `ml` extra: **scikit-learn 1.7.2**, **xgboost 3.3.0**,
**joblib 1.5.2** (`pip install -e ".[ml]"` equivalent — `xgboost` was the only
missing package and was installed with `python -m pip install "xgboost>=2.1"`).

---

## 1. DSP module implementation (`dsp/`)

### Design decisions
- **Config parity with `sensors/`.** Added `dsp/processor_config.yaml` (units in
  every key, physics rationale inline) validated by pydantic v2 models in
  `dsp/config.py` (`extra="forbid"`), with `load_processor_config(path=None, *,
  overrides=None)` resolving arg → `ARGUS_PROCESSOR_SPECS` env var → packaged
  default and deep-merging overrides — the exact pattern of `sensors/config.py`.
- **Amplitude is a feature, not noise.** `preprocess.normalize` defaults to
  `"none"`. With wear, both the tooth-impact amplitude (rising cutting force) and
  the broadband RMS grow, so per-window normalization would *discard* the primary
  wear signal. Scale-invariant modes (`zscore`/`peak`/`rms`) remain available for
  the Day-3 CNN path. This keeps absolute band energies wear-sensitive.
- **`td_` / `fd_` prefixes** on every feature name so the ML layer selects
  feature groups for ablations by prefix (no hard-coded lists needed).
- **Nyquist-safe band-pass.** The band-pass corners are clamped strictly below
  Nyquist at runtime, so one config serves both the 40.96 kHz vibration path and
  the 200 Hz thermal path; `sosfiltfilt` gives zero phase distortion (preserves
  impact timing).
- **Graceful TPF absence.** TPF-relative features return `NaN` (XGBoost-native)
  when `tooth_pass_freq_hz` is missing; all TPF-independent features still compute.

### Key structure — `__init__` and dispatch
```python
def __init__(self, config=None):
    if config is None:            self.config = load_processor_config()
    elif isinstance(config, ProcessorConfig): self.config = config
    elif isinstance(config, dict): self.config = ProcessorConfig.model_validate(config)
    elif isinstance(config, str):  self.config = load_processor_config(config)
    ...
    self.pre, self.freq, self.stft_cfg = (self.config.preprocess,
                                          self.config.frequency, self.config.stft)
```

### Frequency-feature computation with TPF handling (excerpt)
```python
freqs, psd = sp_signal.welch(x, fs=fs, window=self.freq.window,
                             nperseg=nperseg, noverlap=noverlap)
total_power = max(float(np.trapezoid(psd, freqs)), _EPS)
centroid  = float(np.sum(freqs * psd) / (np.sum(psd) + _EPS))
flatness  = float(np.exp(np.mean(np.log(psd + _EPS))) / (np.mean(psd) + _EPS))
...
if tpf_hz and tpf_hz > 0:
    frac = self.freq.tpf_band_frac
    fund = self._band_energy(freqs, psd, tpf_hz*(1-frac), tpf_hz*(1+frac))
    h2   = self._band_energy(freqs, psd, 2*tpf_hz*(1-frac), min(2*tpf_hz*(1+frac), nyq))
    h3   = self._band_energy(freqs, psd, 3*tpf_hz*(1-frac), min(3*tpf_hz*(1+frac), nyq))
    features["fd_tpf_band_energy"]    = fund      # rises with wear (impact force ↑)
    features["fd_tpf_band_ratio"]     = fund / total_power
    features["fd_harmonic_to_fundamental_ratio"] = (h2 + h3) / fund
```

### Why these bands/features for blade-wear physics
- **TPF fundamental + 2×/3× harmonic bands** — periodic tooth-engagement impacts
  deposit energy at `k·TPF`; per-tooth force (impact amplitude) rises with wear,
  so these band energies are the highest-value wear features.
- **Broadband HF band `[6–18 kHz]`** — stochastic chip formation and edge
  friction grow with wear (`broadband_wear_gain` in the simulator).
- **Impulsiveness factors** (crest / shape / impulse / clearance / margin,
  kurtosis) — classic rotating-machinery health indicators that track transient
  sharpening even after amplitude normalization.
- **Hilbert-envelope mean/std** — amplitude-modulation depth of the strike train.

### Feature inventory (28 vibration scalars)
- **13 time-domain (`td_`):** rms, peak, peak_to_peak, crest_factor, kurtosis,
  skewness, shape_factor, impulse_factor, clearance_factor, margin_factor,
  zero_crossing_rate, envelope_mean, envelope_std.
- **15 frequency-domain (`fd_`):** total_power, spectral_centroid_hz,
  spectral_rolloff_hz, spectral_flatness, spectral_bandwidth_hz, dominant_freq_hz,
  dominant_amplitude, tpf_band_energy, tpf_harmonic2_energy, tpf_harmonic3_energy,
  tpf_harmonic_energy_total, broadband_energy, tpf_band_ratio, broadband_ratio,
  harmonic_to_fundamental_ratio.

### Files
- `dsp/processor_config.yaml` (new), `dsp/config.py` (new), `dsp/signal_processor.py`
  (new — `SignalProcessor`, `__version__="0.1.0"`, `TIME_DOMAIN_FEATURES`,
  `FREQUENCY_DOMAIN_FEATURES`), `dsp/__init__.py` (rewritten to export the API).

---

## 2. Data-pipeline updates (`scripts/generate_dataset.py`)

Additive, opt-in, non-breaking:
- New CLI flag `--extract-features / --no-extract-features`
  (`argparse.BooleanOptionalAction`, **default `False`**) plus `--processor-config`.
- `build_row(..., processor=None)`: when a `SignalProcessor` is supplied, its
  scalar features are merged as `vib_{td,fd}_*` columns and two thermal DSP
  statistics are added (`therm_std_c`, `therm_slope_c_per_s` via `np.polyfit`).
  The base columns/order are **identical** whether or not `processor` is set.
- `generate(..., extract_features=False, processor_config_path=None)`: imports
  `dsp.SignalProcessor` **locally** (default path stays dsp-free) and instantiates
  it only when the flag is on.
- `print_summary`: when feature columns are present, reports feature count and the
  **top-6 |correlations|** of features vs `wear_level`, `quality_score`,
  `cycle_time_factor` via `df[feature_cols].apply(lambda s: s.corr(df[target]))`.
- Verified backward compatibility: a 20-sample run **without** the flag yields a
  45-column manifest with `has DSP feats: False` (unchanged schema).

---

## 3. ML baseline implementation (`models/baseline.py`)

- **Model choice — XGBoost.** Fast, low-tuning, handles the moderate tabular
  matrix and missing values natively, and exposes gain importances that we can
  sanity-check against physics — ideal for an interpretable Day-2 baseline on
  physics-informed features.
- **Targets.** Regression: `wear_level` (primary), `cycle_time_factor`,
  `quality_score` (MAE/RMSE/R²). Classification: `health_state` (4 classes;
  accuracy, macro-F1, per-class report).
- **Leakage-aware feature policy.** Features = DSP `vib_td_*`/`vib_fd_*`,
  observable thermal stats (`mean_temp_c`, `max_temp_c`, `temp_rise_c`,
  `therm_std_c`, `therm_slope_c_per_s`) and wear-independent machine setpoints /
  kinematics (blade speed, teeth, feed, depth, kerf, TPF, RPM, cutting velocity,
  MRR). **Excluded** (derived from wear → leakage): specific energy, cutting
  force/power, force multiplier, steady-state temp, heat input, and all `label_*`.
- **Split / ablation.** One shared 80/20 split stratified on `wear_bin`
  (`random_state=42`); a further 80/20 carve of train forms the early-stopping
  validation set. Ablation retrains the wear regressor on **time-only**,
  **freq-only**, and **all** feature groups.
- **Params.** `n_estimators=500, max_depth=5, lr=0.05, subsample=0.9,
  colsample_bytree=0.9, reg_lambda=1.0, early_stopping_rounds=40`.
- **Artifacts.** `joblib` models → `experiments/models/`, importance bar plots →
  `experiments/plots/baseline_xgboost_feature_importance_<target>.png`, metrics →
  `experiments/baseline_metrics.json`, summary → `experiments/baseline_results.md`.
- Updated `pyproject.toml` `[project.optional-dependencies] ml = [scikit-learn>=1.5,
  xgboost>=2.1, joblib>=1.4, # shap>=0.46]` and the deferred note in `requirements.txt`.

---

## 4. Experimental results (actual numbers)

### Commands
```
python -m pip install "xgboost>=2.1"                     # -> xgboost 3.3.0
python scripts/generate_dataset.py --num-samples 300 \
    --output-dir data/test_dsp_v1 --seed 42 --extract-features
python models/baseline.py --data-dir data/test_dsp_v1 --seed 42
python -m pytest -q --tb=short                           # 52 passed
```

### Dataset generation (`data/test_dsp_v1`, 300 samples)
- 168.1 ms/sample; 30 DSP features/sample; 181 MB across 20 record files +
  manifest; alloys {6061: 161, 7075: 139}; anomalies 56 (18.7%).
- Top feature correlations (from the generator summary):
  - `corr(vib_td_rms, wear_level) = +0.420`, `vib_td_envelope_mean = +0.420`,
    `vib_td_envelope_std = +0.416`, `vib_td_peak_to_peak = +0.411`,
    `vib_td_peak = +0.410`, `vib_fd_total_power = +0.404`.
  - Signs flip for `quality_score` (all ≈ −0.42) and match for `cycle_time_factor`
    (all ≈ +0.42), exactly as physics dictates. (Correlations are moderate, not
    ~1, because operating-point confounders — feed/depth/speed/teeth — vary widely.)

### Regression (test n=60)
| Target | MAE | RMSE | R² | best_iter |
| --- | --- | --- | --- | --- |
| wear_level | **0.1503** | 0.1862 | 0.5580 | 363 |
| cycle_time_factor | 0.0699 | 0.0869 | 0.5547 | 366 |
| quality_score | 0.0747 | 0.0932 | 0.5570 | 301 |

### Top-8 feature importances — wear_level model
1. `vib_fd_dominant_amplitude` 0.331
2. `therm_slope_c_per_s` 0.058
3. `max_temp_c` 0.049
4. `vib_td_peak_to_peak` 0.049
5. `material_removal_rate_mm3_s` 0.043
6. `vib_td_margin_factor` 0.034
7. `vib_td_envelope_mean` 0.033
8. `vib_fd_tpf_harmonic2_energy` 0.029
(then `vib_fd_broadband_energy` 0.028, `vib_fd_tpf_band_energy` 0.026) — a healthy
mix of spectral impact energy, thermal heating rate, and impulsiveness stats.

### Ablation — wear_level (MAE, test n=60)
| Feature set | n_features | MAE | RMSE | R² | ΔMAE vs all |
| --- | --- | --- | --- | --- | --- |
| time_only | 13 | 0.2439 | 0.2793 | 0.0059 | +0.0936 |
| freq_only | 15 | 0.2334 | 0.2719 | 0.0582 | +0.0831 |
| **all** | 42 | **0.1503** | 0.1862 | 0.5580 | +0.0000 |
Frequency-domain features slightly edge time-domain alone, but the **combined**
set (DSP + thermal + kinematics context) cuts MAE by ~38% and lifts R² from ~0.01
to 0.56 — sensor/feature fusion is decisive.

### Classification — health_state (test n=60)
- Accuracy **0.5167**, macro-F1 **0.4449** over 4 classes
  (`critical, healthy, monitor, warning`; chance ≈ 0.25).
- Top features: `temp_rise_c` 0.093, `vib_fd_total_power` 0.048, `max_temp_c` 0.047,
  `vib_fd_tpf_band_energy` 0.044, `vib_td_envelope_std` 0.038.

### Plots generated (`experiments/plots/`)
`baseline_xgboost_feature_importance_{wear_level,cycle_time_factor,quality_score,
health_state}.png`. Notebook `experiments/notebooks/baseline_training.ipynb`
(feature-vs-label correlation heatmap, TPF-tracking sweep, importance + ablation
tables) executes end-to-end (`jupyter nbconvert --execute`, exit 0).

---

## 5. Testing & verification

- **`tests/test_signal_processor.py` (new, 20 tests) — all pass.** Covers:
  preprocess length/energy preservation (RMS ratio 1.0004) & DC removal; zscore
  normalization; time/freq feature finiteness & ranges; dominant-freq in-band +
  TPF band energy present; **TPF-band energy monotonic in wear** (0.775 → 3.917 →
  12.345 g²/Hz at wear 0/0.5/1.0); **broadband monotonic in wear**; graceful NaN
  without TPF; STFT shapes `(nperseg//2+1, n_time)`; `process` structure &
  fs-from-metadata & optional spectrogram; **reproducibility** (identical features
  for a fixed seed); `process_batch` shape/columns/monotonicity + metadata join;
  dict/override construction; invalid-fs raises; 200 Hz thermal path.
- **`tests/conftest.py`:** added a `proc` fixture (`SignalProcessor()`).
- **Full suite:** `52 passed` (32 pre-existing simulator tests + 20 new).

### On-the-fly fixes during verification
1. `test_dominant_freq_near_tpf` initially asserted the dominant PSD line sat at a
   low TPF harmonic; the diagnostic showed **dominant_freq = 2070 Hz** — the
   tooth-impact pulse *ring frequency* (`ring_freq_hz=2000`) plus structural modes
   dominate, which is physically correct. Rewrote the test to assert the dominant
   line is **in-band** and that the TPF fundamental band still carries positive
   energy (`fd_tpf_band_ratio ∈ (0,1]`).
2. `test_thermal_path_low_fs` filtered non-finite checks with
   `not k.startswith("fd_tpf")`, which missed `fd_harmonic_to_fundamental_ratio`
   (also TPF-relative). Introduced an explicit `_TPF_RELATIVE` set and asserted
   those are `NaN` while every other feature is finite.

---

## 6. Documentation & repo updates

- `dsp/README.md`: status → “✅ v1 implemented (Day 2)”; pipeline write-up, usage
  snippet with real numbers, integration points.
- `models/README.md`: Day-2 XGBoost baseline + ablation + leakage-policy write-up.
- `models/__init__.py`: version 0.1.0; documents that `models.baseline` is imported
  lazily so `import models` never requires the `ml` extra.
- Root `README.md`: status table (`dsp/` ✅ v1, `models/` 🚧 baseline+ablations),
  quickstart step 5 (features + baseline), repo layout, roadmap (Day 2 ✅).
- `experiments/notebooks/baseline_training.ipynb` (new).

---

## 7. Bugs / tuning encountered and resolved

1. **Variable-TPF band-pass.** TPF spans ~100–2000 Hz across operating points and
   fs differs by modality (40.96 kHz vs 200 Hz). Fixed with a single config whose
   corners are clamped `< 0.999·Nyquist` at runtime and a low-pass fallback when
   the low corner ≤ 0 — the 200 Hz thermal path runs without error.
2. **Normalization vs wear signal.** Defaulting `normalize="none"` (documented)
   was essential: z-scoring per window destroyed the absolute band-energy growth
   that carries wear information (would have inverted the monotonicity tests).
3. **Numerical stability in Welch/flatness.** Added `_EPS` floors to
   `total_power`, the geometric-mean flatness, and all ratio denominators; excluded
   the DC bin from the dominant-peak search.
4. **Dtype / list-column handling.** DSP features cast to float32-representable
   scalars (`float(np.float32(v))`) for compact Parquet; the raw waveform
   `float32` list columns in `records/` are left untouched by the feature path.
5. **XGBoost 3.x early stopping.** `early_stopping_rounds` + `eval_metric` set on
   the constructor with a carved validation `eval_set` in `.fit(...)` (best_iter
   301–366 of 500), with a non-stratified split fallback for sparse classes.

---

## 8. Verification snapshot (commands + key stdout)

```
$ python -m pip install "xgboost>=2.1"
Successfully installed xgboost-3.3.0

$ python -m pytest -q --tb=short
....................................................                     [100%]
52 passed

$ python scripts/generate_dataset.py --num-samples 300 --output-dir data/test_dsp_v1 \
      --seed 42 --extract-features
DSP feature extraction ENABLED (processor v0.1.0)
Done in 50.4s (168.1 ms/sample)
DSP FEATURES: 30 extracted per sample
corr(vib_td_rms, wear_level) = +0.420 ; corr(vib_td_rms, quality_score) = -0.420
manifest: 300 rows x 75 cols ; 0 NaN in feature columns

$ python models/baseline.py --data-dir data/test_dsp_v1 --seed 42
REGRESSION wear_level : MAE=0.1503 RMSE=0.1862 R2=0.5580 (best_iter=363)
ABLATION  time_only : MAE=0.2439 ; freq_only : MAE=0.2334 ; all : MAE=0.1503
CLASSIFICATION health_state : Accuracy=0.5167 macro-F1=0.4449
Baseline complete in 4.0s

# backward-compat (no flag): manifest 45 cols, has DSP feats: False
```

---

## 9. Immediate next steps / TODOs for Day 3

1. **1D-CNN on raw vibration** using the amplitude-normalized `preprocess` output
   (add `normalize="zscore"` config for the DL path) — PyTorch.
2. **Spectrogram CNN** consuming `SignalProcessor.compute_stft` (log-power); wire a
   `--compute-spectrogram` option into the dataset generator or a lazy loader over
   `records/`.
3. **Sensor fusion** (vibration + thermal) heads; compare against the XGBoost
   baseline on the same split/metrics.
4. **ONNX export** + CPU/edge latency benchmarks (ONNX Runtime), with a note for
   Jetson/OpenVINO.
5. **Scale up** to a few-thousand-sample dataset to tighten R²/F1 and add noise-
   robustness ablations; optionally add SHAP (already flagged in the `ml` extra).

Session complete — DSP + baseline ready for Day 3 expansion.
