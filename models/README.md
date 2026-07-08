# `models/` — ML Pipeline & Experiments

> **Status:** XGBoost baseline + ablations (Step 2) · DL (1D-CNN / spectrogram
> / fusion) + ONNX export & benchmarks (Step 3).

Trains and evaluates models on the synthetic Parquet dataset produced by
`scripts/generate_dataset.py --extract-features`.

## Implemented (Step 2): `models/baseline.py`

Interpretable **XGBoost** baselines on the DSP feature matrix in
`manifest.parquet`:

- **Regression** — `wear_level` (primary), `cycle_time_factor`, `quality_score`
  (MAE / RMSE / R²).
- **Classification** — `health_state` (healthy / monitor / warning / critical):
  accuracy, macro-F1, per-class report.
- **Ablation** — time-domain-only vs frequency-domain-only vs all feature groups,
  reporting the MAE deltas that quantify each group's contribution.
- **Interpretability** — sorted feature-importance bar plots saved to
  `experiments/plots/baseline_xgboost_feature_importance_<target>.png`.
- **Artifacts** — trained models (`joblib`) → `experiments/models/`, metrics →
  `experiments/baseline_metrics.json`, summary → `experiments/baseline_results.md`.

Why XGBoost first: fast, low-tuning, handles the moderate tabular matrix and
missing values natively, and exposes importances we can sanity-check against the
physics (tooth-pass band energy, RMS, envelope stats should drive a wear model).

**Leakage-aware feature policy:** only *observable* quantities are used — DSP
`vib_td_*` / `vib_fd_*` features, observable thermal statistics, and machine
setpoints/kinematics known independently of wear. Quantities the force model
derives *from* wear (specific energy, cutting force/power, force multiplier,
steady-state temp, heat input) and all `label_*` columns are excluded.

```bash
pip install -e ".[ml]"          # scikit-learn, xgboost, joblib
python models/baseline.py --data-dir data/test_dsp_v1 --seed 42
```

## Implemented (Step 3): deep learning + ONNX

`models/dl_models.py` — three **multi-task** PyTorch models (shared trunk → a
regression head for `wear_level` / `cycle_time_factor` / `quality_score` and a
classification head for `health_state`):

- **`Vibration1DCNN`** — 1-D conv on the z-score-normalized waveform
  (`SignalProcessor.get_normalized_waveform`); global-avg-pooled so it accepts
  variable-length chunks.
- **`SpectrogramCNN`** — 2-D conv on the log-power STFT spectrogram
  (`SignalProcessor.compute_spectrogram`, shape `(n_freq, n_time)`).
- **`FusionModel`** — late fusion of a 1-D vibration branch + a thermal
  scalar-feature MLP (observable thermal features only, leakage-aware).

Plus `set_seed`, `train_dl_model` / `evaluate_dl_model` (baseline-parity metrics:
MAE/RMSE/R² and Accuracy/macro-F1, Adam + early stopping), a `get_model(name,
config)` factory, and `export_to_onnx` (dynamic batch/length axes, opset 17).

Supporting modules:

- `models/dl_data.py` — Parquet → PyTorch `DataLoader`s. Reads the raw `float32`
  waveforms from `records/`, computes normalized waveforms / spectrograms
  **on-the-fly** via `SignalProcessor`, and reproduces the **exact** baseline
  train/val/test split so DL and XGBoost are compared on the same test rows.
- `models/train_dl.py` — training CLI; saves a checkpoint (`.pt`), an ONNX
  artifact, a metrics JSON, and a training-curve plot, and prints the XGBoost
  comparison on the same split.
- `models/onnx_inference.py` — `ONNXPerceptor` / `infer_onnx` for **torch-free**
  edge inference (only `onnxruntime` needed).

```bash
pip install -e ".[ml,dl]"          # torch, onnx, onnxruntime (+ ml for comparison)
python models/train_dl.py --model 1dcnn       --data-dir data/dl_v1 --epochs 40
python models/train_dl.py --model spectrogram --data-dir data/dl_v1 --epochs 40
python models/train_dl.py --model fusion      --data-dir data/dl_v1 --epochs 40
python scripts/benchmark_onnx.py               # p50/p95 latency, throughput
```

**Step 3 finding (2500-sample dataset, same test split).** On this clean,
physics-informed synthetic data the interpretable XGBoost baseline (wear R²≈0.88)
still beats the raw-signal DL models; among the DL models the **fusion** model is
strongest (wear R²≈0.35), confirming the tabular ablation's "sensor fusion wins".
The z-score normalization that helps CNN generalization also discards the
absolute-amplitude wear cue, which is the main reason the pure vibration CNNs
trail the amplitude-aware tabular model — see `logs/2026-07-06_argus-v1-dl-fusion-onnx.md`.
All ONNX models run in **< 0.5 ms/chunk on CPU** (well under the 50 ms edge target).
