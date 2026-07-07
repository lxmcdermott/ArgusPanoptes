# `models/` — ML Pipeline & Experiments

> **Status:** 🚧 XGBoost baseline + ablations (Day 2). DL models (Day 3+) pending.

Trains and evaluates models on the synthetic Parquet dataset produced by
`scripts/generate_dataset.py --extract-features`.

## Implemented (Day 2): `models/baseline.py`

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

## Planned (Day 3+)

- **Deep learning:** 1D-CNN (raw vibration), spectrogram CNN (`SignalProcessor.compute_stft`),
  sensor-fusion (vibration + thermal).
- **Export:** PyTorch → ONNX with CPU/edge latency benchmarks.
