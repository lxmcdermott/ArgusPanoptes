# `models/` — ML Pipeline & Experiments

> **Status:** scaffold (implemented in Day 2–3 of the execution plan).

Trains and evaluates models on the synthetic Parquet dataset produced by
`scripts/generate_dataset.py`.

## Planned scope (per technical plan §3)

- **Baselines:** XGBoost / LightGBM on engineered features (interpretable
  importances).
- **Deep learning:** 1D-CNN (raw vibration), spectrogram CNN, sensor-fusion
  (vibration + thermal).
- **Tasks:** regression (wear / RUL), classification (health states), anomaly
  detection.
- **Ablations:** feature groups, single vs. fusion sensors, noise robustness,
  model types.
- **Export:** PyTorch → ONNX with CPU/edge latency benchmarks.
