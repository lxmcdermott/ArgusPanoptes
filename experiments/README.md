# `experiments/` — Notebooks, Ablations & Reports

> **Status:** scaffold. Validation plots land in `experiments/plots/`.

This directory collects exploratory notebooks, ablation studies, and generated
figures. `scripts/validate_simulators.py` writes its diagnostic PNGs to
`experiments/plots/` by default.

- `notebooks/` — Jupyter notebooks:
  - `baseline_training.ipynb` — Day-2 XGBoost baseline (correlations, ablation).
  - `dl_training.ipynb` — Day-3 DL: model-vs-baseline comparison, ONNX latency,
    and robustness tables, reproduced from the saved metrics JSONs (runs in
    seconds without retraining).
- `robustness_ablation.py` — Day-3 noise-robustness sweep (Gaussian / drift /
  quantization) comparing the XGBoost DSP path against the 1D-CNN; writes
  `robustness_results.json` + `plots/robustness_ablation.png`.
- `models/` — trained artifacts (`xgb_*.joblib`, `dl_*.pt`, `dl_*.onnx`).
- `plots/` — generated figures (git-ignored except this note).
- Metrics: `baseline_metrics.json`, `dl_*_metrics.json`, `onnx_benchmark.json`.
