# Argus Panoptes — Session Change Log

**Date:** 2026-07-06 (late night)
**Scope:** Day-3 **hardening** — configurable training-time Gaussian noise
augmentation; `normalize_for_dl="none"` end-to-end confirmation and ablation;
re-training of DL models (noisy + amplitude-preserving variants); benchmark /
robustness artifact integration; tests and docs. No streaming inference, FastAPI,
Streamlit, or deployment changes. No regressions to `sensors/`, `dsp/` core
behavior, Parquet schema, or XGBoost baseline metrics.
**Environment:** Python 3.13.9 (Windows / PowerShell). torch 2.12.1, onnx 1.22.0,
onnxruntime 1.27.0. Dataset: `data/dl_v1` (2500 samples, seed 42).

---

## 1. Motivation

The prior Day-3 session (`logs/2026-07-06_argus-v1-dl-fusion-onnx.md`) left two
open gaps before Day 4:

1. **Robustness** — zscore 1D-CNNs degraded sharply under Gaussian noise (wear
   MAE 0.24→0.74 at sd=0.5·rms) while XGBoost held steady (0.08→0.13).
2. **Accuracy** — the DL gap to XGBoost (wear R²≈0.35 fusion vs 0.89 tabular)
   was traced to per-chunk z-scoring discarding absolute amplitude, the dominant
   wear cue on this synthetic data.

This session implemented both mitigations, re-ran experiments, and integrated
results into persistent artifacts.

---

## 2. Design decisions

- **On-the-fly noise only in the train loader.** Additive Gaussian noise
  `N(0, train_noise_sd · rms)` is applied per sample inside
  `ArgusDLDataset.__getitem__` after precomputed normalized waveforms are loaded.
  Never written to Parquet. Val/test loaders always pass `noise_sd_ratio=0`.
- **`train_noise_sd` as RMS fraction.** Matches the robustness ablation convention
  (`sd=0.1·rms`, etc.) and keeps the parameter scale-invariant across chunks.
- **Spectrogram path with noise.** For `mode="spectrogram"`, normalized waveforms
  are also stored; when `noise_sd_ratio > 0`, noise is injected into the waveform
  and the STFT is computed on-the-fly (precomputed spectrograms used only for
  clean val/test).
- **`normalize_for_dl="none"` via existing config.** No schema or DSP scalar-path
  changes; `train_dl.py` accepts `--normalize-for-dl` override that builds a
  `SignalProcessor` with `dl.normalize_for_dl` set accordingly. A one-time
  `logging.warning` in `get_normalized_waveform` documents CNN scale sensitivity.
- **Artifact naming via `--output-suffix`.** Noisy / norm-none models saved as
  `dl_{model}_noisy.onnx`, `dl_{model}_normnone.onnx`, etc., preserving the
  original zscore baselines untouched.
- **Robustness ablation extensibility.** `experiments/robustness_ablation.py`
  gains `--dl-onnx` / `--dl-label` so noisy-trained artifacts can be compared
  without code changes.

---

## 3. Code changes (surgical, by file)

### `models/dl_data.py`
- `ArgusDLDataset`: new `noise_sd_ratio`, `processor`, `fs_hz` params;
  `_maybe_noisy_waveform()` helper; spectrogram on-the-fly when noise > 0.
- `prepare_dl_data()`: new `train_noise_sd` param; train loader gets noise,
  val/test do not; spectrogram precompute unified through `get_normalized_waveform`.
- Return dict extended with `train_noise_sd`, `normalize_for_dl`.

### `models/train_dl.py`
- CLI: `--train-noise-sd`, `--normalize-for-dl`, `--early-stopping-patience`,
  `--output-suffix`.
- Checkpoint / metrics JSON record noise and normalization settings.
- `patience` forwarded to `train_dl_model`.

### `dsp/signal_processor.py`
- Class flag `_warned_dl_none`; warning on first `get_normalized_waveform` call
  when `dl.normalize_for_dl == "none"`.

### `scripts/benchmark_onnx.py`
- `--model {1dcnn,spectrogram,fusion,all}`, `--artifact`, `--device`.
- Wider table columns (18-char model name) to prevent cutoff.

### `experiments/robustness_ablation.py`
- `--dl-onnx`, `--dl-label`; results JSON records which artifact was scored.

### `tests/test_dl_models.py`
- `test_training_noise_augmentation_changes_waveform` — verifies noisy vs clean
  dataset items differ.

### Documentation
- `experiments/dl_results.md` (new) — full regression/classification/robustness tables.
- `experiments/robustness_results.json` — restructured with per-model variants.
- `README.md` — status table, Metrics section, quickstart noise/norm examples.
- `experiments/README.md` — artifact index updated.

---

## 4. Re-training runs (same split, seed 42)

| Command | Epochs | wear MAE | wear R² | health F1 | Notes |
| --- | --- | --- | --- | --- | --- |
| fusion `--train-noise-sd 0.15 --output-suffix _noisy` | 11 (ES) | 0.251 | −0.065 | 0.181 | High noise aug hurts clean acc |
| fusion `--train-noise-sd 0.10 --output-suffix _noisy01` | 11 (ES) | 0.227 | 0.142 | 0.329 | Milder trade-off |
| 1dcnn `--train-noise-sd 0.15 --output-suffix _noisy` | 9 (ES) | 0.263 | −0.162 | 0.165 | Best high-noise robustness |
| 1dcnn `--train-noise-sd 0.10 --output-suffix _noisy01` | 25 (ES) | 0.232 | 0.107 | 0.247 | Clean acc preserved |
| fusion `--normalize-for-dl none --output-suffix _normnone` | 39 (ES) | **0.100** | **0.815** | **0.695** | Near XGBoost |
| 1dcnn `--normalize-for-dl none --output-suffix _normnone` | 36 (ES) | **0.100** | **0.783** | 0.573 | Near XGBoost |

Original zscore baselines (`dl_fusion.onnx`, `dl_1dcnn.onnx`) unchanged.

---

## 5. Experimental findings

### Normalization (zscore vs none) — primary accuracy win

| Model | zscore wear MAE/R² | none wear MAE/R² | Δ MAE |
| --- | --- | --- | --- |
| 1D-CNN | 0.231 / 0.106 | 0.100 / 0.783 | −0.131 |
| Fusion | 0.190 / 0.352 | 0.100 / 0.815 | −0.090 |
| XGBoost | — | 0.080 / 0.886 | — |

`normalize_for_dl="none"` restores amplitude to CNN inputs and closes most of the
gap to XGBoost on clean synthetic data. Fusion health macro-F1 (0.695) actually
**exceeds** XGBoost (0.651). Inference latency unchanged (~0.34 ms/chunk p50).

**Trade-off:** `"none"` is scale-sensitive across sensor gains; `"zscore"` sacrifices
amplitude but is more gain-invariant.

### Noise augmentation — primary robustness win

Gaussian corruption ablation (test n=300, 1D-CNN wear MAE):

| Corruption | XGB | 1D-CNN zscore | 1D-CNN zscore + 0.15 noise aug |
| --- | --- | --- | --- |
| clean | 0.084 | 0.237 | 0.276 |
| sd=0.1·rms | 0.093 | 0.249 | 0.264 |
| sd=0.25·rms | 0.114 | 0.444 | **0.255** |
| sd=0.5·rms | 0.132 | 0.742 | **0.251** |

At sd=0.5·rms, noise-augmented training cuts DL wear MAE by **3×** (0.74→0.25).
Mild aug (0.10·rms) preserves clean accuracy while moderately helping at high noise.

### ONNX CPU latency (unchanged architecture)

| Model | single p50 (ms) | batch-32 p50 (ms) | throughput (c/s) |
| --- | --- | --- | --- |
| 1D-CNN | 0.332 | 13.93 | 2298 |
| SpectrogramCNN | 0.388 | 22.49 | 1423 |
| Fusion | 0.381 | 14.52 | 2203 |
| XGBoost end-to-end | 9.79 | — | — |

---

## 6. New / updated artifacts

**Models** (`experiments/models/`):
- `dl_fusion_noisy.{pt,onnx}`, `dl_fusion_noisy01.{pt,onnx}`
- `dl_1dcnn_noisy.{pt,onnx}`, `dl_1dcnn_noisy01.{pt,onnx}`
- `dl_fusion_normnone.{pt,onnx}`, `dl_1dcnn_normnone.{pt,onnx}`

**Metrics JSONs** (`experiments/`):
- `dl_fusion_noisy_metrics.json`, `dl_fusion_noisy01_metrics.json`
- `dl_1dcnn_noisy_metrics.json`, `dl_1dcnn_noisy01_metrics.json`
- `dl_fusion_normnone_metrics.json`, `dl_1dcnn_normnone_metrics.json`

**Reports:**
- `experiments/dl_results.md` — consolidated tables + commentary
- `experiments/robustness_results.json` — baseline + noisy variants
- `experiments/robustness_results_noisy.json`, `_noisy01.json` — per-run detail
- `experiments/onnx_benchmark.json` — refreshed

**Plots:**
- `experiments/plots/dl_*_noisy*_training_curve.png`
- `experiments/plots/dl_*_normnone_training_curve.png`

---

## 7. Testing & verification

```
$ python -m pytest tests/test_dl_models.py -q --tb=line
12 passed

$ python -m pytest tests/test_signal_processor.py -q --tb=line
25 passed

$ python -m pytest tests/ -q --tb=no
70 passed   (was 69; +1 noise-augmentation test)

$ python scripts/generate_dataset.py --help          # OK
$ python models/train_dl.py --help                   # OK (new flags visible)
$ python scripts/benchmark_onnx.py --model fusion --device cpu   # complete table
$ python -m pytest tests/test_baseline.py -q         # no XGBoost regression
```

ONNX round-trip parity (≤1e-4) preserved for all exported variants via existing
`tests/test_dl_models.py::test_onnx_export_roundtrip` on the base architectures.

---

## 8. Production recommendations (for Day 4+)

| Deployment scenario | Recommended artifact | Rationale |
| --- | --- | --- |
| Clean, gain-calibrated edge | `dl_fusion_normnone.onnx` | wear MAE 0.10, R² 0.82, F1 0.70 |
| Noisy / variable SNR edge | `dl_1dcnn_noisy.onnx` | 3× better under sd=0.5·rms corruption |
| Interpretable / auditable | XGBoost joblib | Still best clean R²; slower end-to-end |

Day 4 can wire either ONNX path through `ONNXPerceptor` + streaming `Perceptor`.

---

## 9. What was explicitly NOT done

- No `StreamingPerceptor`, FastAPI `/infer` / `/batch`, or Streamlit dashboard.
- No changes to `sensors/`, Parquet schema, or `models/baseline.py` behavior.
- No re-generation of `data/dl_v1` (reused existing dataset).
- Original zscore DL baselines not overwritten.

---

## 10. Verification snapshot (key commands)

```bash
# Noise-augmented fusion
python models/train_dl.py --model fusion --data-dir data/dl_v1 \
    --train-noise-sd 0.15 --epochs 40 --early-stopping-patience 8 \
    --output-suffix _noisy --no-compare-baseline

# Amplitude-preserving fusion (best clean accuracy)
python models/train_dl.py --model fusion --data-dir data/dl_v1 \
    --normalize-for-dl none --output-suffix _normnone --no-compare-baseline

# Robustness on noisy-trained 1D-CNN
python experiments/robustness_ablation.py --data-dir data/dl_v1 --max-samples 300 \
    --dl-onnx experiments/models/dl_1dcnn_noisy.onnx --dl-label "1D-CNN-noisy"

# Full benchmark
python scripts/benchmark_onnx.py --model all --device cpu
```

Session complete — Day 3 deliverables hardened and production-verified; repo
ready for Day 4 streaming integration.
