# Argus Panoptes — Deep Learning Results (Day 3 hardened)

Dataset: `data/dl_v1` · 2500 samples · seed 42 · same stratified split as XGBoost
(train 1600 / val 400 / test 500).

## Regression — wear_level (primary)

| Model | norm | noise aug | MAE | R² |
| --- | --- | --- | --- | --- |
| Vibration1DCNN | zscore | — | 0.231 | 0.106 |
| Vibration1DCNN | zscore | 0.10·rms | 0.232 | 0.107 |
| Vibration1DCNN | zscore | 0.15·rms | 0.263 | −0.162 |
| **Vibration1DCNN** | **none** | — | **0.100** | **0.783** |
| SpectrogramCNN | zscore | — | 0.240 | 0.077 |
| FusionModel | zscore | — | 0.190 | 0.352 |
| FusionModel | zscore | 0.10·rms | 0.227 | 0.142 |
| FusionModel | zscore | 0.15·rms | 0.251 | −0.065 |
| **FusionModel** | **none** | — | **0.100** | **0.815** |
| **XGBoost (baseline)** | tabular | — | **0.080** | **0.886** |

## Classification — health_state

| Model | norm | noise aug | Accuracy | macro-F1 |
| --- | --- | --- | --- | --- |
| Vibration1DCNN | zscore | — | 0.314 | 0.294 |
| Vibration1DCNN | none | — | 0.666 | 0.573 |
| FusionModel | zscore | — | 0.402 | 0.373 |
| FusionModel | none | — | **0.726** | **0.695** |
| XGBoost (baseline) | tabular | — | 0.658 | 0.651 |

## Normalization experiment (zscore vs none)

Per-chunk z-scoring removes the dominant wear cue (RMS growth 2.3→9.2 g) that the
tabular path deliberately preserves (`preprocess.normalize="none"`). With
`normalize_for_dl="none"`, both waveform CNNs recover amplitude information and
close most of the gap to XGBoost on clean synthetic data:

| Model | zscore wear MAE/R² | none wear MAE/R² | Δ MAE |
| --- | --- | --- | --- |
| 1D-CNN | 0.231 / 0.106 | 0.100 / 0.783 | −0.131 |
| Fusion | 0.190 / 0.352 | 0.100 / 0.815 | −0.090 |

Inference latency is unchanged (same ONNX graph; only input scaling differs):
~0.34 ms/chunk p50 on CPU for 1D-CNN and fusion.

**Trade-off:** `"none"` is scale-sensitive across sensor gains and operating
points; `"zscore"` generalizes better across absolute levels but sacrifices the
primary amplitude feature on this dataset. Production should pick based on whether
gain calibration is stable.

## Noise augmentation (training-time Gaussian, zscore models)

Additive noise `N(0, train_noise_sd · rms)` applied on-the-fly in the train
loader only (`--train-noise-sd`). Improves robustness on corrupted test waveforms
at the cost of some clean accuracy when `train_noise_sd` is high.

### Robustness ablation — wear MAE under Gaussian corruption (test n=300)

| Corruption | XGB | 1D-CNN (zscore) | 1D-CNN (zscore + 0.15 noise aug) |
| --- | --- | --- | --- |
| clean | 0.084 | 0.237 | 0.276 |
| sd=0.1·rms | 0.093 | 0.249 | 0.264 |
| sd=0.25·rms | 0.114 | 0.444 | **0.255** |
| sd=0.5·rms | 0.132 | 0.742 | **0.251** |

At `sd=0.5·rms`, noise-augmented training cuts DL wear MAE by **3×** (0.74→0.25)
vs the zscore baseline. Mild noise (`0.10·rms` aug) preserves clean accuracy while
moderately improving high-noise performance.

## ONNX CPU latency (p50)

| Model | single (ms) | batch-32 (ms) | throughput (c/s) |
| --- | --- | --- | --- |
| 1D-CNN | 0.336 | 12.96 | 2469 |
| SpectrogramCNN | 0.366 | 21.56 | 1484 |
| Fusion | 0.336 | 13.55 | 2362 |
| XGBoost end-to-end | 9.79 | — | — |

## Why the original zscore DL gap to XGBoost existed — and how mitigations help

1. **Root cause:** z-score per chunk discards absolute amplitude, the strongest
   single wear signal in the synthetic physics. XGBoost uses integrated band
   energies and thermal scalars that retain amplitude.

2. **`normalize_for_dl="none"`** restores amplitude to the CNN input and brings
   fusion wear R² from 0.35 → **0.81** (near XGBoost 0.89).

3. **Noise augmentation** does not fix the amplitude loss but hardens zscore models
   against sensor noise — critical for edge deployment where SNR varies.

Artifacts: `experiments/models/dl_*.{pt,onnx}`, metrics JSONs alongside this file,
`experiments/robustness_results.json`, `experiments/onnx_benchmark.json`.

_Generated during Day 3 hardening._
