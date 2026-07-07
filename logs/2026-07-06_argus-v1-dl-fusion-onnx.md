# Argus Panoptes — Session Change Log

**Date:** 2026-07-06 (late evening)
**Scope:** Day-3 deep learning — 1D-CNN, spectrogram CNN, and vibration+thermal
**fusion** models; DSP DL-input methods; on-the-fly DL data pipeline reusing the
Parquet waveforms; training orchestration with a same-split XGBoost comparison;
**ONNX export + CPU edge benchmarks**; dataset scaling to 2500 samples; a
**noise-robustness ablation**; new tests; docs. Includes explicit **verification
of the recent harmonic-energy bug fix** (commit `5dfd041`). No regressions to the
tabular XGBoost baseline or the Parquet schema.
**Environment:** Python 3.13.9 (Windows / PowerShell). Core deps unchanged
(NumPy 2.3.5, SciPy 1.16.3, Pandas, PyArrow, Pydantic, PyYAML, Matplotlib, tqdm).
`ml` extra: scikit-learn 1.7.2, xgboost 3.3.0, joblib. New **`dl` extra** installed
via `python -m pip install "torch>=2.2" "onnx>=1.16" "onnxruntime>=1.18"`:
**torch 2.12.1**, **onnx 1.22.0**, **onnxruntime 1.27.0**.

---

## 0. Bug-fix verification (harmonic energy, commit `5dfd041`)

Confirmed the fix is present in `dsp/signal_processor.py`
(`extract_frequency_domain_features`): `h2` / `h3` are computed only when
`n_harmonics >= 2` / `>= 3`, else `0.0`:

```python
n_harm = self.freq.n_harmonics
fund = self._band_energy(freqs, psd, tpf_hz * (1 - frac), tpf_hz * (1 + frac))
h2 = (self._band_energy(freqs, psd, 2*tpf*(1-frac), min(2*tpf*(1+frac), nyq)) if n_harm >= 2 else 0.0)
h3 = (self._band_energy(freqs, psd, 3*tpf*(1-frac), min(3*tpf*(1+frac), nyq)) if n_harm >= 3 else 0.0)
```

Runtime check with a forced low config (`n_harmonics: 1`) on a wear=0.5 signal:

```
h2 0.0   h3 0.0   fund 3.9150680157569813   total 3.9150680157569813   all finite: True
```

`h2`/`h3` collapse to `0.0`, the fundamental band still carries positive energy,
the harmonic total equals the fundamental, and **every** feature is finite — no
`NaN`, no error. Added a dedicated regression test
`tests/test_signal_processor.py::test_n_harmonics_one_graceful` asserting exactly
this (`h2==h3==0.0`, `fund>0`, `harmonic_total≈fund`,
`harmonic_to_fundamental_ratio==0.0`, all finite). Verifies robustness for
low-`n_harmonics` configs and small/edge-case datasets.

---

## 1. Design decisions

- **Multi-task heads, one network.** Each model has a shared trunk feeding a
  **regression** head (`wear_level`, `cycle_time_factor`, `quality_score`) and a
  **classification** head (`health_state`, 4 classes). This mirrors the baseline's
  target set and lets a single net be scored with identical metrics (MAE/RMSE/R²
  and Accuracy/macro-F1). `forward` returns a `(regression, health_logits)`
  **tuple** (not a dict) so `torch.onnx.export` keeps stable output names.
- **1D vs 2D.** The **1D-CNN** operates on the normalized waveform to learn the
  *shape* of the sharpening tooth-strike transients; the **spectrogram CNN**
  operates on the log-power STFT to learn time-frequency wear signatures
  (tooth-pass harmonics + broadband lift). Both end in **global average pooling**
  so they accept variable-length chunks — important for streaming and for ONNX
  dynamic axes.
- **Fusion strategy — late fusion.** A 1-D vibration branch and a small thermal
  scalar-feature **MLP** are concatenated into a shared FC head. Thermal uses the
  **leakage-aware** observable scalars (`mean_temp_c, max_temp_c, temp_rise_c,
  therm_std_c, therm_slope_c_per_s`, standardized) — identical to the baseline's
  `THERMAL_FEATURES` — so the comparison is fair and no wear-derived quantity
  leaks in.
- **`normalize_for_dl="zscore"` default.** The tabular path keeps
  `preprocess.normalize="none"` (absolute amplitude is a first-class wear
  feature), but CNNs generalize better on scale-invariant inputs. This is the
  central **trade-off** (see §5): z-scoring per chunk discards the dominant
  amplitude-growth cue, so the pure vibration CNNs give up the single most
  predictive signal. `"none"` remains available in config for amplitude-aware DL.
- **On-the-fly spectrograms, not stored.** A 2500×513×~65 spectrogram tensor is
  ~0.3 GB; storing it in Parquet is wasteful. Instead the DataLoader **precomputes
  in memory once per run** from the raw `float32` waveform columns already in
  `records/` (the 1D-CNN reuses them directly). `--compute-spectrogram` only
  writes a small `dl_config.json` recipe (STFT + normalize params) for
  reproducibility — nothing large is added to disk.
- **Same-split comparison.** `models/dl_data.py` reproduces the baseline split
  *exactly* (one stratified 80/20 test split on `wear_bin` with `seed`, then a
  further 80/20 carve of train for early-stopping validation), so DL and XGBoost
  are always evaluated on the **same held-out test rows**.
- **ONNX export choices.** Legacy TorchScript exporter (`dynamo=False`, opset 17):
  it consumes the `dynamic_axes` mapping directly and avoids the `onnxscript`
  dependency the new dynamo path pulls in (keeps the `dl` extra lean). Dynamic
  **batch** for all models and dynamic **sequence/time length** for the streaming
  models, so a single artifact serves any chunk size.
- **Torch-free edge inference.** `models/onnx_inference.py` (`ONNXPerceptor`)
  needs only `onnxruntime` + NumPy — no torch at inference time, which is the
  whole point of exporting for the edge.
- **Import hygiene.** Nothing DL is imported from `models/__init__.py`, so
  `import models` stays torch-free (verified: `torch in sys.modules == False`).

---

## 2. DSP / config extensions (Step 1, non-breaking)

- **`dsp/config.py`:** new `DlConfig` sub-model (`normalize_for_dl`, validated
  `{none,zscore,peak,rms}`) added to `ProcessorConfig.dl`; exported from
  `dsp/__init__.py`. `_normalize(x, method=None)` generalized to accept a method
  override (default still `preprocess.normalize`).
- **`dsp/processor_config.yaml`:** documented `dl:` section (`normalize_for_dl:
  "zscore"`) plus CNN-tuning notes on the existing `stft:` block (nperseg 1024 →
  513 freq bins @ 40.96 kHz, 75% overlap, dB log-scale).
- **`SignalProcessor.get_normalized_waveform(x, fs) -> float32`:** `preprocess`
  (detrend → band-pass → configured normalize) then the DL normalization on top;
  verified `mean≈0, std≈1.0` for the zscore default.
- **`SignalProcessor.compute_spectrogram(x, fs) -> float32`:** the same
  normalization then `compute_stft`, returning the 2-D **`(n_freq, n_time)`**
  power array (documented layout; `n_freq = nperseg//2 + 1 = 513`). Verified shape
  `(513, 161)` for a 1 s chunk, all finite.
- The scalar `process()` / `process_batch()` and the Parquet feature schema are
  **unchanged** (the 45-column no-flag manifest is byte-for-byte identical).

---

## 3. DL models module (`models/dl_models.py`)

Key excerpt — the shared head and the 1-D backbone:

```python
class _MultiTaskHead(nn.Module):
    def __init__(self, in_dim, cfg):
        self.reg = nn.Linear(in_dim, cfg.n_regression)   # wear / cycle / quality
        self.clf = nn.Linear(in_dim, cfg.n_classes)      # health_state
    def forward(self, z):
        return self.reg(z), self.clf(z)                  # tuple -> clean ONNX

class Vibration1DCNN(nn.Module):
    input_kind = "waveform"
    def forward(self, x):                 # (B, 1, L) or (B, L)
        if x.dim() == 2: x = x.unsqueeze(1)
        z = self.pool(self.features(x)).flatten(1)   # Conv/BN/ReLU/MaxPool + GAP
        return self.head(self.trunk(z))
```

Training loop highlight — multi-task loss + early stopping (best-val restore):

```python
loss = reg_weight * F.mse_loss(reg, y_reg) + clf_weight * F.cross_entropy(logits, y_clf)
# clf_weight defaults to 0.5 (regression MSE vs 4-class CE live on different scales)
...
if val_loss < best_val - 1e-6:
    best_val = val_loss; best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
```

Also: `get_model(name, config)` factory (aliases `1dcnn`/`spectrogram`/`fusion`),
`set_seed`, `train_dl_model`, `evaluate_dl_model` (numpy MAE/RMSE/R² + Accuracy /
macro-F1 — **no sklearn dependency**, keeps the `dl` extra lean),
`example_inputs_for`, `default_dynamic_axes`, and `export_to_onnx`. Model sizes
are deliberately small / edge-friendly: **1D-CNN 84,807**, **SpectrogramCNN
28,135**, **FusionModel 98,855** parameters.

---

## 4. Data pipeline & training orchestration (Steps 3)

- **`scripts/generate_dataset.py`** (additive, non-breaking): new
  `--compute-spectrogram`, `--normalize-for-dl {none,zscore,peak,rms}`,
  `--dl-output-dir` flags. `--normalize-for-dl` overrides `dl.normalize_for_dl`
  via config `overrides` (only affects the DL convenience methods); with
  `--compute-spectrogram` a `dl_config.json` recipe is written. Default behavior
  and schema are unchanged.
- **`models/dl_data.py`:** reads `records/` waveforms (sorted by `sample_id` to
  match manifest order), precomputes normalized waveforms / spectrograms via
  `SignalProcessor`, standardizes thermal scalars, and builds train/val/test
  `DataLoader`s using the exact baseline split. Center-crop/pad to a fixed
  `target_len=16384` (~0.4 s @ 40.96 kHz).
- **`models/train_dl.py`:** CLI (`--model/--data-dir/--epochs/--batch-size/--lr/
  --seed/--target-len/--device`); trains, evaluates on test, runs the **XGBoost
  baseline on the same non-test rows**, and saves a checkpoint (`.pt`), an ONNX
  artifact, a metrics JSON, and a training-curve plot.

**Scaled dataset (`data/dl_v1`):** 2500 samples, fixed 1.0 s duration,
`--extract-features --compute-spectrogram`, seed 42. Generated in **59.8 s**;
435.65 MB across 50 record files + manifest; wear bins balanced (488–513/bin);
`corr(wear, vib_rms_g) = +0.422`; `label_quality_score`/`cycle_time_factor`
correlate −1.00/+1.00 with wear by construction.

---

## 5. Experimental results (actual numbers, 2500 samples, same test split n=500)

### Regression — wear_level (primary)
| Model | params | MAE | R² |
| --- | --- | --- | --- |
| Vibration1DCNN   | 84,807 | 0.2313 | 0.1064 |
| SpectrogramCNN   | 28,135 | 0.2399 | 0.0765 |
| **FusionModel**  | 98,855 | **0.1904** | **0.3516** |
| **XGBoost (baseline)** | — | **0.0796** | **0.8863** |

### All targets (MAE / R²)
| Target | 1D-CNN | Spectrogram | Fusion | XGBoost |
| --- | --- | --- | --- | --- |
| wear_level        | 0.231 / 0.106 | 0.240 / 0.077 | 0.190 / 0.352 | **0.080 / 0.886** |
| cycle_time_factor | 0.116 / −0.056 | 0.117 / −0.054 | 0.110 / −0.042 | **0.037 / 0.887** |
| quality_score     | 0.123 / 0.015 | 0.121 / 0.040 | 0.101 / 0.295 | **0.040 / 0.885** |

### Classification — health_state (4 classes; chance ≈ 0.25)
| Model | Accuracy | macro-F1 |
| --- | --- | --- |
| Vibration1DCNN | 0.314 | 0.294 |
| SpectrogramCNN | 0.300 | 0.227 |
| **FusionModel** | **0.402** | **0.373** |
| **XGBoost (baseline)** | **0.658** | **0.651** |

### XGBoost ablation on the scaled set (wear_level, test n=500)
| Feature set | n_feat | MAE | R² |
| --- | --- | --- | --- |
| time_only | 13 | 0.2232 | 0.165 |
| freq_only | 15 | 0.1770 | 0.418 |
| **all** | 42 | **0.0810** | **0.882** |

**Honest interpretation & physics alignment.** On this clean, physics-informed
synthetic data the interpretable tabular baseline is decisively the strongest
model (wear R²≈0.89). Among the DL models the ordering is **fusion > 1D-CNN ≈
spectrogram**: adding the thermal branch lifts wear R² from ~0.08–0.11 to **0.35**
and health accuracy from ~0.30 to **0.40**, directly echoing the tabular
ablation's "sensor fusion wins" (all-features MAE ≈ ⅓ of any single group). The
gap to XGBoost is expected and **traceable to the `zscore` DL normalization**: it
removes the absolute-amplitude growth (RMS 2.3→9.2 g across wear) that is the
single most predictive wear cue and that the tabular path deliberately keeps
(`normalize="none"`). The DL nets are left with waveform *shape* / time-frequency
structure only, which the small models on 2500 samples learn to a limited degree.
Fusion partially recovers the lost signal through the (amplitude-bearing) thermal
scalars. Training was fast: 1D-CNN 14 epochs / 103 s, spectrogram 29 epochs /
379 s, fusion 30 epochs / 219 s (CPU, early-stopped). Curves saved to
`experiments/plots/dl_<model>_training_curve.png`.

---

## 6. ONNX export + edge benchmarks (Step 4)

All three models export to ONNX with **round-trip parity** vs PyTorch
(`max|Δ| ≈ 1e-8` on the regression output; ≤ 1e-4 asserted in tests). CPU latency
(`scripts/benchmark_onnx.py`, 200 timed iters after warmup, single chunk =
16384 samples):

| Model | single p50 (ms) | single p95 (ms) | batch-32 p50 (ms) | throughput (chunks/s) |
| --- | --- | --- | --- | --- |
| 1D-CNN | **0.371** | 0.481 | 14.60 | 2192 |
| SpectrogramCNN | **0.423** | 0.583 | 24.35 | 1314 |
| Fusion | **0.381** | 0.561 | 15.06 | 2124 |
| XGBoost (DSP extract + predict) | **9.571** end-to-end | — | — | — |

All DL models are **~0.4 ms/chunk on CPU** — >100× under the 50 ms edge target.
The tabular path is dominated by DSP feature extraction (Welch PSD etc.,
**9.15 ms**) vs a 0.42 ms tree predict; the 1D-CNN's end-to-end cost (z-score +
forward) is also sub-millisecond, so **the DL streaming path is actually the
faster end-to-end option** at the cost of accuracy on this dataset. (The
spectrogram model additionally needs an STFT front-end (~a few ms) in production,
not included in the pure-inference number.) Notes for further optimization
(documented, not implemented): NVIDIA Jetson/**TensorRT** (trtevec FP16/INT8 or
the ORT TensorRT EP), Intel **OpenVINO** EP, and ARM **XNNPACK** + INT8 PTQ.

---

## 7. Noise-robustness ablation (Step 5)

`experiments/robustness_ablation.py` corrupts the held-out test vibration
waveforms and re-scores XGBoost (re-extracting DSP features) vs the 1D-CNN
(re-normalize + ONNX), test n=300:

| Corruption | XGB wear MAE | DL wear MAE | XGB F1 | DL F1 |
| --- | --- | --- | --- | --- |
| clean | 0.0838 | 0.2366 | 0.675 | 0.271 |
| gaussian sd=0.1·rms | 0.0931 | 0.2493 | 0.632 | 0.176 |
| gaussian sd=0.25·rms | 0.1142 | 0.4445 | 0.595 | 0.084 |
| gaussian sd=0.5·rms | 0.1323 | 0.7423 | 0.576 | 0.077 |
| drift 0.5·rms | 0.0838 | 0.2366 | 0.675 | 0.271 |
| drift 1.0·rms | 0.0838 | 0.2366 | 0.675 | 0.271 |
| quantize 8-bit | 0.0843 | 0.2366 | 0.667 | 0.271 |
| quantize 4-bit | 0.0917 | 0.2351 | 0.619 | 0.199 |

**Findings.** (1) **Drift is fully rejected** — both models are numerically
unchanged from clean, because the shared `preprocess` front-end (linear detrend +
100 Hz high-pass) removes sub-band baseline wander before either model sees it: a
clean validation of the DSP design. (2) **Quantization** is benign to 8-bit and
only mildly harmful at 4-bit. (3) **Gaussian noise** degrades the raw-waveform
1D-CNN far more (wear MAE 0.24→0.74, F1 0.27→0.08 at sd=0.5) than the XGBoost
path (MAE 0.08→0.13, F1 0.68→0.58), because integrated band energies average out
white noise whereas the z-scored CNN — trained without noise augmentation —
amplifies it. Clear next step: **train the DL models with additive-noise
augmentation** to close this robustness gap. Plot: `experiments/plots/
robustness_ablation.png`.

---

## 8. Testing & verification

- **`tests/test_dl_models.py` (new, 11 tests, torch-gated via `importorskip`):**
  forward-pass shapes for all three models, `(B,L)` auto-unsqueeze, `get_model`
  factory + dict-config + unknown-name error, multi-task train/eval smoke on a
  synthetic loader, **ONNX round-trip parity** (≤1e-4) for all three, ONNX
  **dynamic batch/length** (export at B=1/L=4096, run at B=5/L=6000), and an
  end-to-end `SignalProcessor.get_normalized_waveform → 1D-CNN → ONNX →
  ONNXPerceptor` integration (softmax sums to 1).
- **`tests/test_signal_processor.py` (+4):** `get_normalized_waveform` (float32,
  mean≈0/std≈1), `compute_spectrogram` (2-D `(513, n_time)`, finite),
  `dl.normalize_for_dl` default, and the **low-`n_harmonics=1`** harmonic bug-fix
  guard.
- **Full suite: `69 passed`** (54 pre-existing + 4 DSP + 11 DL), 5.5 s.
- **`scripts/validate_simulators.py --no-show`: OVERALL PASS** (TPF <0.4% error,
  RMS & temperature monotonic in wear, finite/dtype/no-clip, reproducible).
- **Backward-compat:** a 12-sample no-flag run yields a **45-column** manifest
  with `has DSP feats: False` — schema unchanged.

---

## 9. Bugs / tuning encountered and resolved

1. **Model-name vs input-kind mismatch.** `prepare_dl_data` initially received the
   model name (`"1dcnn"`) as the dataset `mode`, but the dataset branches on the
   *input kind* (`"waveform"`), yielding `KeyError: 'waveform'` in the first
   batch. Added a name→kind alias map at the top of `prepare_dl_data`.
2. **ONNX exporter dependency.** torch 2.12 defaults to the dynamo ONNX exporter,
   which requires `onnxscript` (not in the `dl` extra) → `ModuleNotFoundError`.
   Switched to the legacy exporter (`dynamo=False`), which also consumes
   `dynamic_axes` directly. Suppressed the (benign) legacy-exporter
   `DeprecationWarning` locally so test output stays clean.
3. **Autograd scalar warning.** `float(loss)` on a grad-tracking tensor warned;
   switched to `loss.item()`.
4. **Console encoding.** The robustness labels used `σ`, which the Windows cp1252
   console can't encode (`UnicodeEncodeError`); replaced with ASCII `sd=...`.
5. **OpenMP duplicate-runtime notice.** Importing `torch` and `xgboost` together
   in a throwaway one-liner triggered `OMP: Error #15` (multiple `libiomp5md.dll`
   on the Anaconda stack). It did **not** affect any training/benchmark run
   (train_dl imports torch first, then xgboost lazily for the comparison, and all
   runs completed with valid numbers); `KMP_DUPLICATE_LIB_OK=TRUE` is the known
   workaround if it recurs.
6. **Harmonic fix (recent commit).** No issues found — the low-`n_harmonics`
   graceful path behaves exactly as intended (see §0); the new DL frequency-path
   reuse inherits the fix correctly.

---

## 10. Documentation & repo updates

- `pyproject.toml`: new `[project.optional-dependencies] dl = [torch>=2.2,
  onnx>=1.16, onnxruntime>=1.18]`; `requirements.txt` updated with the `dl`
  extra + `pip install -e ".[ml,dl]"` note.
- Root `README.md`: status table (`models/` ✅ DL+ONNX, `dsp/` DL methods),
  quickstart step 6 (DL train + benchmark + robustness), repo layout, tech stack,
  roadmap (Day 3 ✅, Day 4 preview).
- `dsp/README.md`: "Deep-learning input methods" section; `models/README.md`:
  "Implemented (Day 3): deep learning + ONNX" with the honest result summary;
  `experiments/README.md`: notebooks / robustness / artifacts index;
  `models/__init__.py`: documents lazy DL/ml imports.
- `experiments/notebooks/dl_training.ipynb` (new): reproduces the model-vs-baseline
  comparison, ONNX latency, and robustness tables from the saved JSONs (executes
  end-to-end via `nbconvert`, exit 0).

---

## 11. Verification snapshot (commands + key stdout)

```
$ python -m pip install "torch>=2.2" "onnx>=1.16" "onnxruntime>=1.18"
Successfully installed onnx-1.22.0 onnxruntime-1.27.0 torch-2.12.1

$ python -m pytest -q                       # baseline before changes
54 passed

# harmonic bug-fix check (n_harmonics=1)
h2 0.0  h3 0.0  fund 3.915068...  total 3.915068...  all finite: True

$ python scripts/generate_dataset.py --num-samples 2500 --output-dir data/dl_v1 \
      --seed 42 --duration-s 1.0 --extract-features --compute-spectrogram
Done in 59.8s ; DSP FEATURES: 30 ; On-disk size: 435.65 MB ; Wrote DL input recipe -> data\dl_v1\dl_config.json

$ python models/baseline.py --data-dir data/dl_v1
REGRESSION wear_level : MAE=0.0810 R2=0.8820 ; CLASSIFICATION health_state : Accuracy=0.6800 macro-F1=0.6728

$ python models/train_dl.py --model 1dcnn       --data-dir data/dl_v1 --epochs 40   # wear R2=0.106
$ python models/train_dl.py --model spectrogram --data-dir data/dl_v1 --epochs 40   # wear R2=0.077
$ python models/train_dl.py --model fusion      --data-dir data/dl_v1 --epochs 40   # wear R2=0.352, acc 0.402

$ python scripts/benchmark_onnx.py
1dcnn 0.371ms  spectrogram 0.423ms  fusion 0.381ms  (single p50) ; xgboost end-to-end 9.571ms

$ python experiments/robustness_ablation.py --data-dir data/dl_v1 --max-samples 300
drift -> unchanged ; gaussian 0.5*rms -> XGB MAE 0.132 / DL MAE 0.742

$ python -m pytest                          # after all changes
69 passed

$ python scripts/validate_simulators.py --no-show
OVERALL: PASS

# backward-compat (no flags): manifest 45 cols, has DSP feats: False
# import models -> torch in sys.modules: False
```

---

## 12. Immediate next steps / TODOs for Day 4

1. **Streaming `Perceptor`** — a stateful class that windows a live vibration
   stream, calls `get_normalized_waveform` / `compute_spectrogram`, and runs the
   ONNX model via `ONNXPerceptor` at fixed cadence (reuse the dynamic-length axis).
2. **FastAPI service** — `/infer` (single chunk) and `/batch` endpoints wrapping
   the ONNX models + the XGBoost baseline, returning wear/cycle/quality + health
   probabilities; feed the downstream cost/nesting optimizer.
3. **Close the DL robustness/accuracy gap** — train with additive-noise + gain
   augmentation, try `normalize_for_dl="none"` (amplitude-aware) or a hybrid that
   keeps a log-RMS scalar channel, and scale to 10k+ samples.
4. **Streamlit dashboard** — live waveform/spectrogram, predicted wear/health,
   latency, and the robustness curves.
5. **Edge packaging** — TensorRT/OpenVINO conversion of the ONNX artifacts + a
   Docker image; INT8 PTQ.

Session complete — DL + fusion + ONNX ready for Day 4 streaming integration.
