# `dsp/` — Digital Signal Processing & Feature Extraction

> **Status:** v1 implemented (Step 2).

This module hosts the modular `SignalProcessor` class that turns raw
vibration/thermal waveforms produced by [`sensors/`](../sensors/README.md) into
model-ready **scalar features** (and, optionally, STFT spectrograms for the
Step 3 deep-learning path).

## Pipeline

1. **Preprocess** (`preprocess`) — detrend → zero-phase Butterworth band-pass
   (corners clamped below Nyquist, so the same config works for the 40.96 kHz
   vibration and 200 Hz thermal paths) → optional amplitude normalization.
   *Normalization defaults to `none`*: with blade wear both the tooth-impact
   amplitude and the broadband RMS grow, so absolute amplitude is a first-class
   wear feature and is preserved by default.
2. **Time-domain features** (`extract_time_domain_features`, prefix `td_`) —
   RMS, peak, peak-to-peak, crest / shape / impulse / clearance / margin
   factors, excess kurtosis, skewness, zero-crossing rate, Hilbert-envelope
   mean/std. These track *impulsiveness*, which rises as tooth-strike transients
   sharpen with wear.
3. **Frequency-domain features** (`extract_frequency_domain_features`, prefix
   `fd_`) — Welch PSD, total power, spectral centroid / rolloff / flatness /
   bandwidth, dominant peak, and the **TPF-relative band energies**: the
   tooth-pass fundamental band `[tpf·0.75, tpf·1.25]`, its 2×/3× harmonics, and a
   broadband high-frequency band. *Tooth-pass band energy rises with wear because
   per-tooth cutting force (impact amplitude) increases; the broadband band rises
   with stochastic chip formation and edge friction.*
4. **STFT** (`compute_stft`) — power spectrogram (linear or dB); off by default
   on the scalar path.

Config is YAML-driven (`dsp/processor_config.yaml`) and validated by
`pydantic` models in `dsp/config.py`, mirroring `sensors/sensor_specs.yaml`.

## Deep-learning input methods (Step 3)

Two convenience methods prepare model-ready inputs for the DL path without
touching the scalar-feature / Parquet schema:

- **`get_normalized_waveform(x, fs)`** — detrend → band-pass → the configured
  `dl.normalize_for_dl` normalization (default `"zscore"`), returned as 1-D
  `float32`. The tabular path keeps absolute amplitude (a wear feature), but the
  1D-CNN generalizes better on scale-invariant, per-chunk-normalized input.
- **`compute_spectrogram(x, fs)`** — the same normalization followed by
  `compute_stft`, returning just the **`(n_freq, n_time)` `float32`** log-power
  spectrogram (`n_freq = nperseg // 2 + 1`), i.e. the natural single-channel
  `(H, W)` input for the spectrogram-CNN.

Both are driven by the new `dl:` section of `processor_config.yaml`
(`normalize_for_dl`) and the CNN-tuned `stft:` defaults. They are consumed by
`models/dl_data.py` (on-the-fly, no extra storage) and reused for the ONNX edge
path. The scalar `process()` / `process_batch()` / Parquet feature schema are
unchanged.

## Usage

```python
from sensors import SawVibrationSimulator
from dsp import SignalProcessor

sim = SawVibrationSimulator()
sp = SignalProcessor()  # loads dsp/processor_config.yaml defaults

# TPF-relative band energy tracks blade wear:
for wear in (0.0, 0.5, 1.0):
    _, accel, meta = sim.generate(duration_s=3.0, wear=wear, seed=7)
    feats = sp.process(accel, fs=meta["fs_hz"], metadata=meta)["features"]
    print(f"wear={wear}: rms={feats['td_rms']:.3f}  "
          f"tpf_band_energy={feats['fd_tpf_band_energy']:.3f}  "
          f"broadband={feats['fd_broadband_energy']:.4f}")
# wear=0.0: rms=2.293  tpf_band_energy=0.775  broadband=0.0047
# wear=0.5: rms=5.153  tpf_band_energy=3.917  broadband=0.0241
# wear=1.0: rms=9.147  tpf_band_energy=12.345 broadband=0.0750
```

Batch extraction into a `pandas.DataFrame` (one row per waveform):

```python
df = sp.process_batch(list_of_waveforms, fs=40960.0, metadatas=list_of_meta,
                      prefix="vib_")
```

## Integration points

- **`scripts/generate_dataset.py --extract-features`** runs `SignalProcessor`
  on each vibration waveform and merges the `vib_*` scalar features into
  `manifest.parquet` (raw waveform record columns are untouched).
- **`models/baseline.py`** consumes those feature columns for XGBoost baselines
  and time-vs-frequency ablations (feature groups selected by the `td_`/`fd_`
  prefixes).
- **`models/dl_data.py` (Step 3)** uses `get_normalized_waveform` for the 1D-CNN
  / fusion branches and `compute_spectrogram` for the spectrogram-CNN, computed
  on-the-fly from the raw `records/` waveforms.
