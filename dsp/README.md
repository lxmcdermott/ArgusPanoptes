# `dsp/` — Digital Signal Processing & Feature Extraction

> **Status:** scaffold (implemented in Day 1–2 of the execution plan).

This module will host the modular `SignalProcessor` class that turns raw
vibration/thermal waveforms produced by [`sensors/`](../sensors/README.md) into
model-ready features.

## Planned scope (per technical plan §2)

- **Preprocessing:** detrend, bandpass (`scipy.signal`), normalization.
- **Time-domain features:** RMS, crest factor, kurtosis, skewness, peak-to-peak.
- **Frequency-domain features:** Welch PSD, band energies, spectral
  centroid / rolloff / flatness, dominant peaks (tooth-pass tracking).
- **Time-frequency:** STFT spectrograms for CNN inputs.
- **Config-driven** via YAML, mirroring `sensors/sensor_specs.yaml`.
