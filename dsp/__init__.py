"""DSP & feature-extraction layer for Argus Panoptes.

Planned (Day 1-2 of the execution plan): a modular ``SignalProcessor`` class
providing detrend / bandpass / normalization preprocessing, time-domain
features (RMS, crest factor, kurtosis, skewness, peak-to-peak), frequency-domain
features (Welch PSD, band energies, spectral centroid/rolloff/flatness, dominant
peaks) and time-frequency STFT spectrograms, all driven by YAML config.

This package is intentionally a scaffold for v1; see the technical plan
(Key Technical Component #2) for the full specification.
"""

__version__ = "0.0.0"
