"""DSP & feature-extraction layer for Argus Panoptes.

Public API
----------
>>> from dsp import SignalProcessor, load_processor_config
>>> sp = SignalProcessor()
>>> from sensors import SawVibrationSimulator
>>> t, accel, meta = SawVibrationSimulator().generate(duration_s=2.0, wear=0.6, seed=0)
>>> result = sp.process(accel, fs=meta["fs_hz"], metadata=meta)
>>> result["features"]["fd_tpf_band_energy"]  # doctest: +SKIP

The modular :class:`SignalProcessor` provides detrend / band-pass / (optional)
normalization preprocessing, time-domain features (RMS, crest / shape / impulse
/ clearance / margin factors, kurtosis, skewness, peak-to-peak, zero-crossing
rate, envelope stats), frequency-domain features (Welch PSD, spectral centroid /
rolloff / flatness / bandwidth, dominant peak, and TPF-relative band energies)
and an optional STFT spectrogram, all driven by a YAML config that mirrors
``sensors/sensor_specs.yaml``.
"""

from __future__ import annotations

from dsp.config import (
    ProcessorConfig,
    PreprocessConfig,
    FrequencyConfig,
    StftConfig,
    DlConfig,
    load_processor_config,
)
from dsp.signal_processor import (
    FREQUENCY_DOMAIN_FEATURES,
    TIME_DOMAIN_FEATURES,
    SignalProcessor,
)

__version__ = "0.1.0"

__all__ = [
    "SignalProcessor",
    "load_processor_config",
    "ProcessorConfig",
    "PreprocessConfig",
    "FrequencyConfig",
    "StftConfig",
    "DlConfig",
    "TIME_DOMAIN_FEATURES",
    "FREQUENCY_DOMAIN_FEATURES",
    "__version__",
]
