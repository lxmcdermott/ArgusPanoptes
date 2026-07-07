"""Typed configuration models and loader for the Argus Panoptes DSP module.

Mirrors :mod:`sensors.config`: :func:`load_processor_config` reads
``processor_config.yaml`` (or a caller-supplied path / dict) into a validated
:class:`ProcessorConfig` pydantic model. Every field is strongly typed with
sensible defaults so :class:`~dsp.signal_processor.SignalProcessor` can be
constructed with zero arguments yet remain fully overridable via:

* an explicit ``path`` to an alternative YAML file,
* an ``overrides`` mapping (deep-merged onto the file values), or
* the ``ARGUS_PROCESSOR_SPECS`` environment variable pointing at a YAML file.

Example
-------
>>> from dsp.config import load_processor_config
>>> cfg = load_processor_config()
>>> cfg.frequency.welch_nperseg
8192
>>> cfg = load_processor_config(overrides={"frequency": {"n_harmonics": 4}})
>>> cfg.frequency.n_harmonics
4
"""

from __future__ import annotations

import os
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

__all__ = [
    "ProcessorConfig",
    "PreprocessConfig",
    "BandpassConfig",
    "FrequencyConfig",
    "StftConfig",
    "DlConfig",
    "load_processor_config",
    "DEFAULT_PROCESSOR_SPECS_PATH",
]

DEFAULT_PROCESSOR_SPECS_PATH: Path = Path(__file__).with_name("processor_config.yaml")
_ENV_VAR = "ARGUS_PROCESSOR_SPECS"


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------
class _Strict(BaseModel):
    """Base model that forbids unknown keys to catch config typos early."""

    model_config = ConfigDict(extra="forbid")


class BandpassConfig(_Strict):
    """Butterworth band-pass corners (zero-phase ``filtfilt``)."""

    enabled: bool = True
    low_hz: float = Field(default=100.0, ge=0)
    high_hz: float = Field(default=18000.0, gt=0)
    order: int = Field(default=4, ge=1, le=10)

    @field_validator("high_hz")
    @classmethod
    def _check_band(cls, v: float, info: Any) -> float:
        low = info.data.get("low_hz", 0.0)
        if v <= low:
            raise ValueError(f"bandpass.high_hz ({v}) must exceed low_hz ({low}).")
        return v


class PreprocessConfig(_Strict):
    """Detrend -> band-pass -> (optional) amplitude normalization."""

    detrend: str = "linear"
    bandpass: BandpassConfig = Field(default_factory=BandpassConfig)
    normalize: str = "none"

    @field_validator("detrend")
    @classmethod
    def _check_detrend(cls, v: str) -> str:
        allowed = {"linear", "constant", "none"}
        if v not in allowed:
            raise ValueError(f"preprocess.detrend must be one of {allowed}, got {v!r}")
        return v

    @field_validator("normalize")
    @classmethod
    def _check_normalize(cls, v: str) -> str:
        allowed = {"none", "zscore", "peak", "rms"}
        if v not in allowed:
            raise ValueError(f"preprocess.normalize must be one of {allowed}, got {v!r}")
        return v


class FrequencyConfig(_Strict):
    """Welch-PSD, spectral-shape, and TPF-relative-band parameters."""

    welch_nperseg: int = Field(default=8192, ge=16)
    welch_overlap: float = Field(default=0.5, ge=0.0, lt=1.0)
    window: str = "hann"
    rolloff_fraction: float = Field(default=0.95, gt=0.0, le=1.0)
    tpf_band_frac: float = Field(default=0.25, gt=0.0, lt=1.0)
    n_harmonics: int = Field(default=3, ge=1, le=10)
    broadband_band_hz: tuple[float, float] = (6000.0, 18000.0)

    @field_validator("broadband_band_hz")
    @classmethod
    def _check_broadband(cls, v: tuple[float, float]) -> tuple[float, float]:
        if v[1] <= v[0]:
            raise ValueError(f"broadband_band_hz must be (low, high) with high>low, got {v}")
        return v


class StftConfig(_Strict):
    """Short-time Fourier transform (spectrogram) parameters."""

    nperseg: int = Field(default=1024, ge=16)
    overlap: float = Field(default=0.75, ge=0.0, lt=1.0)
    window: str = "hann"
    log_scale: bool = True


class DlConfig(_Strict):
    """Deep-learning input-preparation options (Day-3 CNN / fusion paths).

    These are consumed only by the convenience methods
    :meth:`SignalProcessor.get_normalized_waveform` and
    :meth:`SignalProcessor.compute_spectrogram`; the scalar-feature / Parquet
    path is untouched, so enabling them never changes the tabular schema.
    """

    #: Per-chunk amplitude normalization for the DL paths. Unlike the tabular
    #: path (where absolute amplitude is a wear feature, hence
    #: ``preprocess.normalize="none"``), CNNs generalize better on
    #: scale-invariant inputs, so ``"zscore"`` is the recommended default.
    normalize_for_dl: str = "zscore"

    @field_validator("normalize_for_dl")
    @classmethod
    def _check_normalize_for_dl(cls, v: str) -> str:
        allowed = {"none", "zscore", "peak", "rms"}
        if v not in allowed:
            raise ValueError(f"dl.normalize_for_dl must be one of {allowed}, got {v!r}")
        return v


class ProcessorConfig(_Strict):
    """Top-level, fully-typed configuration for the DSP module."""

    processor_version: str = "0.1.0"
    preprocess: PreprocessConfig = Field(default_factory=PreprocessConfig)
    frequency: FrequencyConfig = Field(default_factory=FrequencyConfig)
    stft: StftConfig = Field(default_factory=StftConfig)
    dl: DlConfig = Field(default_factory=DlConfig)


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------
def _deep_merge(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``overrides`` onto a copy of ``base``."""
    out = deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = deepcopy(value)
    return out


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Processor spec file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"Processor spec file {path} must contain a top-level mapping.")
    return data


def load_processor_config(
    path: str | os.PathLike[str] | None = None,
    *,
    overrides: dict[str, Any] | None = None,
) -> ProcessorConfig:
    """Load and validate the DSP / feature-extraction configuration.

    Parameters
    ----------
    path:
        Path to a YAML spec file. If ``None``, uses the ``ARGUS_PROCESSOR_SPECS``
        environment variable when set, otherwise the packaged
        :data:`DEFAULT_PROCESSOR_SPECS_PATH`.
    overrides:
        Optional nested mapping deep-merged on top of the file values.

    Returns
    -------
    ProcessorConfig
        A fully validated configuration object.
    """
    resolved = (
        Path(path)
        if path is not None
        else Path(os.environ.get(_ENV_VAR, DEFAULT_PROCESSOR_SPECS_PATH))
    )
    data = _read_yaml(resolved)
    if overrides:
        data = _deep_merge(data, overrides)
    return ProcessorConfig.model_validate(data)
