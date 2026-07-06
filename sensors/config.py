"""Typed configuration models and loader for the Argus Panoptes sensors module.

The :func:`load_config` function reads ``sensor_specs.yaml`` (or a caller-supplied
path / dict) into a validated :class:`SensorConfig` pydantic model. Every field
is strongly typed with sensible industrial defaults so the simulators can be
constructed with zero arguments, yet remain fully overridable via:

* an explicit ``path`` to an alternative YAML file,
* an ``overrides`` mapping (deep-merged onto the file values), or
* the ``ARGUS_SENSOR_SPECS`` environment variable pointing at a YAML file.

Example
-------
>>> from sensors.config import load_config
>>> cfg = load_config()
>>> cfg.vibration.fs_hz
40960
>>> cfg = load_config(overrides={"vibration": {"fs_hz": 20480}})
>>> cfg.vibration.fs_hz
20480
"""

from __future__ import annotations

import os
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

__all__ = [
    "SensorConfig",
    "VibrationConfig",
    "ThermalConfig",
    "MachiningConfig",
    "LabelConfig",
    "AlloySpec",
    "load_config",
    "DEFAULT_SPECS_PATH",
]

DEFAULT_SPECS_PATH: Path = Path(__file__).with_name("sensor_specs.yaml")
_ENV_VAR = "ARGUS_SENSOR_SPECS"


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------
class _Strict(BaseModel):
    """Base model that forbids unknown keys to catch config typos early."""

    model_config = ConfigDict(extra="forbid")


class StructuralMode(_Strict):
    """A single lightly-damped structural resonance of the machine."""

    center_hz: float = Field(gt=0)
    q: float = Field(gt=0, description="Quality factor (center/bandwidth).")
    gain: float = Field(gt=0, description="Linear gain applied at the mode center.")


class PulseSpec(_Strict):
    """Per-tooth engagement pulse shape parameters."""

    shape: str = "exp_decay_sine"
    ring_freq_hz: float = Field(default=2000.0, gt=0)
    damping: float = Field(default=180.0, gt=0)
    width_frac: float = Field(default=0.6, gt=0, le=1.0)

    @field_validator("shape")
    @classmethod
    def _check_shape(cls, v: str) -> str:
        allowed = {"half_sine", "exp_decay_sine"}
        if v not in allowed:
            raise ValueError(f"pulse.shape must be one of {allowed}, got {v!r}")
        return v


class VibrationConfig(_Strict):
    """IEPE accelerometer + machine-dynamics configuration."""

    type: str = "IEPE_accelerometer"
    mounting: str = "stud_or_magnet_on_blade_guide_or_spindle_housing"
    sensitivity_mV_per_g: float = Field(default=100.0, gt=0)
    bandwidth_hz_up_to: float = Field(default=20000.0, gt=0)
    resonant_freq_hz: float = Field(default=35000.0, gt=0)
    noise_floor_g_rms: float = Field(default=0.001, ge=0)
    fs_hz: float = Field(default=40960.0, gt=0)
    axes: int = Field(default=1, ge=1)
    clip_g: float = Field(default=50.0, gt=0)
    accel_g_per_kN: float = Field(default=8.0, gt=0)
    broadband_base_g: float = Field(default=0.05, ge=0)
    broadband_wear_gain: float = Field(default=3.0, ge=0)
    broadband_exponent: float = Field(default=1.0, ge=0)
    structural_modes: list[StructuralMode] = Field(default_factory=list)
    pulse: PulseSpec = Field(default_factory=PulseSpec)
    harmonics: list[float] = Field(default_factory=lambda: [1.0, 0.5, 0.3, 0.15])

    @field_validator("harmonics")
    @classmethod
    def _check_harmonics(cls, v: list[float]) -> list[float]:
        if not v:
            raise ValueError("harmonics must contain at least the fundamental (1x).")
        if any(h < 0 for h in v):
            raise ValueError("harmonic amplitudes must be non-negative.")
        return v


class ThermalModel(_Strict):
    """Lumped first-order thermal model of the cut zone."""

    ambient_c: float = 25.0
    heat_partition: float = Field(default=0.75, ge=0, le=1.0)
    loss_coefficient_w_per_c: float = Field(default=2.2, gt=0)
    thermal_mass_j_per_c: float = Field(default=9.0, gt=0)
    friction_power_gain_w: float = Field(default=260.0, ge=0)
    max_temp_c: float = Field(default=600.0, gt=0)


class ThermalConfig(_Strict):
    """IR pyrometer + lumped thermal model configuration."""

    type: str = "IR_pyrometer_cut_zone"
    focus: str = "tool_chip_interface_or_rake_face"
    bandwidth_hz: float = Field(default=100.0, gt=0)
    noise_std_c: float = Field(default=0.5, ge=0)
    fs_hz: float = Field(default=200.0, gt=0)
    emissivity: float = Field(default=0.4, gt=0, le=1.0)
    thermal_model: ThermalModel = Field(default_factory=ThermalModel)


class AlloySpec(_Strict):
    """Specific-energy model for one aluminum alloy."""

    base_specific_energy_j_per_mm3: float = Field(gt=0)
    wear_energy_gain: float = Field(ge=0)


class MachiningSampling(_Strict):
    """Ranges used by the dataset generator to sample the parameter space."""

    alloys: list[str] = Field(default_factory=lambda: ["6061", "7075"])
    blade_speed_sfpm: tuple[float, float] = (500.0, 1200.0)
    feed_per_tooth_mm: tuple[float, float] = (0.05, 0.40)
    depth_mm: tuple[float, float] = (5.0, 50.0)
    kerf_width_mm: tuple[float, float] = (2.0, 5.0)
    num_teeth: tuple[int, int] = (40, 120)
    wear: tuple[float, float] = (0.0, 1.0)
    duration_s: tuple[float, float] = (2.0, 5.0)


class MachiningConfig(_Strict):
    """Default machining parameters, force-model coefficients, sampling ranges."""

    saw_type: str = "circular"
    alloys: dict[str, AlloySpec]
    default_alloy: str = "6061"
    blade_speed_sfpm: float = Field(default=800.0, gt=0)
    blade_diameter_mm: float = Field(default=350.0, gt=0)
    num_teeth: int = Field(default=80, gt=0)
    tpi: int = Field(default=4, gt=0)
    band_speed_sfpm: float = Field(default=800.0, gt=0)
    feed_per_tooth_mm: float = Field(default=0.12, gt=0)
    depth_mm: float = Field(default=25.0, gt=0)
    kerf_width_mm: float = Field(default=3.0, gt=0)
    friction_wear_gain: float = Field(default=0.6, ge=0)
    force_wear_gain: float = Field(default=0.5, ge=0)
    sampling: MachiningSampling = Field(default_factory=MachiningSampling)

    @field_validator("saw_type")
    @classmethod
    def _check_saw_type(cls, v: str) -> str:
        if v not in {"circular", "band"}:
            raise ValueError(f"saw_type must be 'circular' or 'band', got {v!r}")
        return v


class LabelConfig(_Strict):
    """Coefficients for deterministic label generation."""

    rul_max_cycles: int = Field(default=1000, gt=0)
    cycle_time_wear_gain: float = Field(default=0.30, ge=0)
    quality_wear_gain: float = Field(default=0.50, ge=0)
    quality_kurtosis_gain: float = Field(default=0.10, ge=0)
    anomaly_wear_threshold: float = Field(default=0.85, ge=0, le=1.0)
    anomaly_temp_threshold_c: float = Field(default=350.0, gt=0)


class SensorConfig(_Strict):
    """Top-level, fully-typed configuration for the sensors module."""

    simulator_version: str = "0.1.0"
    random_seed: int = 42
    vibration: VibrationConfig = Field(default_factory=VibrationConfig)
    thermal: ThermalConfig = Field(default_factory=ThermalConfig)
    machining: MachiningConfig
    labels: LabelConfig = Field(default_factory=LabelConfig)

    # -- Convenience accessors -------------------------------------------------
    def alloy(self, name: str | None = None) -> AlloySpec:
        """Return the :class:`AlloySpec` for ``name`` (or the default alloy)."""
        key = name or self.machining.default_alloy
        try:
            return self.machining.alloys[key]
        except KeyError as exc:  # pragma: no cover - defensive
            valid = list(self.machining.alloys)
            raise KeyError(f"Unknown alloy {key!r}; known alloys: {valid}") from exc


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
        raise FileNotFoundError(f"Sensor spec file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"Sensor spec file {path} must contain a top-level mapping.")
    return data


def load_config(
    path: str | os.PathLike[str] | None = None,
    *,
    overrides: dict[str, Any] | None = None,
) -> SensorConfig:
    """Load and validate the sensor/machining configuration.

    Parameters
    ----------
    path:
        Path to a YAML spec file. If ``None``, uses the ``ARGUS_SENSOR_SPECS``
        environment variable when set, otherwise the packaged
        :data:`DEFAULT_SPECS_PATH`.
    overrides:
        Optional nested mapping deep-merged on top of the file values (useful for
        programmatic sweeps and tests).

    Returns
    -------
    SensorConfig
        A fully validated configuration object.

    Raises
    ------
    FileNotFoundError
        If the resolved spec path does not exist.
    pydantic.ValidationError
        If the merged configuration fails validation.
    """
    resolved = Path(path) if path is not None else Path(os.environ.get(_ENV_VAR, DEFAULT_SPECS_PATH))
    data = _read_yaml(resolved)
    if overrides:
        data = _deep_merge(data, overrides)
    return SensorConfig.model_validate(data)
