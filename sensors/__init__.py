"""Argus Panoptes - synthetic sensor (physics-informed data generation) package.

Public API
----------
>>> from sensors import SawVibrationSimulator, ThermalSimulator, load_config
>>> cfg = load_config()
>>> vib = SawVibrationSimulator(cfg)
>>> t, accel_g, meta = vib.generate(duration_s=2.0, wear=0.3, seed=0)

This package generates physics-informed vibration and thermal signals for
aluminum sawing / CNC machining, with auto-generated labels (wear, RUL,
cycle-time factor, quality score, anomaly / health state) and rich metadata,
ready to be logged to Parquet for the DSP / ML / integration layers.
"""

from __future__ import annotations

from sensors.config import (
    LabelConfig,
    MachiningConfig,
    SensorConfig,
    ThermalConfig,
    VibrationConfig,
    load_config,
)
from sensors.thermal_simulator import ThermalSimulator
from sensors.vibration_simulator import SawVibrationSimulator

__version__ = "0.1.0"

__all__ = [
    "SawVibrationSimulator",
    "ThermalSimulator",
    "load_config",
    "SensorConfig",
    "VibrationConfig",
    "ThermalConfig",
    "MachiningConfig",
    "LabelConfig",
    "__version__",
]
