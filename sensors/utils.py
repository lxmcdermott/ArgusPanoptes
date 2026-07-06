"""Shared physics helpers and deterministic label functions.

This module contains the *pure* building blocks used by both simulators:

* **Kinematics** - :func:`calculate_tpf` (tooth-pass frequency), :func:`rpm_from_sfpm`,
  :func:`cutting_velocity_m_s`.
* **Machining physics** - :func:`specific_energy`, :func:`compute_force_model`
  (chip-load -> specific-energy -> power -> force, with wear modulation).
* **Signal helpers** - :func:`generate_colored_noise`, :func:`apply_sensor_noise`,
  :func:`rms`, :func:`kurtosis`, :func:`crest_factor`.
* **Labels** - :func:`generate_labels` and its deterministic components
  (wear level, RUL proxy, cycle-time factor, quality score, anomaly / health state).

All functions are side-effect free (aside from an explicit ``rng`` argument) so
they are trivially unit-testable and reproducible.

References
----------
Specific cutting energy ``u`` for aluminum is ~0.4-1.4 J/mm^3. Note that
1 J/mm^3 == 1000 N/mm^2, i.e. cutting power ``P = u * MRR`` (W, with MRR in
mm^3/s) and cutting force ``F = P / V_c`` (N).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:  # avoid import cycle at runtime
    from sensors.config import SensorConfig

__all__ = [
    "FloatArray",
    "MM_PER_FOOT",
    "ForceModel",
    "rpm_from_sfpm",
    "cutting_velocity_m_s",
    "calculate_tpf",
    "specific_energy",
    "compute_force_model",
    "generate_colored_noise",
    "apply_sensor_noise",
    "rms",
    "kurtosis",
    "crest_factor",
    "generate_labels",
]

FloatArray = npt.NDArray[np.float64]

MM_PER_FOOT: float = 304.8
_FT_PER_MIN_TO_M_PER_S: float = 0.3048 / 60.0


# ---------------------------------------------------------------------------
# Kinematics
# ---------------------------------------------------------------------------
def rpm_from_sfpm(blade_speed_sfpm: float, blade_diameter_mm: float) -> float:
    """Convert surface speed (SFPM) to spindle RPM for a circular blade.

    ``V_c = pi * D * RPM``  =>  ``RPM = V_c / (pi * D)``.
    """
    if blade_diameter_mm <= 0:
        raise ValueError("blade_diameter_mm must be positive.")
    diameter_ft = blade_diameter_mm / MM_PER_FOOT
    return blade_speed_sfpm / (np.pi * diameter_ft)


def cutting_velocity_m_s(blade_speed_sfpm: float) -> float:
    """Surface cutting velocity in m/s from SFPM."""
    return blade_speed_sfpm * _FT_PER_MIN_TO_M_PER_S


def calculate_tpf(
    *,
    saw_type: str = "circular",
    blade_speed_sfpm: float,
    num_teeth: int | None = None,
    blade_diameter_mm: float | None = None,
    tpi: int | None = None,
) -> float:
    """Compute the tooth-pass frequency (Hz) exactly.

    Circular saw
        ``RPM = SFPM / (pi * D_ft)``; ``TPF = RPM * num_teeth / 60``.
    Band saw
        Teeth pass a fixed point at ``TPF = V_s[ft/min] * 12[in/ft] * TPI / 60``.

    Parameters
    ----------
    saw_type:
        ``"circular"`` or ``"band"``.
    blade_speed_sfpm:
        Surface / blade linear speed in surface-feet-per-minute.
    num_teeth, blade_diameter_mm:
        Required for a circular saw.
    tpi:
        Teeth-per-inch, required for a band saw.

    Returns
    -------
    float
        Tooth-pass frequency in Hz.
    """
    if saw_type == "circular":
        if num_teeth is None or blade_diameter_mm is None:
            raise ValueError("circular saw requires num_teeth and blade_diameter_mm.")
        rpm = rpm_from_sfpm(blade_speed_sfpm, blade_diameter_mm)
        return rpm * num_teeth / 60.0
    if saw_type == "band":
        if tpi is None:
            raise ValueError("band saw requires tpi.")
        return blade_speed_sfpm * 12.0 * tpi / 60.0
    raise ValueError(f"Unknown saw_type {saw_type!r}; expected 'circular' or 'band'.")


# ---------------------------------------------------------------------------
# Machining physics
# ---------------------------------------------------------------------------
def specific_energy(base_u: float, wear_energy_gain: float, wear: float) -> float:
    """Wear-adjusted specific cutting energy ``u`` (J/mm^3).

    A dull edge rubs and forms built-up-edge on gummy aluminum, so the specific
    energy rises with wear: ``u = base_u * (1 + wear_energy_gain * wear)``.
    """
    return base_u * (1.0 + wear_energy_gain * max(0.0, wear))


@dataclass(frozen=True)
class ForceModel:
    """Derived cutting mechanics for one operating point (all averages)."""

    tpf_hz: float
    cutting_velocity_m_s: float
    feed_velocity_mm_s: float
    material_removal_rate_mm3_s: float
    specific_energy_j_per_mm3: float
    cutting_power_w: float
    avg_cutting_force_n: float
    per_tooth_force_n: float
    force_wear_multiplier: float
    rpm: float | None = None
    extras: dict[str, float] = field(default_factory=dict)


def compute_force_model(
    cfg: "SensorConfig",
    *,
    alloy: str | None = None,
    blade_speed_sfpm: float | None = None,
    num_teeth: int | None = None,
    feed_per_tooth_mm: float | None = None,
    depth_mm: float | None = None,
    kerf_width_mm: float | None = None,
    wear: float = 0.0,
) -> ForceModel:
    """Build the physics-informed cutting-force model for an operating point.

    The chain of reasoning (see module docstring):

    1. ``feed_velocity = feed_per_tooth * TPF`` (mm/s).
    2. ``MRR = kerf_width * depth * feed_velocity`` (mm^3/s).
    3. ``u = specific_energy(alloy, wear)`` (J/mm^3).
    4. ``P = u * MRR`` (W)  ->  ``F_avg = P / V_c`` (N).
    5. A direct force/friction wear multiplier
       ``m = 1 + force_wear_gain*wear + friction_wear_gain*wear`` scales the
       force to capture edge rubbing beyond the specific-energy effect.
    6. Per-tooth force ``F_tooth = F_avg * m`` drives the impulse amplitudes.

    Any operating-point argument left as ``None`` falls back to the machining
    defaults in ``cfg``.
    """
    m = cfg.machining
    alloy_name = alloy or m.default_alloy
    blade_speed_sfpm = m.blade_speed_sfpm if blade_speed_sfpm is None else blade_speed_sfpm
    num_teeth = m.num_teeth if num_teeth is None else int(num_teeth)
    feed_per_tooth_mm = m.feed_per_tooth_mm if feed_per_tooth_mm is None else feed_per_tooth_mm
    depth_mm = m.depth_mm if depth_mm is None else depth_mm
    kerf_width_mm = m.kerf_width_mm if kerf_width_mm is None else kerf_width_mm
    wear = float(np.clip(wear, 0.0, 1.0))

    if m.saw_type == "circular":
        tpf = calculate_tpf(
            saw_type="circular",
            blade_speed_sfpm=blade_speed_sfpm,
            num_teeth=num_teeth,
            blade_diameter_mm=m.blade_diameter_mm,
        )
        rpm: float | None = rpm_from_sfpm(blade_speed_sfpm, m.blade_diameter_mm)
    else:
        tpf = calculate_tpf(saw_type="band", blade_speed_sfpm=blade_speed_sfpm, tpi=m.tpi)
        rpm = None

    v_c = cutting_velocity_m_s(blade_speed_sfpm)
    feed_velocity = feed_per_tooth_mm * tpf  # mm/s
    mrr = kerf_width_mm * depth_mm * feed_velocity  # mm^3/s

    alloy_spec = cfg.alloy(alloy_name)
    u_eff = specific_energy(
        alloy_spec.base_specific_energy_j_per_mm3, alloy_spec.wear_energy_gain, wear
    )
    power_w = u_eff * mrr  # J/mm^3 * mm^3/s == W
    avg_force = power_w / v_c if v_c > 0 else 0.0  # N

    multiplier = 1.0 + m.force_wear_gain * wear + m.friction_wear_gain * wear
    per_tooth_force = avg_force * multiplier

    return ForceModel(
        tpf_hz=tpf,
        cutting_velocity_m_s=v_c,
        feed_velocity_mm_s=feed_velocity,
        material_removal_rate_mm3_s=mrr,
        specific_energy_j_per_mm3=u_eff,
        cutting_power_w=power_w,
        avg_cutting_force_n=avg_force,
        per_tooth_force_n=per_tooth_force,
        force_wear_multiplier=multiplier,
        rpm=rpm,
        extras={
            "alloy": alloy_name,  # type: ignore[dict-item]
            "num_teeth": float(num_teeth),
            "feed_per_tooth_mm": feed_per_tooth_mm,
            "depth_mm": depth_mm,
            "kerf_width_mm": kerf_width_mm,
            "blade_speed_sfpm": blade_speed_sfpm,
            "wear": wear,
        },
    )


# ---------------------------------------------------------------------------
# Signal helpers
# ---------------------------------------------------------------------------
def generate_colored_noise(
    n: int,
    fs_hz: float,
    *,
    exponent: float = 1.0,
    rms_g: float = 1.0,
    rng: np.random.Generator | None = None,
) -> FloatArray:
    """Generate ``1/f**exponent`` colored Gaussian noise scaled to ``rms_g``.

    ``exponent=0`` -> white, ``exponent=1`` -> pink, ``exponent=2`` -> brown.
    Implemented by shaping the FFT magnitude of white noise, which is fast and
    fully vectorized.
    """
    if n <= 0:
        return np.zeros(0, dtype=np.float64)
    if rng is None:
        rng = np.random.default_rng()
    if rms_g <= 0:
        return np.zeros(n, dtype=np.float64)

    white = rng.standard_normal(n)
    spectrum = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs_hz)
    scaling = np.ones_like(freqs)
    nonzero = freqs > 0
    scaling[nonzero] = 1.0 / np.power(freqs[nonzero], exponent / 2.0)
    scaling[~nonzero] = 0.0  # drop DC to keep the noise zero-mean
    shaped = np.fft.irfft(spectrum * scaling, n=n)

    current = float(np.sqrt(np.mean(shaped**2)))
    if current > 0:
        shaped *= rms_g / current
    return shaped.astype(np.float64)


def apply_sensor_noise(
    signal: FloatArray,
    noise_floor_g_rms: float,
    *,
    rng: np.random.Generator | None = None,
) -> FloatArray:
    """Add white Gaussian sensor/electrical noise at the given RMS level (g)."""
    if noise_floor_g_rms <= 0:
        return signal
    if rng is None:
        rng = np.random.default_rng()
    return signal + rng.normal(0.0, noise_floor_g_rms, size=signal.shape)


def rms(x: FloatArray) -> float:
    """Root-mean-square of a signal."""
    if x.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64))))


def kurtosis(x: FloatArray, *, fisher: bool = True) -> float:
    """Excess (Fisher) kurtosis; 0 for a Gaussian, high for impulsive signals."""
    if x.size < 2:
        return 0.0
    mean = float(np.mean(x))
    std = float(np.std(x))
    if std == 0:
        return 0.0
    k = float(np.mean(((x - mean) / std) ** 4))
    return k - 3.0 if fisher else k


def crest_factor(x: FloatArray) -> float:
    """Peak-to-RMS ratio; grows with impulsiveness (early bearing/tooth faults)."""
    r = rms(x)
    if r == 0:
        return 0.0
    return float(np.max(np.abs(x)) / r)


# ---------------------------------------------------------------------------
# Labels (deterministic, pure functions of the operating point + signal stats)
# ---------------------------------------------------------------------------
def _health_state(wear: float) -> str:
    if wear < 0.25:
        return "healthy"
    if wear < 0.55:
        return "monitor"
    if wear < 0.85:
        return "warning"
    return "critical"


def generate_labels(
    cfg: "SensorConfig",
    *,
    wear: float,
    force: ForceModel,
    signal_kurtosis: float = 0.0,
    cut_zone_temp_c: float | None = None,
) -> dict[str, float | int | str | bool]:
    """Produce the full deterministic label set for one sample.

    Labels
    ------
    wear_level : float
        Pass-through 0 (sharp) .. 1 (end-of-life).
    rul_fraction : float
        Remaining-useful-life fraction ``1 - wear`` (simple linear degradation).
    rul_cycles : int
        RUL expressed in remaining cuts (``rul_max_cycles * (1 - wear)``).
    cycle_time_factor : float
        Effective cycle-time multiplier ``>= 1`` (higher forces slow the cut).
    quality_score : float
        Part-quality proxy in ``[0, 1]`` (drops with wear and impulsiveness).
    anomaly_flag : bool / health_state : str
        Condition-monitoring flags.
    """
    lc = cfg.labels
    wear = float(np.clip(wear, 0.0, 1.0))

    rul_fraction = 1.0 - wear
    rul_cycles = int(round(lc.rul_max_cycles * rul_fraction))

    # Extra time from elevated forces relative to a sharp edge (multiplier - 1).
    force_time_term = 0.15 * max(0.0, force.force_wear_multiplier - 1.0)
    cycle_time_factor = 1.0 + lc.cycle_time_wear_gain * wear + force_time_term

    # Normalize kurtosis into ~[0, 1] (excess kurtosis of ~10 => strong penalty).
    norm_kurt = float(np.clip(max(0.0, signal_kurtosis) / 10.0, 0.0, 1.0))
    quality_score = float(
        np.clip(
            1.0 - lc.quality_wear_gain * wear - lc.quality_kurtosis_gain * norm_kurt,
            0.0,
            1.0,
        )
    )

    temp_anomaly = (
        cut_zone_temp_c is not None and cut_zone_temp_c >= lc.anomaly_temp_threshold_c
    )
    anomaly_flag = bool(wear >= lc.anomaly_wear_threshold or temp_anomaly)

    labels: dict[str, float | int | str | bool] = {
        "wear_level": wear,
        "rul_fraction": rul_fraction,
        "rul_cycles": rul_cycles,
        "cycle_time_factor": cycle_time_factor,
        "quality_score": quality_score,
        "health_state": _health_state(wear),
        "anomaly_flag": anomaly_flag,
    }
    if cut_zone_temp_c is not None:
        labels["thermal_anomaly_flag"] = bool(temp_anomaly)
    return labels
