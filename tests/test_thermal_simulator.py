"""Tests for :class:`sensors.thermal_simulator.ThermalSimulator`."""

from __future__ import annotations

import numpy as np
import pytest

from sensors import ThermalSimulator


# --------------------------------------------------------------------------- #
# Wear monotonicity & physical ranges
# --------------------------------------------------------------------------- #
def test_steady_state_temp_monotonic_in_wear(therm):
    temps = [
        therm.generate(duration_s=6.0, wear=w, seed=1)[2]["steady_state_temp_c"]
        for w in (0.0, 0.25, 0.5, 0.75, 1.0)
    ]
    assert all(b > a for a, b in zip(temps, temps[1:]))


def test_temp_in_physical_range(therm):
    """Aluminum cut-zone temps should stay in a believable band."""
    for w in (0.0, 0.5, 1.0):
        _, temp, meta = therm.generate(duration_s=8.0, wear=w, seed=1)
        assert 20.0 < meta["steady_state_temp_c"] < 500.0
        assert np.all(temp < therm.model.max_temp_c)


def test_temp_never_exceeds_clamp(therm):
    _, temp, _ = therm.generate(
        duration_s=5.0,
        params={"depth_mm": 50, "feed_per_tooth_mm": 0.4, "blade_speed_sfpm": 1200},
        wear=1.0,
        seed=0,
    )
    assert np.all(temp <= therm.model.max_temp_c)


# --------------------------------------------------------------------------- #
# Thermal dynamics
# --------------------------------------------------------------------------- #
def test_transient_rises_from_ambient(therm):
    """Temperature should start near ambient and rise over the cut."""
    _, temp, meta = therm.generate(duration_s=8.0, wear=0.5, seed=0)
    ambient = meta["ambient_c"]
    assert temp[0] == pytest.approx(ambient, abs=3.0 * therm.thermal.noise_std_c + 1e-6)
    assert meta["final_temp_c"] > temp[0]


def test_approaches_steady_state(therm):
    """After several time constants the mean should near the steady-state value."""
    _, _, meta = therm.generate(duration_s=40.0, wear=0.5, seed=0)
    assert meta["final_temp_c"] == pytest.approx(meta["steady_state_temp_c"], rel=0.05)


def test_time_constant_positive(therm):
    _, _, meta = therm.generate(duration_s=2.0, wear=0.0, seed=0)
    assert meta["time_constant_s"] > 0.0


def test_friction_heat_scales_with_wear(therm):
    q0 = therm.generate(duration_s=2.0, wear=0.0, seed=0)[2]["q_friction_w"]
    q1 = therm.generate(duration_s=2.0, wear=1.0, seed=0)[2]["q_friction_w"]
    assert q0 == pytest.approx(0.0)
    assert q1 > q0


# --------------------------------------------------------------------------- #
# Reproducibility / structure / validation
# --------------------------------------------------------------------------- #
def test_reproducibility_same_seed(therm):
    _, t1, _ = therm.generate(duration_s=2.0, wear=0.3, seed=55)
    _, t2, _ = therm.generate(duration_s=2.0, wear=0.3, seed=55)
    assert np.array_equal(t1, t2)


def test_signal_length_and_dtype(therm, config):
    fs = config.thermal.fs_hz
    t, temp, meta = therm.generate(duration_s=3.0, wear=0.1, seed=0)
    assert temp.shape == t.shape == (int(round(3.0 * fs)),)
    assert temp.dtype == np.float64
    assert meta["n_samples"] == temp.size


def test_no_nan_inf(therm):
    _, temp, _ = therm.generate(duration_s=3.0, wear=0.9, seed=0)
    assert np.all(np.isfinite(temp))


def test_metadata_completeness(therm):
    _, _, meta = therm.generate(duration_s=2.0, wear=0.4, seed=0)
    required = {
        "modality", "simulator", "simulator_version", "timestamp_utc", "seed",
        "sensor_type", "sensor_focus", "fs_hz", "duration_s", "n_samples", "units",
        "alloy", "wear", "tooth_pass_freq_hz", "cutting_velocity_m_s",
        "avg_cutting_power_w", "heat_partition", "q_cut_w", "q_friction_w", "q_in_w",
        "ambient_c", "time_constant_s", "steady_state_temp_c", "mean_temp_c",
        "max_temp_c", "final_temp_c", "temp_rise_c",
    }
    assert required.issubset(meta.keys())


def test_label_ranges_and_thermal_anomaly(therm):
    _, _, meta = therm.generate(duration_s=2.0, wear=0.4, seed=0)
    assert meta["label_wear_level"] == pytest.approx(0.4)
    assert 0.0 <= meta["label_quality_score"] <= 1.0
    assert "label_thermal_anomaly_flag" in meta
    assert isinstance(meta["label_thermal_anomaly_flag"], bool)


def test_invalid_duration_raises(therm):
    with pytest.raises(ValueError):
        therm.generate(duration_s=-1.0)
