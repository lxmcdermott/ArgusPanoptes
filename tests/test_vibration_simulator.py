"""Tests for :class:`sensors.vibration_simulator.SawVibrationSimulator`."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import signal as sp_signal

from sensors import SawVibrationSimulator
from sensors.utils import calculate_tpf, rpm_from_sfpm


# --------------------------------------------------------------------------- #
# TPF / kinematics
# --------------------------------------------------------------------------- #
def test_tpf_calculation_exact(config):
    """TPF must match the closed-form circular-saw formula exactly."""
    m = config.machining
    rpm = rpm_from_sfpm(m.blade_speed_sfpm, m.blade_diameter_mm)
    expected = rpm * m.num_teeth / 60.0
    got = calculate_tpf(
        saw_type="circular",
        blade_speed_sfpm=m.blade_speed_sfpm,
        num_teeth=m.num_teeth,
        blade_diameter_mm=m.blade_diameter_mm,
    )
    assert got == pytest.approx(expected, rel=1e-12)
    # Sanity: hundreds of Hz for a real aluminum saw.
    assert 100.0 < got < 2000.0


def test_band_saw_tpf():
    got = calculate_tpf(saw_type="band", blade_speed_sfpm=800.0, tpi=4)
    assert got == pytest.approx(800.0 * 12.0 * 4 / 60.0)


def test_metadata_tpf_matches_util(vib, config):
    _, _, meta = vib.generate(duration_s=1.0, wear=0.0, seed=0)
    expected = calculate_tpf(
        saw_type="circular",
        blade_speed_sfpm=config.machining.blade_speed_sfpm,
        num_teeth=config.machining.num_teeth,
        blade_diameter_mm=config.machining.blade_diameter_mm,
    )
    assert meta["tooth_pass_freq_hz"] == pytest.approx(expected, rel=1e-9)


def test_tpf_detected_in_spectrum(vib, config):
    """The dominant line near the analytic TPF should be within 1%."""
    _, accel, meta = vib.generate(duration_s=4.0, wear=0.2, seed=1)
    fs = config.vibration.fs_hz
    freqs, psd = sp_signal.welch(accel, fs=fs, nperseg=8192)
    tpf = meta["tooth_pass_freq_hz"]
    mask = (freqs >= tpf * 0.5) & (freqs <= tpf * 1.5)
    peak_hz = freqs[mask][int(np.argmax(psd[mask]))]
    assert abs(peak_hz - tpf) / tpf < 0.01


# --------------------------------------------------------------------------- #
# Wear monotonicity
# --------------------------------------------------------------------------- #
def test_rms_monotonic_in_wear(vib):
    rms_vals = [
        vib.generate(duration_s=3.0, wear=w, seed=42)[2]["signal_rms_g"]
        for w in (0.0, 0.25, 0.5, 0.75, 1.0)
    ]
    assert all(b > a for a, b in zip(rms_vals, rms_vals[1:]))


def test_spectral_energy_monotonic_in_wear(vib, config):
    """Total spectral energy should grow with wear."""
    fs = config.vibration.fs_hz
    energies = []
    for w in (0.0, 0.5, 1.0):
        _, accel, _ = vib.generate(duration_s=3.0, wear=w, seed=7)
        _, psd = sp_signal.welch(accel, fs=fs, nperseg=8192)
        energies.append(float(np.sum(psd)))
    assert energies[0] < energies[1] < energies[2]


def test_broadband_energy_increases_with_wear(vib, config):
    """High-frequency broadband band energy should rise with wear."""
    fs = config.vibration.fs_hz
    band = (6000.0, 15000.0)  # above the main tooth-pass harmonics
    vals = []
    for w in (0.0, 1.0):
        _, accel, _ = vib.generate(duration_s=3.0, wear=w, seed=5)
        freqs, psd = sp_signal.welch(accel, fs=fs, nperseg=8192)
        mask = (freqs >= band[0]) & (freqs <= band[1])
        vals.append(float(np.trapezoid(psd[mask], freqs[mask])))
    assert vals[1] > vals[0]


# --------------------------------------------------------------------------- #
# Reproducibility / structure / validation
# --------------------------------------------------------------------------- #
def test_reproducibility_same_seed(vib):
    _, a1, _ = vib.generate(duration_s=1.0, wear=0.3, seed=123)
    _, a2, _ = vib.generate(duration_s=1.0, wear=0.3, seed=123)
    assert np.array_equal(a1, a2)


def test_different_seed_differs(vib):
    _, a1, _ = vib.generate(duration_s=1.0, wear=0.3, seed=1)
    _, a2, _ = vib.generate(duration_s=1.0, wear=0.3, seed=2)
    assert not np.array_equal(a1, a2)


def test_signal_length_and_dtype(vib, config):
    fs = config.vibration.fs_hz
    t, accel, meta = vib.generate(duration_s=2.0, wear=0.1, seed=0)
    assert accel.shape == t.shape == (int(round(2.0 * fs)),)
    assert accel.dtype == np.float64
    assert meta["n_samples"] == accel.size


def test_no_nan_inf(vib):
    _, accel, _ = vib.generate(duration_s=2.0, wear=0.9, seed=0)
    assert np.all(np.isfinite(accel))


def test_no_clipping_at_nominal(vib, config):
    _, accel, meta = vib.generate(duration_s=2.0, wear=0.5, seed=0)
    assert not meta["clipped"]
    assert np.max(np.abs(accel)) < config.vibration.clip_g


def test_metadata_completeness(vib):
    _, _, meta = vib.generate(duration_s=1.0, wear=0.4, seed=0)
    required = {
        "modality", "simulator", "simulator_version", "timestamp_utc", "seed",
        "fs_hz", "duration_s", "n_samples", "units", "alloy", "blade_speed_sfpm",
        "num_teeth", "feed_per_tooth_mm", "depth_mm", "kerf_width_mm", "wear",
        "tooth_pass_freq_hz", "rpm", "cutting_velocity_m_s",
        "material_removal_rate_mm3_s", "specific_energy_j_per_mm3",
        "avg_cutting_power_w", "avg_cutting_force_n", "per_tooth_force_n",
        "signal_rms_g", "signal_kurtosis", "signal_crest_factor", "signal_peak_g",
        "clipped",
    }
    assert required.issubset(meta.keys())


def test_label_ranges(vib):
    _, _, meta = vib.generate(duration_s=1.0, wear=0.4, seed=0)
    assert meta["label_wear_level"] == pytest.approx(0.4)
    assert 0.0 <= meta["label_rul_fraction"] <= 1.0
    assert 0 <= meta["label_rul_cycles"] <= 1000
    assert meta["label_cycle_time_factor"] >= 1.0
    assert 0.0 <= meta["label_quality_score"] <= 1.0
    assert meta["label_health_state"] in {"healthy", "monitor", "warning", "critical"}
    assert isinstance(meta["label_anomaly_flag"], bool)


def test_high_wear_flags_anomaly(vib):
    _, _, meta = vib.generate(duration_s=1.0, wear=0.95, seed=0)
    assert meta["label_anomaly_flag"] is True
    assert meta["label_health_state"] == "critical"


def test_params_override(vib):
    _, _, meta = vib.generate(
        duration_s=1.0, params={"alloy": "7075", "num_teeth": 100}, wear=0.0, seed=0
    )
    assert meta["alloy"] == "7075"
    assert meta["num_teeth"] == 100


def test_invalid_duration_raises(vib):
    with pytest.raises(ValueError):
        vib.generate(duration_s=0.0)


def test_stream_chunks_cover_signal(vib, config):
    chunks = list(vib.stream(duration_s=1.0, chunk_s=0.1, wear=0.2, seed=0))
    total = sum(len(a) for _, a in chunks)
    assert total == int(round(1.0 * config.vibration.fs_hz))


def test_construct_from_dict(config):
    sim = SawVibrationSimulator(config.model_dump())
    _, accel, _ = sim.generate(duration_s=0.5, wear=0.1, seed=0)
    assert accel.size > 0
