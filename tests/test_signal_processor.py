"""Tests for :class:`dsp.signal_processor.SignalProcessor`.

These exercise the preprocessing, time/frequency feature extraction, STFT, the
full ``process`` pipeline, and ``process_batch``, with specific *physical*
assertions (band-energy monotonicity with wear, finiteness / range checks,
reproducibility, and graceful degradation when the tooth-pass frequency is
unknown).
"""

from __future__ import annotations

import numpy as np
import pytest

from dsp import (
    FREQUENCY_DOMAIN_FEATURES,
    TIME_DOMAIN_FEATURES,
    SignalProcessor,
    load_processor_config,
)


# --------------------------------------------------------------------------- #
# Preprocessing
# --------------------------------------------------------------------------- #
def test_preprocess_preserves_length_and_energy(proc, vib):
    """Detrend + band-pass keeps length and (roughly) the in-band energy."""
    _, accel, meta = vib.generate(duration_s=2.0, wear=0.4, seed=0)
    x = proc.preprocess(accel, fs=meta["fs_hz"])
    assert x.shape == accel.shape
    assert np.all(np.isfinite(x))
    ratio = np.sqrt(np.mean(x**2)) / np.sqrt(np.mean(accel**2))
    # Most energy is in-band; band-pass should retain the bulk of it.
    assert 0.7 < ratio < 1.2


def test_preprocess_removes_dc_offset(proc):
    fs = 40960.0
    t = np.arange(int(fs)) / fs
    x = 5.0 + np.sin(2 * np.pi * 1000.0 * t)  # large DC offset
    y = proc.preprocess(x, fs=fs)
    assert abs(float(np.mean(y))) < 1e-3


def test_preprocess_normalize_zscore():
    sp = SignalProcessor(load_processor_config(overrides={"preprocess": {"normalize": "zscore"}}))
    fs = 40960.0
    t = np.arange(int(fs)) / fs
    x = 3.0 * np.sin(2 * np.pi * 1500.0 * t)
    y = sp.preprocess(x, fs=fs)
    assert float(np.std(y)) == pytest.approx(1.0, abs=0.05)


# --------------------------------------------------------------------------- #
# Time-domain features
# --------------------------------------------------------------------------- #
def test_time_domain_features_finite_and_present(proc, vib):
    _, accel, meta = vib.generate(duration_s=2.0, wear=0.5, seed=1)
    x = proc.preprocess(accel, fs=meta["fs_hz"])
    feats = proc.extract_time_domain_features(x)
    assert set(feats) == {f"td_{k}" for k in TIME_DOMAIN_FEATURES}
    assert all(np.isfinite(v) for v in feats.values())
    assert feats["td_rms"] > 0.0
    assert feats["td_crest_factor"] >= 1.0  # peak >= rms by definition
    assert feats["td_peak_to_peak"] > 0.0


def test_time_domain_rms_matches_numpy(proc):
    rng = np.random.default_rng(0)
    x = rng.standard_normal(4096)
    feats = proc.extract_time_domain_features(x)
    assert feats["td_rms"] == pytest.approx(float(np.sqrt(np.mean(x**2))), rel=1e-6)


# --------------------------------------------------------------------------- #
# Frequency-domain features
# --------------------------------------------------------------------------- #
def test_frequency_features_finite_and_ranges(proc, vib):
    _, accel, meta = vib.generate(duration_s=3.0, wear=0.5, seed=2)
    fs = meta["fs_hz"]
    x = proc.preprocess(accel, fs=fs)
    feats = proc.extract_frequency_domain_features(x, fs=fs, tpf_hz=meta["tooth_pass_freq_hz"])
    assert set(feats) == {f"fd_{k}" for k in FREQUENCY_DOMAIN_FEATURES}
    assert all(np.isfinite(v) for v in feats.values())
    assert feats["fd_total_power"] > 0.0
    assert 0.0 <= feats["fd_spectral_flatness"] <= 1.0
    assert 0.0 <= feats["fd_spectral_centroid_hz"] <= fs / 2.0
    assert 0.0 <= feats["fd_spectral_rolloff_hz"] <= fs / 2.0
    assert 0.0 <= feats["fd_tpf_band_ratio"] <= 1.0


def test_dominant_freq_in_band_and_tpf_energy_present(proc, vib):
    """Dominant PSD line sits in-band (near the impact ring / structural modes)
    and the tooth-pass fundamental band carries measurable energy."""
    _, accel, meta = vib.generate(duration_s=4.0, wear=0.2, seed=3)
    fs = meta["fs_hz"]
    tpf = meta["tooth_pass_freq_hz"]
    x = proc.preprocess(accel, fs=fs)
    feats = proc.extract_frequency_domain_features(x, fs=fs, tpf_hz=tpf)
    # Dominant energy is driven by the ~2 kHz tooth-impact ring / structural modes.
    assert 100.0 <= feats["fd_dominant_freq_hz"] <= 18000.0
    # The tooth-pass fundamental band still carries real, positive energy.
    assert feats["fd_tpf_band_energy"] > 0.0
    assert 0.0 < feats["fd_tpf_band_ratio"] <= 1.0


def test_tpf_band_energy_monotonic_in_wear(proc, vib):
    """TPF fundamental-band energy must rise with blade wear (impact force ↑)."""
    fs = None
    energies = []
    for w in (0.0, 0.5, 1.0):
        _, accel, meta = vib.generate(duration_s=3.0, wear=w, seed=11)
        fs = meta["fs_hz"]
        x = proc.preprocess(accel, fs=fs)
        f = proc.extract_frequency_domain_features(x, fs=fs, tpf_hz=meta["tooth_pass_freq_hz"])
        energies.append(f["fd_tpf_band_energy"])
    assert energies[0] < energies[1] < energies[2]


def test_broadband_energy_monotonic_in_wear(proc, vib):
    """Broadband high-frequency energy must rise with wear (chip/friction ↑)."""
    vals = []
    for w in (0.0, 1.0):
        _, accel, meta = vib.generate(duration_s=3.0, wear=w, seed=13)
        x = proc.preprocess(accel, fs=meta["fs_hz"])
        f = proc.extract_frequency_domain_features(
            x, fs=meta["fs_hz"], tpf_hz=meta["tooth_pass_freq_hz"]
        )
        vals.append(f["fd_broadband_energy"])
    assert vals[1] > vals[0]


def test_frequency_features_graceful_without_tpf(proc, vib):
    """Missing TPF -> TPF-relative features are NaN, others still finite."""
    _, accel, meta = vib.generate(duration_s=2.0, wear=0.5, seed=4)
    fs = meta["fs_hz"]
    x = proc.preprocess(accel, fs=fs)
    feats = proc.extract_frequency_domain_features(x, fs=fs, tpf_hz=None)
    assert np.isnan(feats["fd_tpf_band_energy"])
    assert np.isnan(feats["fd_tpf_harmonic_energy_total"])
    # TPF-independent features remain finite and sensible.
    assert np.isfinite(feats["fd_total_power"]) and feats["fd_total_power"] > 0
    assert np.isfinite(feats["fd_spectral_centroid_hz"])
    assert np.isfinite(feats["fd_broadband_energy"])


# --------------------------------------------------------------------------- #
# STFT
# --------------------------------------------------------------------------- #
def test_stft_shapes(proc, vib):
    _, accel, meta = vib.generate(duration_s=1.0, wear=0.3, seed=5)
    out = proc.compute_stft(accel, fs=meta["fs_hz"])
    assert set(out) == {"freqs", "times", "power"}
    n_freq = out["freqs"].size
    n_time = out["times"].size
    assert out["power"].shape == (n_freq, n_time)
    assert n_freq == proc.stft_cfg.nperseg // 2 + 1
    assert np.all(np.isfinite(out["power"]))


# --------------------------------------------------------------------------- #
# DL input-preparation convenience methods (Day 3)
# --------------------------------------------------------------------------- #
def test_get_normalized_waveform_zscore(proc, vib):
    """Default DL normalization is z-score: float32, zero-mean, unit-std."""
    _, accel, meta = vib.generate(duration_s=1.0, wear=0.4, seed=7)
    w = proc.get_normalized_waveform(accel, fs=meta["fs_hz"])
    assert w.dtype == np.float32
    assert w.shape == accel.shape
    assert abs(float(np.mean(w))) < 1e-3
    assert float(np.std(w)) == pytest.approx(1.0, abs=0.05)


def test_compute_spectrogram_shape(proc, vib):
    """Spectrogram is 2-D (n_freq, n_time) float32 with n_freq = nperseg//2 + 1."""
    _, accel, meta = vib.generate(duration_s=1.0, wear=0.4, seed=7)
    spec = proc.compute_spectrogram(accel, fs=meta["fs_hz"])
    assert spec.dtype == np.float32
    assert spec.ndim == 2
    assert spec.shape[0] == proc.stft_cfg.nperseg // 2 + 1
    assert np.all(np.isfinite(spec))


def test_dl_config_normalize_default(proc):
    assert proc.dl.normalize_for_dl == "zscore"


# --------------------------------------------------------------------------- #
# Full pipeline / process / process_batch
# --------------------------------------------------------------------------- #
def test_process_returns_expected_structure(proc, vib):
    _, accel, meta = vib.generate(duration_s=2.0, wear=0.6, seed=6)
    result = proc.process(accel, fs=meta["fs_hz"], metadata=meta)
    assert set(result) == {"features", "spectrogram", "processed_metadata"}
    assert result["spectrogram"] is None
    n_expected = len(TIME_DOMAIN_FEATURES) + len(FREQUENCY_DOMAIN_FEATURES)
    assert len(result["features"]) == n_expected
    assert result["processed_metadata"]["tpf_available"] is True
    assert all(np.isfinite(v) for v in result["features"].values())


def test_process_uses_fs_from_metadata(proc, vib):
    _, accel, meta = vib.generate(duration_s=1.0, wear=0.3, seed=6)
    result = proc.process(accel, fs=None, metadata=meta)
    assert result["processed_metadata"]["fs_hz"] == pytest.approx(meta["fs_hz"])


def test_process_spectrogram_optional(proc, vib):
    _, accel, meta = vib.generate(duration_s=1.0, wear=0.3, seed=6)
    result = proc.process(accel, fs=meta["fs_hz"], metadata=meta, compute_spectrogram=True)
    assert result["spectrogram"] is not None
    assert "power" in result["spectrogram"]


def test_process_reproducible(proc, vib):
    """Deterministic features for a fixed simulator seed."""
    _, a1, m1 = vib.generate(duration_s=2.0, wear=0.5, seed=99)
    _, a2, m2 = vib.generate(duration_s=2.0, wear=0.5, seed=99)
    f1 = proc.process(a1, fs=m1["fs_hz"], metadata=m1)["features"]
    f2 = proc.process(a2, fs=m2["fs_hz"], metadata=m2)["features"]
    assert f1 == f2


def test_process_batch_shape_and_columns(proc, vib):
    waveforms, metas = [], []
    for w in (0.1, 0.5, 0.9):
        _, accel, meta = vib.generate(duration_s=1.5, wear=w, seed=20)
        waveforms.append(accel)
        metas.append(meta)
    df = proc.process_batch(waveforms, fs=metas[0]["fs_hz"], metadatas=metas, prefix="vib_")
    assert df.shape[0] == 3
    n_expected = len(TIME_DOMAIN_FEATURES) + len(FREQUENCY_DOMAIN_FEATURES)
    assert df.shape[1] == n_expected
    assert all(c.startswith("vib_") for c in df.columns)
    # Monotone TPF band energy across the wear sweep.
    assert df["vib_fd_tpf_band_energy"].is_monotonic_increasing


def test_process_batch_include_metadata(proc, vib):
    waveforms, metas = [], []
    for w in (0.2, 0.8):
        _, accel, meta = vib.generate(duration_s=1.0, wear=w, seed=21)
        waveforms.append(accel)
        metas.append(meta)
    df = proc.process_batch(
        waveforms, fs=metas[0]["fs_hz"], metadatas=metas, prefix="vib_", include_metadata=True
    )
    assert "wear" in df.columns
    assert "tooth_pass_freq_hz" in df.columns


# --------------------------------------------------------------------------- #
# Config / construction
# --------------------------------------------------------------------------- #
def test_construct_from_dict_and_overrides():
    cfg = load_processor_config(overrides={"frequency": {"n_harmonics": 2}})
    sp = SignalProcessor(cfg)
    assert sp.freq.n_harmonics == 2
    sp2 = SignalProcessor(cfg.model_dump())
    assert sp2.freq.n_harmonics == 2


def test_n_harmonics_limits_harmonic_bands(proc, vib):
    """``frequency.n_harmonics`` must gate 2x/3x band integration (fixed schema)."""
    sp2 = SignalProcessor(load_processor_config(overrides={"frequency": {"n_harmonics": 2}}))
    _, accel, meta = vib.generate(duration_s=3.0, wear=0.5, seed=31)
    fs = meta["fs_hz"]
    tpf = meta["tooth_pass_freq_hz"]
    x = sp2.preprocess(accel, fs=fs)
    f3 = sp2.extract_frequency_domain_features(x, fs=fs, tpf_hz=tpf)
    f2 = proc.extract_frequency_domain_features(
        proc.preprocess(accel, fs=fs), fs=fs, tpf_hz=tpf
    )
    assert f3["fd_tpf_harmonic3_energy"] == 0.0
    assert f3["fd_tpf_harmonic2_energy"] > 0.0
    assert f3["fd_tpf_harmonic_energy_total"] < f2["fd_tpf_harmonic_energy_total"]


def test_n_harmonics_one_graceful(vib):
    """Low ``n_harmonics=1`` must zero out 2x/3x bands without NaNs/errors.

    Regression guard for the harmonic-energy fix (commit 5dfd041): ``h2``/``h3``
    are only band-integrated when ``n_harmonics >= 2 / >= 3`` and default to 0.0
    otherwise. The fundamental band must still carry positive energy and the
    harmonic total must collapse to the fundamental alone.
    """
    sp1 = SignalProcessor(load_processor_config(overrides={"frequency": {"n_harmonics": 1}}))
    _, accel, meta = vib.generate(duration_s=3.0, wear=0.5, seed=31)
    fs = meta["fs_hz"]
    tpf = meta["tooth_pass_freq_hz"]
    x = sp1.preprocess(accel, fs=fs)
    f = sp1.extract_frequency_domain_features(x, fs=fs, tpf_hz=tpf)
    assert all(np.isfinite(v) for v in f.values())
    assert f["fd_tpf_harmonic2_energy"] == 0.0
    assert f["fd_tpf_harmonic3_energy"] == 0.0
    assert f["fd_tpf_band_energy"] > 0.0
    assert f["fd_tpf_harmonic_energy_total"] == pytest.approx(f["fd_tpf_band_energy"])
    # With no harmonics above the fundamental, the harmonic-to-fundamental ratio is 0.
    assert f["fd_harmonic_to_fundamental_ratio"] == 0.0


def test_invalid_fs_raises(proc):
    with pytest.raises(ValueError):
        proc.preprocess(np.ones(100), fs=0.0)
    with pytest.raises(ValueError):
        proc.extract_frequency_domain_features(np.ones(100), fs=-1.0)


#: TPF-relative frequency features (NaN when tooth-pass frequency is unknown).
_TPF_RELATIVE = {
    "fd_tpf_band_energy",
    "fd_tpf_harmonic2_energy",
    "fd_tpf_harmonic3_energy",
    "fd_tpf_harmonic_energy_total",
    "fd_tpf_band_ratio",
    "fd_harmonic_to_fundamental_ratio",
}


def test_thermal_path_low_fs(proc, therm):
    """Processor must also handle the 200 Hz thermal signal without error.

    No TPF is supplied, so only the TPF-relative features are NaN; every other
    feature must be finite despite the much lower sample rate and near-DC signal.
    """
    _, temp, meta = therm.generate(duration_s=3.0, wear=0.5, seed=0)
    result = proc.process(temp, fs=meta["fs_hz"], metadata={"fs_hz": meta["fs_hz"]})
    for k, v in result["features"].items():
        if k in _TPF_RELATIVE:
            assert np.isnan(v)
        else:
            assert np.isfinite(v), f"{k} not finite: {v}"
