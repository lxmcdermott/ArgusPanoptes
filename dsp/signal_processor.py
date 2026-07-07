"""Modular DSP feature extractor for saw / CNC condition monitoring.

:class:`SignalProcessor` turns a raw vibration (or thermal) waveform plus the
simulator's metadata dict into model-ready **scalar features** and, optionally,
a time-frequency spectrogram for the deep-learning path.

Pipeline
--------
1. **Preprocess** - detrend (remove DC / slow drift), zero-phase Butterworth
   band-pass (keep the tooth-pass harmonics + wear-driven broadband, drop
   out-of-band drift and above-Nyquist junk), and optional amplitude
   normalization (off by default - amplitude carries wear information).
2. **Time-domain features** - RMS, crest / shape / impulse / clearance /
   margin factors, excess kurtosis, skewness, peak-to-peak, zero-crossing rate,
   and Hilbert-envelope statistics. These track the *impulsiveness* of the
   signal, which grows as tooth impacts sharpen and broadband rises with wear.
3. **Frequency-domain features** - Welch PSD, total power, spectral centroid /
   rolloff / flatness / dominant peak, and - most importantly for blade wear -
   **TPF-relative band energies**: the tooth-pass fundamental band and its
   harmonics, plus a high-frequency broadband band. Tooth-pass band energy rises
   with wear because per-tooth cutting force (impact amplitude) increases; the
   broadband band rises with stochastic chip formation and edge friction.
4. **STFT** (optional) - power spectrogram for spectrogram-CNN inputs (Day 3).

Feature-name convention
------------------------
Time-domain feature keys are prefixed ``td_`` and frequency-domain keys ``fd_``
so the ML layer can trivially select feature groups for ablations.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy import signal as sp_signal
from scipy import stats as sp_stats

from dsp.config import ProcessorConfig, load_processor_config

__version__ = "0.1.0"
_logger = logging.getLogger(__name__)

__all__ = ["SignalProcessor", "TIME_DOMAIN_FEATURES", "FREQUENCY_DOMAIN_FEATURES", "__version__"]

FloatArray = npt.NDArray[np.float64]

_EPS = 1e-12

#: Ordered names of the time-domain scalar features (without the ``td_`` prefix).
TIME_DOMAIN_FEATURES: tuple[str, ...] = (
    "rms",
    "peak",
    "peak_to_peak",
    "crest_factor",
    "kurtosis",
    "skewness",
    "shape_factor",
    "impulse_factor",
    "clearance_factor",
    "margin_factor",
    "zero_crossing_rate",
    "envelope_mean",
    "envelope_std",
)

#: Ordered names of the frequency-domain scalar features (without the ``fd_`` prefix).
FREQUENCY_DOMAIN_FEATURES: tuple[str, ...] = (
    "total_power",
    "spectral_centroid_hz",
    "spectral_rolloff_hz",
    "spectral_flatness",
    "spectral_bandwidth_hz",
    "dominant_freq_hz",
    "dominant_amplitude",
    "tpf_band_energy",
    "tpf_harmonic2_energy",
    "tpf_harmonic3_energy",
    "tpf_harmonic_energy_total",
    "broadband_energy",
    "tpf_band_ratio",
    "broadband_ratio",
    "harmonic_to_fundamental_ratio",
)


class SignalProcessor:
    """Physics-informed DSP feature extractor.

    Parameters
    ----------
    config:
        A :class:`~dsp.config.ProcessorConfig`, a raw ``dict`` (validated via
        :meth:`ProcessorConfig.model_validate`), a ``str`` / path to an
        alternative ``processor_config.yaml``, or ``None`` to load the packaged
        default.
    """

    version: str = __version__
    _warned_dl_none: bool = False

    def __init__(self, config: ProcessorConfig | dict[str, Any] | str | None = None) -> None:
        if config is None:
            self.config = load_processor_config()
        elif isinstance(config, ProcessorConfig):
            self.config = config
        elif isinstance(config, dict):
            self.config = ProcessorConfig.model_validate(config)
        elif isinstance(config, str):
            self.config = load_processor_config(config)
        else:  # pragma: no cover - defensive
            raise TypeError(
                "config must be ProcessorConfig, dict, str path, or None; "
                f"got {type(config).__name__}"
            )
        self.pre = self.config.preprocess
        self.freq = self.config.frequency
        self.stft_cfg = self.config.stft
        self.dl = self.config.dl

    # ------------------------------------------------------------------ #
    # Preprocessing
    # ------------------------------------------------------------------ #
    def preprocess(self, waveform: npt.ArrayLike, fs: float) -> FloatArray:
        """Detrend, band-pass, and (optionally) normalize a raw waveform.

        The band-pass corners are clamped strictly below Nyquist so the same
        config works for any sample rate (e.g. the 40.96 kHz vibration path and
        the 200 Hz thermal path). ``filtfilt`` gives zero phase distortion so
        the timing of tooth impacts is preserved.

        Parameters
        ----------
        waveform:
            1-D array-like signal.
        fs:
            Sample rate in Hz (> 0).

        Returns
        -------
        np.ndarray
            The processed signal (float64, same length as the input).
        """
        x = np.asarray(waveform, dtype=np.float64).ravel()
        if fs <= 0:
            raise ValueError("fs must be positive.")
        if x.size == 0:
            return x

        # 1) Detrend (remove DC offset / slow linear drift).
        if self.pre.detrend != "none" and x.size >= 2:
            x = sp_signal.detrend(x, type=self.pre.detrend)

        # 2) Zero-phase Butterworth band-pass, corners clamped below Nyquist.
        bp = self.pre.bandpass
        if bp.enabled and x.size > 3 * (bp.order + 1):
            nyq = fs / 2.0
            low = max(bp.low_hz, 0.0)
            high = min(bp.high_hz, nyq * 0.999)
            if 0.0 < low < high < nyq:
                sos = sp_signal.butter(
                    bp.order, [low / nyq, high / nyq], btype="bandpass", output="sos"
                )
                x = sp_signal.sosfiltfilt(sos, x)
            elif high < nyq:  # low corner at/below 0 -> low-pass only
                sos = sp_signal.butter(bp.order, high / nyq, btype="lowpass", output="sos")
                x = sp_signal.sosfiltfilt(sos, x)

        # 3) Optional amplitude normalization (default "none"; see config docs).
        x = self._normalize(x)
        return np.ascontiguousarray(x, dtype=np.float64)

    def _normalize(self, x: FloatArray, method: str | None = None) -> FloatArray:
        method = self.pre.normalize if method is None else method
        if method == "none" or x.size == 0:
            return x
        if method == "zscore":
            std = float(np.std(x))
            return (x - float(np.mean(x))) / std if std > _EPS else x - float(np.mean(x))
        if method == "peak":
            peak = float(np.max(np.abs(x)))
            return x / peak if peak > _EPS else x
        if method == "rms":
            r = float(np.sqrt(np.mean(x**2)))
            return x / r if r > _EPS else x
        return x  # pragma: no cover - validated upstream

    # ------------------------------------------------------------------ #
    # Time-domain features
    # ------------------------------------------------------------------ #
    def extract_time_domain_features(self, x: npt.ArrayLike) -> dict[str, float]:
        """Compute impulsiveness / amplitude statistics of a (processed) signal.

        Returns keys prefixed ``td_``. The dimensionless *factors* (crest,
        shape, impulse, clearance, margin) are classic rotating-machinery
        health indicators: they grow as the signal becomes more impulsive
        (sharpening tooth-strike transients) even when overall amplitude is
        normalized away.
        """
        x = np.asarray(x, dtype=np.float64).ravel()
        n = x.size
        if n == 0:
            return {f"td_{k}": 0.0 for k in TIME_DOMAIN_FEATURES}

        abs_x = np.abs(x)
        peak = float(np.max(abs_x))
        rms = float(np.sqrt(np.mean(x**2)))
        mean_abs = float(np.mean(abs_x))
        # "square-root amplitude" for the clearance/margin factors.
        sra = float(np.mean(np.sqrt(abs_x)) ** 2)

        crest = peak / rms if rms > _EPS else 0.0
        shape = rms / mean_abs if mean_abs > _EPS else 0.0
        impulse = peak / mean_abs if mean_abs > _EPS else 0.0
        clearance = peak / sra if sra > _EPS else 0.0
        margin = clearance / rms if rms > _EPS else 0.0  # peak / (sra * rms)

        kurt = float(sp_stats.kurtosis(x, fisher=True, bias=False)) if n > 3 else 0.0
        skew = float(sp_stats.skew(x, bias=False)) if n > 2 else 0.0

        # Zero-crossing rate: proxy for dominant-frequency content / roughness.
        signs = np.signbit(x)
        zcr = float(np.count_nonzero(signs[1:] != signs[:-1])) / max(1, n - 1)

        # Hilbert-envelope statistics: amplitude-modulation depth of tooth strikes.
        if n >= 4:
            env = np.abs(sp_signal.hilbert(x))
            env_mean = float(np.mean(env))
            env_std = float(np.std(env))
        else:
            env_mean = mean_abs
            env_std = 0.0

        return {
            "td_rms": rms,
            "td_peak": peak,
            "td_peak_to_peak": float(np.max(x) - np.min(x)),
            "td_crest_factor": crest,
            "td_kurtosis": kurt,
            "td_skewness": skew,
            "td_shape_factor": shape,
            "td_impulse_factor": impulse,
            "td_clearance_factor": clearance,
            "td_margin_factor": margin,
            "td_zero_crossing_rate": zcr,
            "td_envelope_mean": env_mean,
            "td_envelope_std": env_std,
        }

    # ------------------------------------------------------------------ #
    # Frequency-domain features
    # ------------------------------------------------------------------ #
    def extract_frequency_domain_features(
        self, x: npt.ArrayLike, fs: float, tpf_hz: float | None = None
    ) -> dict[str, float]:
        """Compute Welch-PSD spectral-shape and TPF-relative band features.

        The tooth-pass-relative bands are the highest-value blade-wear features:
        the tooth-pass fundamental band ``[tpf*(1-frac), tpf*(1+frac)]`` and its
        harmonics carry the periodic impact energy (which scales with cutting
        force, hence wear), while the broadband high-frequency band captures the
        stochastic chip-formation / friction energy that also rises with wear.

        When ``tpf_hz`` is ``None`` (or non-positive) the TPF-relative features
        are returned as ``NaN`` while all TPF-independent spectral features are
        still computed - a graceful degradation for signals of unknown kinematics.
        """
        x = np.asarray(x, dtype=np.float64).ravel()
        if fs <= 0:
            raise ValueError("fs must be positive.")

        nan_tpf = {
            "fd_tpf_band_energy": np.nan,
            "fd_tpf_harmonic2_energy": np.nan,
            "fd_tpf_harmonic3_energy": np.nan,
            "fd_tpf_harmonic_energy_total": np.nan,
            "fd_tpf_band_ratio": np.nan,
            "fd_harmonic_to_fundamental_ratio": np.nan,
        }
        if x.size < 8:
            base = {f"fd_{k}": 0.0 for k in FREQUENCY_DOMAIN_FEATURES}
            base.update(nan_tpf if tpf_hz is None else {})
            return base

        nperseg = int(min(self.freq.welch_nperseg, x.size))
        noverlap = int(nperseg * self.freq.welch_overlap)
        freqs, psd = sp_signal.welch(
            x, fs=fs, window=self.freq.window, nperseg=nperseg, noverlap=noverlap
        )
        psd = np.asarray(psd, dtype=np.float64)
        total_power = float(np.trapezoid(psd, freqs))
        total_power = max(total_power, _EPS)

        # --- Spectral shape (TPF-independent) ---
        psd_sum = float(np.sum(psd)) + _EPS
        centroid = float(np.sum(freqs * psd) / psd_sum)
        variance = float(np.sum(((freqs - centroid) ** 2) * psd) / psd_sum)
        bandwidth = float(np.sqrt(max(variance, 0.0)))

        cumulative = np.cumsum(psd)
        rolloff_target = self.freq.rolloff_fraction * cumulative[-1]
        rolloff_idx = int(np.searchsorted(cumulative, rolloff_target))
        rolloff_idx = min(rolloff_idx, freqs.size - 1)
        rolloff = float(freqs[rolloff_idx])

        # Geometric-mean / arithmetic-mean flatness (0 = tonal, 1 = white).
        log_psd = np.log(psd + _EPS)
        flatness = float(np.exp(np.mean(log_psd)) / (np.mean(psd) + _EPS))

        # Dominant peak (ignore the DC bin).
        if psd.size > 1:
            dom_idx = int(np.argmax(psd[1:])) + 1
        else:  # pragma: no cover - guarded by size check above
            dom_idx = 0
        dominant_freq = float(freqs[dom_idx])
        dominant_amp = float(psd[dom_idx])

        features: dict[str, float] = {
            "fd_total_power": total_power,
            "fd_spectral_centroid_hz": centroid,
            "fd_spectral_rolloff_hz": rolloff,
            "fd_spectral_flatness": flatness,
            "fd_spectral_bandwidth_hz": bandwidth,
            "fd_dominant_freq_hz": dominant_freq,
            "fd_dominant_amplitude": dominant_amp,
        }

        # --- Broadband high-frequency band (TPF-independent) ---
        bb_lo, bb_hi = self.freq.broadband_band_hz
        broadband = self._band_energy(freqs, psd, bb_lo, min(bb_hi, fs / 2.0))
        features["fd_broadband_energy"] = broadband
        features["fd_broadband_ratio"] = broadband / total_power

        # --- TPF-relative bands (the key wear features) ---
        if tpf_hz is not None and tpf_hz > 0:
            frac = self.freq.tpf_band_frac
            nyq = fs / 2.0
            n_harm = self.freq.n_harmonics
            fund = self._band_energy(freqs, psd, tpf_hz * (1 - frac), tpf_hz * (1 + frac))
            # Fixed feature schema supports 2x and 3x; higher orders are omitted in v1.
            h2 = (
                self._band_energy(
                    freqs, psd, 2 * tpf_hz * (1 - frac), min(2 * tpf_hz * (1 + frac), nyq)
                )
                if n_harm >= 2
                else 0.0
            )
            h3 = (
                self._band_energy(
                    freqs, psd, 3 * tpf_hz * (1 - frac), min(3 * tpf_hz * (1 + frac), nyq)
                )
                if n_harm >= 3
                else 0.0
            )
            harmonic_total = fund + h2 + h3
            features["fd_tpf_band_energy"] = fund
            features["fd_tpf_harmonic2_energy"] = h2
            features["fd_tpf_harmonic3_energy"] = h3
            features["fd_tpf_harmonic_energy_total"] = harmonic_total
            features["fd_tpf_band_ratio"] = fund / total_power
            features["fd_harmonic_to_fundamental_ratio"] = (
                (h2 + h3) / fund if fund > _EPS else 0.0
            )
        else:
            features.update(nan_tpf)

        return features

    @staticmethod
    def _band_energy(freqs: FloatArray, psd: FloatArray, lo: float, hi: float) -> float:
        """Integrate the PSD over ``[lo, hi]`` (trapezoidal), 0 if empty band."""
        if hi <= lo:
            return 0.0
        mask = (freqs >= lo) & (freqs <= hi)
        if np.count_nonzero(mask) < 2:
            return 0.0
        return float(np.trapezoid(psd[mask], freqs[mask]))

    # ------------------------------------------------------------------ #
    # Time-frequency (STFT)
    # ------------------------------------------------------------------ #
    def compute_stft(self, x: npt.ArrayLike, fs: float) -> dict[str, np.ndarray]:
        """Compute a power spectrogram via STFT.

        Returns ``{'freqs', 'times', 'power'}`` where ``power`` is
        ``(n_freq, n_time)``. If ``stft.log_scale`` is set the power is returned
        in dB (``10*log10(power + eps)``). Not used on the scalar-feature path;
        provided for the Day-3 spectrogram-CNN models.
        """
        x = np.asarray(x, dtype=np.float64).ravel()
        if fs <= 0:
            raise ValueError("fs must be positive.")
        nperseg = int(min(self.stft_cfg.nperseg, max(2, x.size)))
        noverlap = int(nperseg * self.stft_cfg.overlap)
        freqs, times, zxx = sp_signal.stft(
            x, fs=fs, window=self.stft_cfg.window, nperseg=nperseg, noverlap=noverlap
        )
        power = np.abs(zxx) ** 2
        if self.stft_cfg.log_scale:
            power = 10.0 * np.log10(power + _EPS)
        return {
            "freqs": freqs.astype(np.float32),
            "times": times.astype(np.float32),
            "power": power.astype(np.float32),
        }

    # ------------------------------------------------------------------ #
    # Deep-learning input preparation (Day-3 CNN / fusion paths)
    # ------------------------------------------------------------------ #
    def get_normalized_waveform(self, x: npt.ArrayLike, fs: float) -> np.ndarray:
        """Return a model-ready 1-D ``float32`` waveform for the 1D-CNN path.

        Runs the standard :meth:`preprocess` (detrend -> zero-phase band-pass ->
        the configured ``preprocess.normalize``) and then applies the DL
        normalization ``dl.normalize_for_dl`` (default ``"zscore"``) on top, so
        every chunk is amplitude scale-invariant. With the default
        ``preprocess.normalize="none"`` this is exactly *detrend + band-pass +
        z-score*.

        Rationale: the tabular path keeps absolute amplitude (a first-class wear
        feature), but per-chunk scale invariance lets the 1D-CNN focus on the
        *shape* of the tooth-strike transients and generalize across operating
        points / gains. Returns ``float32`` for compact tensor storage.

        Parameters
        ----------
        x:
            Raw 1-D signal.
        fs:
            Sample rate in Hz (> 0).

        Returns
        -------
        np.ndarray
            1-D ``float32`` array, same length as the (raveled) input.
        """
        xp = self.preprocess(x, fs)
        if self.dl.normalize_for_dl == "none" and not SignalProcessor._warned_dl_none:
            _logger.warning(
                "dl.normalize_for_dl='none': CNN inputs keep absolute amplitude (wear cue) "
                "but are scale-sensitive across gains/operating points. Prefer 'zscore' when "
                "generalization matters; use 'none' only when amplitude is an intentional feature."
            )
            SignalProcessor._warned_dl_none = True
        xn = self._normalize(xp, method=self.dl.normalize_for_dl)
        return np.ascontiguousarray(xn, dtype=np.float32)

    def compute_spectrogram(self, x: npt.ArrayLike, fs: float) -> np.ndarray:
        """Return a log-power spectrogram for the spectrogram-CNN path.

        Convenience wrapper that (1) normalizes the waveform exactly like
        :meth:`get_normalized_waveform` (detrend -> band-pass ->
        ``dl.normalize_for_dl``) for a consistent, scale-invariant CNN front-end
        and (2) computes the STFT power via :meth:`compute_stft` (dB when
        ``stft.log_scale`` is set).

        Shape convention
        ----------------
        The returned array is **2-D ``(n_freq, n_time)``** ``float32`` -
        frequency bins along axis 0 (``nperseg // 2 + 1`` rows), time frames
        along axis 1. This matches ``compute_stft(...)["power"]`` and is the
        natural ``(H, W)`` layout for a single-channel 2-D conv input (add a
        channel dim to get ``(1, n_freq, n_time)``).

        Parameters
        ----------
        x:
            Raw 1-D signal.
        fs:
            Sample rate in Hz (> 0).

        Returns
        -------
        np.ndarray
            ``(n_freq, n_time)`` ``float32`` log-power (or linear-power)
            spectrogram.
        """
        xn = self.get_normalized_waveform(x, fs)
        return self.compute_stft(xn, fs)["power"]

    # ------------------------------------------------------------------ #
    # Full pipeline
    # ------------------------------------------------------------------ #
    def process(
        self,
        waveform: npt.ArrayLike,
        fs: float,
        metadata: dict[str, Any] | None = None,
        *,
        compute_spectrogram: bool = False,
    ) -> dict[str, Any]:
        """Run the full feature-extraction pipeline for one waveform.

        Parameters
        ----------
        waveform:
            Raw 1-D signal.
        fs:
            Sample rate (Hz). If ``None`` in metadata, must be passed here.
        metadata:
            Optional simulator metadata dict. ``tooth_pass_freq_hz`` (or
            ``fs_hz``) are used when present.
        compute_spectrogram:
            If ``True`` also return the STFT power spectrogram.

        Returns
        -------
        dict
            ``{'features': dict[str, float], 'spectrogram': dict | None,
            'processed_metadata': dict}``. Features are float32-friendly scalars.
        """
        metadata = metadata or {}
        tpf_hz = metadata.get("tooth_pass_freq_hz")
        if fs is None:
            fs = metadata.get("fs_hz")
        if fs is None:
            raise ValueError("fs must be provided (argument or metadata['fs_hz']).")

        x = self.preprocess(waveform, fs)
        features: dict[str, float] = {}
        features.update(self.extract_time_domain_features(x))
        features.update(self.extract_frequency_domain_features(x, fs, tpf_hz=tpf_hz))
        # Cast to plain floats (float32-representable) for compact Parquet storage.
        features = {k: float(np.float32(v)) for k, v in features.items()}

        spectrogram = self.compute_stft(x, fs) if compute_spectrogram else None

        processed_metadata = {
            "processor_version": self.version,
            "fs_hz": float(fs),
            "n_samples": int(x.size),
            "tpf_hz": float(tpf_hz) if tpf_hz is not None else None,
            "tpf_available": bool(tpf_hz is not None and tpf_hz > 0),
            "n_features": len(features),
        }
        return {
            "features": features,
            "spectrogram": spectrogram,
            "processed_metadata": processed_metadata,
        }

    def process_batch(
        self,
        waveforms: list[npt.ArrayLike] | np.ndarray,
        fs: float,
        metadatas: list[dict[str, Any]] | None = None,
        *,
        prefix: str = "",
        include_metadata: bool = False,
    ) -> "Any":
        """Extract features for many waveforms into a :class:`pandas.DataFrame`.

        Parameters
        ----------
        waveforms:
            Iterable of 1-D signals.
        fs:
            Common sample rate (Hz).
        metadatas:
            Optional per-waveform metadata dicts (for ``tooth_pass_freq_hz``).
        prefix:
            Optional prefix applied to every feature column (e.g. ``"vib_"``).
        include_metadata:
            If ``True`` and ``metadatas`` is given, join each metadata dict's
            scalar (non-array) values as additional columns.

        Returns
        -------
        pandas.DataFrame
            One row per waveform; columns are the (prefixed) feature names.
        """
        import pandas as pd

        metadatas = metadatas or [None] * len(waveforms)  # type: ignore[list-item]
        rows: list[dict[str, Any]] = []
        for wf, md in zip(waveforms, metadatas):
            result = self.process(wf, fs, metadata=md)
            row = {f"{prefix}{k}": v for k, v in result["features"].items()}
            if include_metadata and md:
                for k, v in md.items():
                    if np.isscalar(v) or isinstance(v, (str, bool, int, float)):
                        row.setdefault(k, v)
            rows.append(row)
        return pd.DataFrame(rows)
