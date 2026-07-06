"""Physics-informed vibration simulator for saw / CNC blade-wear monitoring.

:class:`SawVibrationSimulator` synthesizes single-axis IEPE-accelerometer
time-series (in *g*) that reproduce the salient features real condition-monitoring
systems exploit:

* **Tooth-pass forcing** - a train of shaped tooth-engagement impacts at the
  tooth-pass frequency (TPF) and its harmonics; TPF is derived exactly from
  saw kinematics (see :func:`sensors.utils.calculate_tpf`).
* **Force modulation** - impact amplitude scales with the physics-based
  per-tooth cutting force (specific energy x chip load), which rises with wear.
* **Wear-modulated broadband** - colored (pink) Gaussian noise whose power grows
  with wear, capturing stochastic chip formation and increased edge friction.
* **Machine dynamics** - configurable structural-resonance boosts.
* **Sensor noise** - white Gaussian at the accelerometer noise floor.

The output units are standard gravity (g).
"""

from __future__ import annotations

import datetime as _dt
from typing import Any, Iterator

import numpy as np
from scipy import signal as sp_signal

from sensors.config import SensorConfig, load_config
from sensors.utils import (
    FloatArray,
    ForceModel,
    compute_force_model,
    crest_factor,
    generate_colored_noise,
    generate_labels,
    kurtosis,
    rms,
)

__version__ = "0.1.0"

__all__ = ["SawVibrationSimulator", "__version__"]


class SawVibrationSimulator:
    """Generate physics-informed vibration signals for a sawing operation.

    Parameters
    ----------
    config:
        A :class:`~sensors.config.SensorConfig`, a raw ``dict`` (validated via
        :meth:`SensorConfig.model_validate`), or ``None`` to load the packaged
        default ``sensor_specs.yaml``.
    """

    version: str = __version__

    def __init__(self, config: SensorConfig | dict[str, Any] | None = None) -> None:
        if config is None:
            self.config = load_config()
        elif isinstance(config, SensorConfig):
            self.config = config
        elif isinstance(config, dict):
            self.config = SensorConfig.model_validate(config)
        else:  # pragma: no cover - defensive
            raise TypeError(
                "config must be SensorConfig, dict, or None; "
                f"got {type(config).__name__}"
            )
        self.vib = self.config.vibration

    # ------------------------------------------------------------------ public
    def generate(
        self,
        duration_s: float = 5.0,
        params: dict[str, Any] | None = None,
        wear: float = 0.0,
        seed: int | None = None,
    ) -> tuple[FloatArray, FloatArray, dict[str, Any]]:
        """Synthesize one vibration recording.

        Parameters
        ----------
        duration_s:
            Recording length in seconds (> 0).
        params:
            Optional operating-point overrides. Recognized keys: ``alloy``,
            ``blade_speed_sfpm``, ``num_teeth``, ``feed_per_tooth_mm``,
            ``depth_mm``, ``kerf_width_mm``. Unspecified keys fall back to the
            machining defaults in the config.
        wear:
            Blade wear in ``[0, 1]`` (0 = sharp, 1 = end-of-life).
        seed:
            Seed for reproducibility. Falls back to the config's ``random_seed``.

        Returns
        -------
        (t, accel_g, metadata):
            Time vector (s), acceleration signal (g), and a rich metadata dict
            containing every input parameter, derived physics, signal statistics,
            and deterministic labels.
        """
        if duration_s <= 0:
            raise ValueError("duration_s must be positive.")
        params = dict(params or {})
        wear = float(np.clip(wear, 0.0, 1.0))
        seed = self.config.random_seed if seed is None else seed
        rng = np.random.default_rng(seed)

        fs = self.vib.fs_hz
        n = int(round(duration_s * fs))
        if n < 2:
            raise ValueError("duration_s too short for the configured sample rate.")
        t = np.arange(n, dtype=np.float64) / fs

        force = compute_force_model(
            self.config,
            alloy=params.get("alloy"),
            blade_speed_sfpm=params.get("blade_speed_sfpm"),
            num_teeth=params.get("num_teeth"),
            feed_per_tooth_mm=params.get("feed_per_tooth_mm"),
            depth_mm=params.get("depth_mm"),
            kerf_width_mm=params.get("kerf_width_mm"),
            wear=wear,
        )

        # Peak acceleration budget from per-tooth force (kN -> g).
        peak_g = self.vib.accel_g_per_kN * (force.per_tooth_force_n / 1000.0)

        # Slowly-varying force envelope (chip-thickness / engagement variation).
        force_env = self._force_envelope(t, rng)

        periodic = self._tooth_impact_train(t, force.tpf_hz, peak_g, force_env, rng)
        periodic += self._harmonic_comb(t, force.tpf_hz, peak_g, force_env)

        broadband = self._broadband_component(n, wear, rng)

        sensor = rng.normal(0.0, self.vib.noise_floor_g_rms, size=n)

        accel = (periodic + broadband + sensor).astype(np.float64)

        # Clip check against accelerometer full-scale range.
        clipped = bool(np.any(np.abs(accel) > self.vib.clip_g))
        if clipped:
            accel = np.clip(accel, -self.vib.clip_g, self.vib.clip_g)

        metadata = self._build_metadata(
            duration_s=duration_s,
            n=n,
            wear=wear,
            seed=seed,
            params=params,
            force=force,
            accel=accel,
            peak_g=peak_g,
            clipped=clipped,
        )
        return t, accel, metadata

    def stream(
        self,
        duration_s: float = 5.0,
        chunk_s: float = 0.1,
        params: dict[str, Any] | None = None,
        wear: float = 0.0,
        seed: int | None = None,
    ) -> Iterator[tuple[FloatArray, FloatArray]]:
        """Yield consecutive ``(t, accel)`` chunks (stub for ``StreamingPerceptor``).

        v1 generates the full recording then partitions it, which is sufficient
        for developing the downstream streaming interface. A future version will
        synthesize chunks with maintained phase/state continuity.
        """
        t, accel, _ = self.generate(duration_s, params, wear, seed)
        step = max(1, int(round(chunk_s * self.vib.fs_hz)))
        for start in range(0, len(accel), step):
            yield t[start : start + step], accel[start : start + step]

    # --------------------------------------------------------------- internals
    def _force_envelope(self, t: FloatArray, rng: np.random.Generator) -> FloatArray:
        """A gentle (~5%) low-frequency modulation of the cutting force."""
        f_slow = rng.uniform(3.0, 9.0)  # Hz
        phase = rng.uniform(0.0, 2.0 * np.pi)
        return 1.0 + 0.05 * np.sin(2.0 * np.pi * f_slow * t + phase)

    def _tooth_impact_train(
        self,
        t: FloatArray,
        tpf_hz: float,
        peak_g: float,
        force_env: FloatArray,
        rng: np.random.Generator,
    ) -> FloatArray:
        """Place one shaped impact per tooth engagement at rate ``tpf_hz``."""
        n = t.size
        fs = self.vib.fs_hz
        out = np.zeros(n, dtype=np.float64)
        if tpf_hz <= 0 or peak_g <= 0:
            return out

        period = 1.0 / tpf_hz
        period_samples = period * fs
        pulse = self._pulse_kernel(period_samples)
        if pulse.size == 0:
            return out

        n_strikes = int(np.floor((n - 1) / period_samples)) + 1
        # Per-strike amplitude: base peak * local force envelope * small jitter.
        for i in range(n_strikes):
            idx = int(round(i * period_samples))
            if idx >= n:
                break
            amp = peak_g * float(force_env[idx]) * (1.0 + 0.03 * rng.standard_normal())
            end = min(idx + pulse.size, n)
            out[idx:end] += amp * pulse[: end - idx]
        return out

    def _pulse_kernel(self, period_samples: float) -> FloatArray:
        """Build a unit-peak tooth-engagement pulse kernel."""
        p = self.vib.pulse
        fs = self.vib.fs_hz
        length = max(1, int(round(p.width_frac * period_samples)))
        tau = np.arange(length, dtype=np.float64) / fs

        if p.shape == "half_sine":
            kernel = np.sin(np.pi * np.arange(length) / max(1, length - 1))
        else:  # "exp_decay_sine": damped ring, a good impact model
            kernel = np.exp(-p.damping * tau) * np.sin(2.0 * np.pi * p.ring_freq_hz * tau)

        peak = np.max(np.abs(kernel))
        if peak > 0:
            kernel = kernel / peak
        return kernel.astype(np.float64)

    def _harmonic_comb(
        self,
        t: FloatArray,
        tpf_hz: float,
        peak_g: float,
        force_env: FloatArray,
    ) -> FloatArray:
        """Secondary tonal comb at TPF harmonics (keeps spectral lines crisp)."""
        harmonics = self.vib.harmonics
        if tpf_hz <= 0 or peak_g <= 0 or not harmonics:
            return np.zeros_like(t)
        nyquist = self.vib.fs_hz / 2.0
        comb = np.zeros_like(t)
        for k, h in enumerate(harmonics, start=1):
            freq = k * tpf_hz
            if h <= 0 or freq >= nyquist:
                continue
            comb += h * np.sin(2.0 * np.pi * freq * t)
        # Keep the comb secondary to the physical impact train.
        return 0.25 * peak_g * force_env * comb

    def _broadband_component(
        self, n: int, wear: float, rng: np.random.Generator
    ) -> FloatArray:
        """Wear-modulated colored noise, shaped by structural resonances."""
        rms_g = self.vib.broadband_base_g * (1.0 + self.vib.broadband_wear_gain * wear)
        noise = generate_colored_noise(
            n,
            self.vib.fs_hz,
            exponent=self.vib.broadband_exponent,
            rms_g=rms_g,
            rng=rng,
        )
        return self._apply_structural_modes(noise)

    def _apply_structural_modes(self, x: FloatArray) -> FloatArray:
        """Boost energy near configured structural modes via narrow band-pass mix."""
        if not self.vib.structural_modes or x.size == 0:
            return x
        fs = self.vib.fs_hz
        nyquist = fs / 2.0
        out = x.copy()
        for mode in self.vib.structural_modes:
            if mode.center_hz >= nyquist:
                continue
            bw = mode.center_hz / mode.q
            low = max(mode.center_hz - bw / 2.0, 1.0)
            high = min(mode.center_hz + bw / 2.0, nyquist * 0.999)
            if low >= high:
                continue
            sos = sp_signal.butter(
                2, [low / nyquist, high / nyquist], btype="bandpass", output="sos"
            )
            band = sp_signal.sosfilt(sos, x)
            out = out + (mode.gain - 1.0) * band
        return out.astype(np.float64)

    def _build_metadata(
        self,
        *,
        duration_s: float,
        n: int,
        wear: float,
        seed: int,
        params: dict[str, Any],
        force: ForceModel,
        accel: FloatArray,
        peak_g: float,
        clipped: bool,
    ) -> dict[str, Any]:
        sig_rms = rms(accel)
        sig_kurt = kurtosis(accel)
        labels = generate_labels(
            self.config, wear=wear, force=force, signal_kurtosis=sig_kurt
        )
        return {
            "modality": "vibration",
            "simulator": "SawVibrationSimulator",
            "simulator_version": self.version,
            "config_version": self.config.simulator_version,
            "timestamp_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
            "seed": int(seed),
            # --- Sensor spec summary ---
            "sensor_type": self.vib.type,
            "sensor_mounting": self.vib.mounting,
            "fs_hz": self.vib.fs_hz,
            "duration_s": duration_s,
            "n_samples": int(n),
            "units": "g",
            # --- Input operating point (resolved) ---
            "saw_type": self.config.machining.saw_type,
            "alloy": force.extras["alloy"],
            "blade_speed_sfpm": force.extras["blade_speed_sfpm"],
            "num_teeth": int(force.extras["num_teeth"]),
            "feed_per_tooth_mm": force.extras["feed_per_tooth_mm"],
            "depth_mm": force.extras["depth_mm"],
            "kerf_width_mm": force.extras["kerf_width_mm"],
            "wear": wear,
            "params_override": params,
            # --- Derived physics ---
            "tooth_pass_freq_hz": force.tpf_hz,
            "rpm": force.rpm,
            "cutting_velocity_m_s": force.cutting_velocity_m_s,
            "feed_velocity_mm_s": force.feed_velocity_mm_s,
            "material_removal_rate_mm3_s": force.material_removal_rate_mm3_s,
            "specific_energy_j_per_mm3": force.specific_energy_j_per_mm3,
            "avg_cutting_power_w": force.cutting_power_w,
            "avg_cutting_force_n": force.avg_cutting_force_n,
            "per_tooth_force_n": force.per_tooth_force_n,
            "force_wear_multiplier": force.force_wear_multiplier,
            "peak_accel_budget_g": peak_g,
            # --- Signal statistics ---
            "signal_rms_g": sig_rms,
            "signal_kurtosis": sig_kurt,
            "signal_crest_factor": crest_factor(accel),
            "signal_peak_g": float(np.max(np.abs(accel))),
            "clipped": clipped,
            # --- Labels ---
            **{f"label_{k}": v for k, v in labels.items()},
        }
