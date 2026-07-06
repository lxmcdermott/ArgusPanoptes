"""Physics-informed thermal simulator for cut-zone / tool-chip monitoring.

:class:`ThermalSimulator` models the cut-zone temperature seen by a fast IR
pyrometer aimed at the tool-chip interface, using a lumped first-order thermal
system:

    C * dT/dt = Q_in - k * (T - T_amb)

where

* ``C``     - lumped heat capacity of the cut zone (J/degC),
* ``k``     - effective loss coefficient (conduction + convection + radiation, W/degC),
* ``Q_in``  - heat into the zone = ``heat_partition * cutting_power + friction_power(wear)``,
* ``T_amb`` - shop-floor ambient.

Wear directly raises the friction-heat term, making cut-zone temperature the
primary observable for cut-condition monitoring. The steady state is
``T_ss = T_amb + Q_in / k`` and the time constant is ``tau = C / k``.
"""

from __future__ import annotations

import datetime as _dt
from typing import Any

import numpy as np

from sensors.config import SensorConfig, load_config
from sensors.utils import FloatArray, ForceModel, compute_force_model, generate_labels

__version__ = "0.1.0"

__all__ = ["ThermalSimulator", "__version__"]


class ThermalSimulator:
    """Generate physics-informed cut-zone temperature traces.

    Parameters
    ----------
    config:
        A :class:`~sensors.config.SensorConfig`, a raw ``dict``, or ``None`` to
        load the packaged default ``sensor_specs.yaml``.
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
        self.thermal = self.config.thermal
        self.model = self.thermal.thermal_model

    # ------------------------------------------------------------------ public
    def generate(
        self,
        duration_s: float = 5.0,
        params: dict[str, Any] | None = None,
        wear: float = 0.0,
        seed: int | None = None,
    ) -> tuple[FloatArray, FloatArray, dict[str, Any]]:
        """Synthesize one cut-zone temperature recording.

        Parameters
        ----------
        duration_s:
            Recording length in seconds (> 0).
        params:
            Optional operating-point overrides (same keys as the vibration
            simulator: ``alloy``, ``blade_speed_sfpm``, ``num_teeth``,
            ``feed_per_tooth_mm``, ``depth_mm``, ``kerf_width_mm``).
        wear:
            Blade wear in ``[0, 1]``.
        seed:
            Seed for reproducibility (falls back to the config's ``random_seed``).

        Returns
        -------
        (t, temp_c, metadata):
            Time vector (s), temperature trace (deg C), and metadata dict with
            inputs, derived physics, statistics, and labels.
        """
        if duration_s <= 0:
            raise ValueError("duration_s must be positive.")
        params = dict(params or {})
        wear = float(np.clip(wear, 0.0, 1.0))
        seed = self.config.random_seed if seed is None else seed
        rng = np.random.default_rng(seed)

        fs = self.thermal.fs_hz
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

        q_cut = self.model.heat_partition * force.cutting_power_w
        q_friction = self.model.friction_power_gain_w * wear
        q_in = q_cut + q_friction

        clean = self._integrate(t, q_in, fs)

        # Sensor noise (white Gaussian, 1-sigma = noise_std_c).
        temp = clean + rng.normal(0.0, self.thermal.noise_std_c, size=n)
        temp = np.clip(temp, -50.0, self.model.max_temp_c).astype(np.float64)

        metadata = self._build_metadata(
            duration_s=duration_s,
            n=n,
            wear=wear,
            seed=seed,
            params=params,
            force=force,
            q_cut=q_cut,
            q_friction=q_friction,
            q_in=q_in,
            clean=clean,
            temp=temp,
        )
        return t, temp, metadata

    # --------------------------------------------------------------- internals
    def _integrate(self, t: FloatArray, q_in: float, fs: float) -> FloatArray:
        """Forward-Euler integration of the lumped first-order thermal ODE.

        Starting from ambient, the zone heats toward
        ``T_ss = T_amb + Q_in / k`` with time constant ``tau = C / k``.
        """
        n = t.size
        dt = 1.0 / fs
        k = self.model.loss_coefficient_w_per_c
        c = self.model.thermal_mass_j_per_c
        t_amb = self.model.ambient_c

        temp = np.empty(n, dtype=np.float64)
        temp[0] = t_amb
        # Vectorized closed-form of the first-order response is exact & stable:
        #   T(t) = T_ss + (T0 - T_ss) * exp(-t / tau)
        tau = c / k
        t_ss = t_amb + q_in / k
        temp = t_ss + (t_amb - t_ss) * np.exp(-t / tau)
        return temp

    def _build_metadata(
        self,
        *,
        duration_s: float,
        n: int,
        wear: float,
        seed: int,
        params: dict[str, Any],
        force: ForceModel,
        q_cut: float,
        q_friction: float,
        q_in: float,
        clean: FloatArray,
        temp: FloatArray,
    ) -> dict[str, Any]:
        k = self.model.loss_coefficient_w_per_c
        t_ss = self.model.ambient_c + q_in / k
        mean_temp = float(np.mean(temp))
        max_temp = float(np.max(temp))
        labels = generate_labels(
            self.config, wear=wear, force=force, cut_zone_temp_c=mean_temp
        )
        return {
            "modality": "thermal",
            "simulator": "ThermalSimulator",
            "simulator_version": self.version,
            "config_version": self.config.simulator_version,
            "timestamp_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
            "seed": int(seed),
            # --- Sensor spec summary ---
            "sensor_type": self.thermal.type,
            "sensor_focus": self.thermal.focus,
            "fs_hz": self.thermal.fs_hz,
            "duration_s": duration_s,
            "n_samples": int(n),
            "units": "deg_C",
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
            "cutting_velocity_m_s": force.cutting_velocity_m_s,
            "material_removal_rate_mm3_s": force.material_removal_rate_mm3_s,
            "specific_energy_j_per_mm3": force.specific_energy_j_per_mm3,
            "avg_cutting_power_w": force.cutting_power_w,
            "heat_partition": self.model.heat_partition,
            "q_cut_w": q_cut,
            "q_friction_w": q_friction,
            "q_in_w": q_in,
            "ambient_c": self.model.ambient_c,
            "time_constant_s": self.model.thermal_mass_j_per_c / k,
            "steady_state_temp_c": t_ss,
            # --- Signal statistics ---
            "mean_temp_c": mean_temp,
            "max_temp_c": max_temp,
            "final_temp_c": float(temp[-1]),
            "temp_rise_c": max_temp - self.model.ambient_c,
            # --- Labels ---
            **{f"label_{k}": v for k, v in labels.items()},
        }
