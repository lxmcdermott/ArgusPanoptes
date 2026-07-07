"""Pre-built demo scenarios for repeatable, high-quality dashboard recordings.

Each :class:`Scenario` bundles a realistic operating point with a deterministic
per-chunk wear "plan", so a single button press produces a compelling, repeatable
live run (steady healthy operation, a progressive wear-to-failure ramp, or a
sudden anomaly injection). Pure Python / NumPy - no Streamlit dependency.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class Scenario:
    """A named demo scenario driving the Live Monitor's simulated run.

    Attributes
    ----------
    key, name, icon, description, narrative:
        UI-facing identity and explanatory copy.
    params:
        Operating-point overrides passed to the simulator (alloy, kinematics).
    n_chunks, delay_s, seed:
        Run length, per-chunk pacing, and base seed for reproducibility.
    wear_plan:
        ``(step, n_chunks) -> wear`` function returning the injected wear for
        each chunk of the run.
    """

    key: str
    name: str
    icon: str
    description: str
    narrative: str
    params: dict[str, float | str]
    n_chunks: int
    delay_s: float
    seed: int
    wear_plan: Callable[[int, int], float] = field(repr=False)
    noise_sd: float = 0.0  # optional sensor noise multiplier (×rms) for robustness demos

    def wear_for_step(self, step: int) -> float:
        """Injected wear for ``step`` (clamped to ``[0, 1]``)."""
        return float(np.clip(self.wear_plan(step, self.n_chunks), 0.0, 1.0))


def _normal_plan(step: int, n: int) -> float:
    # Gentle noise around a healthy operating point.
    rng = np.random.default_rng(1000 + step)
    return 0.14 + 0.03 * float(rng.standard_normal())


def _progressive_plan(step: int, n: int) -> float:
    # Linear ramp from nearly-sharp to end-of-life across the run, plus jitter.
    frac = step / max(1, n - 1)
    rng = np.random.default_rng(2000 + step)
    return 0.08 + 0.9 * frac + 0.02 * float(rng.standard_normal())


def _anomaly_plan(step: int, n: int) -> float:
    # Steady healthy operation, then a sudden step change (e.g. chipped tooth).
    trigger = max(2, int(round(n * 0.55)))
    rng = np.random.default_rng(3000 + step)
    base = 0.18 if step < trigger else 0.92
    return base + 0.02 * float(rng.standard_normal())


def _noisy_robust_plan(step: int, n: int) -> float:
    # Moderate wear with a mid-run sensor-noise event (robustness ablation demo).
    rng = np.random.default_rng(4000 + step)
    base = 0.42 + 0.04 * float(rng.standard_normal())
    return float(np.clip(base, 0.0, 1.0))


SCENARIOS: dict[str, Scenario] = {
    "normal": Scenario(
        key="normal",
        name="Normal Operation",
        icon="\U0001f7e2",  # green circle
        description="Sharp blade, nominal cut. Steady healthy readings.",
        narrative="Baseline healthy run: stable wear, high quality, no alerts.",
        params={
            "alloy": "6061",
            "blade_speed_sfpm": 800.0,
            "feed_per_tooth_mm": 0.12,
            "depth_mm": 25.0,
            "num_teeth": 80,
        },
        n_chunks=24,
        delay_s=0.10,
        seed=7,
        wear_plan=_normal_plan,
    ),
    "progressive": Scenario(
        key="progressive",
        name="Progressive Wear to Failure",
        icon="\U0001f4c8",  # chart increasing
        description="Blade wears from sharp to end-of-life; watch KPIs climb.",
        narrative="Wear ramps 0.1\u21921.0: health degrades healthy\u2192warning\u2192critical.",
        params={
            "alloy": "7075",
            "blade_speed_sfpm": 900.0,
            "feed_per_tooth_mm": 0.18,
            "depth_mm": 32.0,
            "num_teeth": 72,
        },
        n_chunks=32,
        delay_s=0.11,
        seed=11,
        wear_plan=_progressive_plan,
    ),
    "anomaly": Scenario(
        key="anomaly",
        name="Sudden Anomaly",
        icon="\u26a1",  # lightning
        description="Healthy cut interrupted by an abrupt fault (e.g. chipped tooth).",
        narrative="Steady healthy run, then a sudden jump to critical mid-cut.",
        params={
            "alloy": "6061",
            "blade_speed_sfpm": 750.0,
            "feed_per_tooth_mm": 0.14,
            "depth_mm": 28.0,
            "num_teeth": 84,
        },
        n_chunks=26,
        delay_s=0.11,
        seed=17,
        wear_plan=_anomaly_plan,
    ),
    "noisy_sensor": Scenario(
        key="noisy_sensor",
        name="Noisy Sensor Robustness",
        icon="\U0001f4f6",  # antenna
        description="Moderate wear under elevated sensor noise — compare normnone vs noisy models.",
        narrative=(
            "Fixed moderate wear (~0.42) with a mid-run noise burst. Use Simulation Lab "
            "presets to compare 1dcnn_normnone vs 1dcnn_noisy."
        ),
        params={
            "alloy": "6061",
            "blade_speed_sfpm": 850.0,
            "feed_per_tooth_mm": 0.15,
            "depth_mm": 30.0,
            "num_teeth": 80,
        },
        n_chunks=20,
        delay_s=0.10,
        seed=23,
        wear_plan=_noisy_robust_plan,
        noise_sd=0.35,
    ),
}


def scenario_options() -> list[tuple[str, str]]:
    """Return ``(key, "icon name")`` tuples for populating a selector."""
    return [(s.key, f"{s.icon}  {s.name}") for s in SCENARIOS.values()]
