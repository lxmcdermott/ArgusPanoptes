"""Transparent downstream production-impact model for the Optimization Sandbox.

This module is a **generic, fully transparent demonstration** of how accurate
perception outputs (wear, cycle-time factor, quality score, health state) feed a
downstream production planning / costing / optimization layer. None of this is a
hidden black box: every formula is plain Python with an explanatory comment, and
the same structured payload the perception layer emits is what a real MES / cost
& nesting optimizer would consume.

The numbers here are illustrative planning proxies (cycle time, blade life,
yield, a composite efficiency score and a per-good-part cost). They are meant to
show *directionally correct* impact of perception accuracy on production
economics, not to be a validated cost model for any specific shop.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import numpy as np


@dataclass
class ProductionInputs:
    """Editable shop-floor assumptions for the impact model (all transparent)."""

    baseline_cycle_time_s: float = 45.0       # sharp-blade cycle time for one cut/part
    blade_life_cycles: float = 1000.0         # nominal blade life in cuts (see labels.rul_max_cycles)
    machine_rate_usd_per_hr: float = 90.0     # loaded machine + labor rate
    material_cost_usd_per_part: float = 12.0  # stock cost per part
    blade_cost_usd: float = 220.0             # cost of one replacement blade
    scrap_disposal_usd_per_part: float = 3.0  # cost to scrap a bad part
    shift_hours: float = 8.0                  # planning horizon for shift projections


@dataclass
class PerceptionState:
    """The perception outputs that drive the model (subset of Predictions)."""

    wear_level: float = 0.0
    cycle_time_factor: float = 1.0
    quality_score: float = 1.0
    health_state: str = "healthy"
    confidence: float = 1.0
    anomaly_flag: bool = False


def compute_production_impact(
    state: PerceptionState, inputs: ProductionInputs
) -> dict[str, Any]:
    """Compute production-planning metrics from perception outputs.

    Returns a dict with a ``reference`` (ideal sharp-blade baseline) block, a
    ``current`` (perception-informed) block, ``delta`` callouts, and a
    ``maintenance`` recommendation. All intermediate formulas are inline and
    commented so the demonstration is auditable.
    """
    inp = inputs

    # --- Effective cycle time -------------------------------------------------
    # The model predicts a cycle_time_factor >= 1.0; a dull blade cuts slower, so
    # the effective cycle time scales the sharp-blade baseline by that factor.
    ctf = max(1.0, float(state.cycle_time_factor))
    eff_cycle_time_s = inp.baseline_cycle_time_s * ctf
    ref_cycle_time_s = inp.baseline_cycle_time_s  # reference = sharp blade (factor 1.0)

    # --- Throughput (parts / hour) -------------------------------------------
    cur_throughput = 3600.0 / eff_cycle_time_s if eff_cycle_time_s > 0 else 0.0
    ref_throughput = 3600.0 / ref_cycle_time_s if ref_cycle_time_s > 0 else 0.0

    # --- Yield / scrap --------------------------------------------------------
    # quality_score in [0,1] is treated as first-pass yield; scrap is the shortfall.
    yield_frac = float(np.clip(state.quality_score, 0.0, 1.0))
    scrap_rate = 1.0 - yield_frac
    ref_yield = 1.0  # reference sharp blade -> nominal full yield

    # Good (sellable) parts per hour = throughput * yield.
    cur_good_pph = cur_throughput * yield_frac
    ref_good_pph = ref_throughput * ref_yield

    # --- Blade life / maintenance urgency ------------------------------------
    # Remaining life is the un-worn fraction of nominal blade life; urgency is
    # simply the predicted wear fraction (0 = fresh, 1 = end-of-life).
    wear = float(np.clip(state.wear_level, 0.0, 1.0))
    remaining_life_cycles = (1.0 - wear) * inp.blade_life_cycles
    maintenance_urgency = wear  # 0..1

    # --- Cost per good part ---------------------------------------------------
    # Sum of: machine/labor time cost + amortized blade cost + material (grossed
    # up for scrap) + scrap disposal, all divided per *good* part produced.
    def _cost_per_good_part(throughput: float, good_pph: float, yield_f: float) -> float:
        if good_pph <= 0:
            return float("inf")
        machine_cost = inp.machine_rate_usd_per_hr / good_pph
        blade_cost = inp.blade_cost_usd / max(inp.blade_life_cycles, 1.0)
        material_cost = inp.material_cost_usd_per_part / max(yield_f, 1e-6)
        scrap_cost = inp.scrap_disposal_usd_per_part * (1.0 - yield_f) / max(yield_f, 1e-6)
        return machine_cost + blade_cost + material_cost + scrap_cost

    cur_cost = _cost_per_good_part(cur_throughput, cur_good_pph, yield_frac)
    ref_cost = _cost_per_good_part(ref_throughput, ref_good_pph, ref_yield)

    # --- Composite Production Efficiency Score (0..100) -----------------------
    # OEE-style blend: performance (throughput vs reference) x quality (yield),
    # gently de-rated by low model confidence so uncertain reads plan cautiously.
    performance = cur_throughput / ref_throughput if ref_throughput > 0 else 0.0
    confidence_derate = 0.85 + 0.15 * float(np.clip(state.confidence, 0.0, 1.0))
    efficiency_score = 100.0 * performance * yield_frac * confidence_derate
    ref_efficiency = 100.0

    # --- Shift projections ----------------------------------------------------
    cur_shift_good = cur_good_pph * inp.shift_hours
    ref_shift_good = ref_good_pph * inp.shift_hours

    maintenance = _maintenance_recommendation(state, remaining_life_cycles)

    return {
        "reference": {
            "cycle_time_s": ref_cycle_time_s,
            "throughput_pph": ref_throughput,
            "yield_pct": ref_yield * 100.0,
            "good_pph": ref_good_pph,
            "cost_per_good_part": ref_cost,
            "efficiency_score": ref_efficiency,
            "shift_good_parts": ref_shift_good,
        },
        "current": {
            "cycle_time_s": eff_cycle_time_s,
            "throughput_pph": cur_throughput,
            "yield_pct": yield_frac * 100.0,
            "good_pph": cur_good_pph,
            "cost_per_good_part": cur_cost,
            "efficiency_score": efficiency_score,
            "shift_good_parts": cur_shift_good,
            "remaining_blade_life_cycles": remaining_life_cycles,
            "maintenance_urgency": maintenance_urgency,
        },
        "delta": {
            "cycle_time_s": eff_cycle_time_s - ref_cycle_time_s,
            "throughput_pph": cur_throughput - ref_throughput,
            "yield_pct": (yield_frac - ref_yield) * 100.0,
            "good_pph": cur_good_pph - ref_good_pph,
            "cost_per_good_part": cur_cost - ref_cost,
            "efficiency_score": efficiency_score - ref_efficiency,
            "shift_good_parts": cur_shift_good - ref_shift_good,
        },
        "maintenance": maintenance,
        "inputs": asdict(inp),
    }


def _maintenance_recommendation(
    state: PerceptionState, remaining_life_cycles: float
) -> dict[str, Any]:
    """Map perception state -> a maintenance urgency band + planning message."""
    wear = float(np.clip(state.wear_level, 0.0, 1.0))
    hs = (state.health_state or "").lower()
    if hs == "critical" or state.anomaly_flag or wear >= 0.85:
        level, urgency = "critical", "immediate"
        msg = "Stop and replace blade now; downstream jobs should be re-planned."
    elif hs == "warning" or wear >= 0.70:
        level, urgency = "warning", "this_shift"
        msg = "Schedule a blade change this shift; reduce feed to protect quality."
    elif hs == "monitor" or wear >= 0.45:
        level, urgency = "monitor", "planned"
        msg = "Keep running; fold a blade change into the next planned maintenance."
    else:
        level, urgency = "healthy", "none"
        msg = "Blade healthy; no maintenance action required."
    return {
        "level": level,
        "urgency": urgency,
        "message": msg,
        "remaining_blade_life_cycles": remaining_life_cycles,
    }


def downstream_payload(
    state: PerceptionState,
    inputs: ProductionInputs,
    impact: dict[str, Any],
    *,
    model: str = "unknown",
    source: str = "dashboard-optimization-sandbox",
) -> dict[str, Any]:
    """Build the exact structured payload a downstream planner would consume.

    Mirrors the perception layer's clean contract (predictions +
    recommendations) and augments it with the computed production-planning
    block, so it is obvious how a cost / cycle-time / nesting optimizer plugs in.
    """
    cur = impact["current"]
    maint = impact["maintenance"]
    return {
        "schema": "argus.production_plan.v1",
        "source": source,
        "model": model,
        "perception": {
            "wear_level": round(state.wear_level, 4),
            "cycle_time_factor": round(state.cycle_time_factor, 4),
            "quality_score": round(state.quality_score, 4),
            "health_state": state.health_state,
            "confidence": round(state.confidence, 4),
            "anomaly_flag": bool(state.anomaly_flag),
        },
        "production_plan": {
            "effective_cycle_time_s": round(cur["cycle_time_s"], 3),
            "throughput_parts_per_hr": round(cur["throughput_pph"], 2),
            "first_pass_yield_pct": round(cur["yield_pct"], 2),
            "good_parts_per_hr": round(cur["good_pph"], 2),
            "cost_per_good_part_usd": round(cur["cost_per_good_part"], 4),
            "production_efficiency_score": round(cur["efficiency_score"], 2),
            "remaining_blade_life_cycles": round(cur["remaining_blade_life_cycles"], 1),
        },
        "maintenance": {
            "level": maint["level"],
            "urgency": maint["urgency"],
            "message": maint["message"],
        },
        "assumptions": impact["inputs"],
    }
