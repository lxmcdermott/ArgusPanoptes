"""Argus Panoptes - High-performance NiceGUI operator dashboard (Day 5, v2).

A production-grade, truly reactive monitoring UI for the multi-modal industrial
perception stack. It replaces the legacy Streamlit dashboard (now under
``_legacy/``) with a **NiceGUI + Plotly** front-end whose WebSocket-backed
reactive updates, ``ui.timer`` refresh, and a decoupled background
:class:`~dashviz.orchestrator.SimulationOrchestrator` deliver smooth,
flicker-free multi-stream live visualization (waveform / FFT / STFT / thermal +
KPI gauges) at a stable 5-10 Hz.

Operation modes
---------------
* **Standalone (direct)** - an in-process :class:`StreamingPerceptor` for the
  lowest-latency demo (default; recommended for recordings).
* **Connected to API** - the same contract over HTTP against the FastAPI service
  (``uvicorn app.main:app``).

Run
---
    pip install -e ".[ml,dl,app,dashboard-nicegui]"
    python -m app.nicegui_dashboard          # or: python app/nicegui_dashboard.py

Then open http://127.0.0.1:8080 .
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Ensure the repo root is importable whether run as a module or a script.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from nicegui import run, ui  # noqa: E402

from dashviz import metrics, plots  # noqa: E402
from dashviz import optimization as opt  # noqa: E402
from dashviz import orchestrator as orch  # noqa: E402
from dashviz.scenarios import SCENARIOS  # noqa: E402
from dashviz.theme import (  # noqa: E402
    COLORS,
    alert_banner_html,
    kpi_card_html,
    recommendation_card_html,
    status_pill_html,
)

#: Plotly config for live charts: no mode bar (nothing to repaint per tick).
_PLOTLY_LIVE = {"displayModeBar": False, "responsive": True, "displaylogo": False,
                "scrollZoom": False, "doubleClick": "reset", "showTips": False}
#: Plotly config for analysis charts (Lab / History / System).
_PLOTLY_STATIC = {"displayModeBar": True, "responsive": True, "displaylogo": False,
                  "modeBarButtonsToRemove": ["select2d", "lasso2d", "autoScale2d"]}

#: Simulation-Lab presets (operating points + expected behavior notes).
LAB_PRESETS: dict[str, dict[str, Any]] = {
    "clean_sharp": {
        "label": "Clean sharp blade",
        "params": {"alloy": "6061", "blade_speed_sfpm": 800.0, "feed_per_tooth_mm": 0.12,
                   "depth_mm": 25.0, "num_teeth": 80},
        "wear": 0.12, "noise": 0.0, "seed": 42,
        "note": "Nominal cut; expect healthy/monitor states and low wear MAE on clean data.",
    },
    "high_wear": {
        "label": "High wear (0.85)",
        "params": {"alloy": "7075", "blade_speed_sfpm": 900.0, "feed_per_tooth_mm": 0.18,
                   "depth_mm": 32.0, "num_teeth": 72},
        "wear": 0.85, "noise": 0.0, "seed": 7,
        "note": "End-of-life blade; expect warning/critical health and a blade-change flag.",
    },
    "noisy_robustness": {
        "label": "Noisy input (sd=0.5x rms)",
        "params": {"alloy": "6061", "blade_speed_sfpm": 800.0, "feed_per_tooth_mm": 0.12,
                   "depth_mm": 25.0, "num_teeth": 80},
        "wear": 0.55, "noise": 0.5, "seed": 99,
        "note": "Robustness ablation: compare 1dcnn vs 1dcnn_noisy - the noisy variant "
                "should hold ~3x better wear MAE under corruption.",
    },
}

_TREND_LABELS = {
    "wear_level": "Wear level", "cycle_time_factor": "Cycle-time factor",
    "quality_score": "Quality score", "confidence": "Confidence",
    "mean_temp_c": "Cut-zone temp (C)",
}
_HEALTH_ORDER = ["healthy", "monitor", "warning", "critical"]


# =========================================================================== #
# Theme (Quasar/NiceGUI dark) + Argus component CSS
# =========================================================================== #
def _install_theme() -> None:
    """Apply the dark industrial palette and inject the Argus component CSS."""
    c = COLORS
    ui.colors(
        primary=c.accent, secondary=c.info, accent=c.warning,
        dark=c.bg, dark_page=c.bg, positive=c.accent, negative=c.critical,
        warning=c.warning, info=c.info,
    )
    ui.add_head_html(f"""
<style>
  body, .nicegui-content {{ background: {c.bg}; color: {c.text};
     font-family: Inter, 'Segoe UI', system-ui, sans-serif; }}
  .q-page, .q-tab-panels, .q-tab-panel {{ background: transparent !important; }}
  .argus-appbg {{
     background:
        radial-gradient(1200px 600px at 15% -10%, #16233a 0%, rgba(15,23,42,0) 55%),
        radial-gradient(1000px 500px at 100% 0%, #12202f 0%, rgba(15,23,42,0) 50%),
        {c.bg};
  }}
  .argus-panel {{
     background: linear-gradient(160deg, {c.surface} 0%, {c.bg_alt} 100%);
     border: 1px solid {c.border}; border-radius: 14px;
     box-shadow: 0 2px 6px rgba(0,0,0,0.3); }}
  .argus-title {{ color: {c.text}; font-weight:800; letter-spacing:-0.01em; }}
  .argus-caption {{ color: {c.text_muted}; font-size: 0.78rem; }}

  /* KPI cards */
  .argus-kpi {{ background: linear-gradient(160deg, {c.surface} 0%, {c.bg_alt} 100%);
     border: 1px solid {c.border}; border-left: 4px solid {c.accent};
     border-radius: 12px; padding: 12px 14px 10px 14px;
     box-shadow: 0 2px 6px rgba(0,0,0,0.3); }}
  .argus-kpi .kpi-label {{ color: {c.text_muted}; font-size: 0.72rem;
     text-transform: uppercase; letter-spacing: 0.06em; font-weight: 600;
     display:flex; align-items:center; gap:6px; }}
  .argus-kpi .kpi-value {{ color: {c.text}; font-size: 1.7rem; font-weight: 700;
     line-height: 1.15; font-variant-numeric: tabular-nums; }}
  .argus-kpi .kpi-sub {{ color: {c.text_muted}; font-size: 0.74rem; }}

  /* Alert banner */
  .argus-alert {{ border-radius: 12px; padding: 12px 16px; margin: 2px 0;
     display:flex; align-items:center; gap:12px; font-size:0.92rem; font-weight:600;
     border:1px solid transparent; }}
  .argus-alert .a-icon {{ font-size: 1.25rem; }}
  .argus-alert .a-title {{ font-weight:800; letter-spacing:0.02em; }}
  .argus-alert .a-msg {{ font-weight:500; color:{c.text}; opacity:0.92; }}

  /* Recommendation card */
  .argus-rec {{ background: linear-gradient(160deg, {c.surface} 0%, {c.bg_alt} 100%);
     border:1px solid {c.border}; border-radius:14px; padding:16px 18px; }}
  .argus-rec .rec-action {{ font-size:1.35rem; font-weight:800; letter-spacing:-0.01em; }}
  .argus-rec .rec-note {{ color:{c.text_muted}; font-size:0.86rem; margin-top:4px; }}
  .argus-rec .rec-flag {{ margin-top:12px; padding:8px 12px; border-radius:9px;
     font-weight:700; font-size:0.86rem; }}

  /* Status pill + live dot */
  .argus-pill {{ display:inline-flex; align-items:center; gap:7px; padding:4px 12px;
     border-radius:999px; font-size:0.8rem; font-weight:700; letter-spacing:0.02em; }}
  .argus-dot {{ width:9px; height:9px; border-radius:50%; display:inline-block; }}
  .argus-dot.live {{ box-shadow: 0 0 0 3px rgba(45,212,191,0.28), 0 0 8px 2px rgba(45,212,191,0.45);
     animation: argus-pulse 1.6s ease-in-out infinite; }}
  @keyframes argus-pulse {{ 0%,100% {{ opacity:1; }} 50% {{ opacity:0.45; }} }}

  /* Status bar */
  .argus-status-bar {{ display:flex; flex-wrap:wrap; align-items:center; gap:10px 22px;
     background:{c.bg_alt}; border:1px solid {c.border}; border-radius:12px;
     padding:10px 16px; }}
  .argus-status-bar .sb-item {{ display:flex; align-items:center; gap:7px; font-size:0.84rem; }}
  .argus-status-bar .sb-label {{ color:{c.text_muted}; text-transform:uppercase;
     font-size:0.68rem; letter-spacing:0.06em; }}
  .argus-status-bar .sb-value {{ color:{c.text}; font-weight:700; font-variant-numeric:tabular-nums; }}

  /* Quasar widget tweaks for the dark theme */
  .q-field__native, .q-field__prefix, .q-field__suffix, .q-item__label {{ color:{c.text}; }}
  .q-tab {{ color:{c.text_muted}; }}
  .q-tab--active {{ color:{c.accent_soft}; }}
  .q-tab-panel {{ padding: 10px 0 0 0; }}
</style>
""")


# =========================================================================== #
# Small shared UI helpers
# =========================================================================== #
def _panel(title: str | None = None, *, classes: str = "") -> Any:
    """A styled dark panel (card). Returns the card element (use as context)."""
    card = ui.card().classes(f"argus-panel p-3 gap-2 {classes}")
    if title:
        with card:
            ui.label(title).classes("argus-title text-sm")
    return card


def _live_plot(fig: Any) -> Any:
    """Create a live Plotly chart element (config carried on the figure)."""
    return ui.plotly(_with_config(fig, _PLOTLY_LIVE)).classes("w-full")


def _static_plot(fig: Any) -> Any:
    return ui.plotly(_with_config(fig, _PLOTLY_STATIC)).classes("w-full")


def _with_config(fig: Any, config: dict) -> Any:
    """Attach a Plotly ``config`` so NiceGUI forwards it to ``Plotly.react``.

    NiceGUI serializes a ``go.Figure`` via ``to_plotly_json()`` (data+layout
    only); stashing ``layout.meta['config']`` here is ignored by Plotly but lets
    us pass a dict figure with a top-level ``config`` for mode-bar control.
    """
    d = fig.to_plotly_json() if hasattr(fig, "to_plotly_json") else dict(fig)
    d["config"] = config
    return d


def _update_plot(el: Any, fig: Any) -> None:
    """Swap a live Plotly figure into an existing element and push the update."""
    el.figure = _with_config(fig, _PLOTLY_LIVE)
    el.update()


def _env_flag(name: str, default: bool = False) -> bool:
    v = os.environ.get(name)
    return default if v is None else v.strip().lower() in {"1", "true", "yes", "on"}


def _config_from_env() -> orch.SessionConfig:
    """Fresh per-client config, with optional env overrides (used by Docker)."""
    cfg = orch.SessionConfig()
    cfg.use_api = _env_flag("ARGUS_DASHBOARD_USE_API", cfg.use_api)
    cfg.api_base_url = os.environ.get("ARGUS_API_BASE_URL", cfg.api_base_url)
    cfg.model = os.environ.get("ARGUS_DASHBOARD_MODEL", cfg.model)
    if (log_dir := os.environ.get("ARGUS_LOG_DIR")):
        cfg.log_dir = log_dir
    # Persist live predictions by default in deployed/containerised setups so the
    # Historical Explorer populates without the operator toggling the switch.
    cfg.persist_logs = _env_flag("ARGUS_DASHBOARD_PERSIST", cfg.persist_logs)
    return cfg


def _mode_label(use_api: bool) -> str:
    return "API" if use_api else "Standalone"


def _fmt(v: Any, ndigits: int = 3) -> str:
    """Format an optional numeric metric for a table cell."""
    if v is None:
        return "-"
    try:
        return f"{float(v):.{ndigits}f}"
    except (TypeError, ValueError):
        return str(v)


def _add_noise(accel: np.ndarray, noise: float, seed: int) -> np.ndarray:
    if noise <= 0:
        return accel
    rng = np.random.default_rng(seed + 999)
    rms = float(np.sqrt(np.mean(accel**2))) or 1.0
    return accel + rng.normal(0.0, noise * rms, size=accel.shape)


def _labeled_slider(label: str, lo: float, hi: float, step: float, value: float,
                    on_change: Any) -> Any:
    """A slider with a label + live value badge above it (returns the slider)."""
    with ui.column().classes("w-full gap-0 q-mb-xs"):
        with ui.row().classes("w-full items-center justify-between no-wrap"):
            ui.label(label).classes("argus-caption")
            val_lbl = ui.label(f"{value:g}").classes("argus-caption").style(
                f"color:{COLORS.accent_soft};font-weight:700")
        sl = ui.slider(min=lo, max=hi, step=step, value=value)

        def _handler(e: Any) -> None:
            val_lbl.set_text(f"{e.value:g}")
            on_change(e)

        sl.on_value_change(_handler)
    return sl


# =========================================================================== #
# Dashboard (one instance per connected client)
# =========================================================================== #
class Dashboard:
    """Owns the per-client orchestrator, all UI elements, and the refresh timer."""

    def __init__(self) -> None:
        self.orc = orch.SimulationOrchestrator(_config_from_env())
        self._last_gen = -1
        self._stft_tick = 0
        self._scenario_key = "progressive"  # bound to the scenario selector
        self._hist_df: pd.DataFrame | None = None
        self._els: dict[str, Any] = {}   # live-view elements updated each tick
        self.timer: Any = None

    # ---------------------------------------------------------------- lifecycle
    def build(self) -> None:
        _install_theme()
        ui.query("body").classes("argus-appbg")
        self._build_drawer()
        self._build_header()
        with ui.tabs().classes("w-full").props(f'active-color=positive indicator-color=positive') as tabs:
            ui.tab("live", label="Live Monitor", icon="sensors")
            ui.tab("lab", label="Simulation Lab", icon="science")
            ui.tab("history", label="Historical Explorer", icon="history")
            ui.tab("opt", label="Optimization", icon="factory")
            ui.tab("system", label="System & Models", icon="build")
        with ui.tab_panels(tabs, value="live").classes("w-full argus-appbg"):
            with ui.tab_panel("live"):
                self._build_live_tab()
            with ui.tab_panel("lab"):
                self._build_lab_tab()
            with ui.tab_panel("history"):
                self._build_history_tab()
            with ui.tab_panel("opt"):
                self._build_optimization_tab()
            with ui.tab_panel("system"):
                self._build_system_tab()

        # Single refresh timer; the callback is cheap when nothing changed.
        self.timer = ui.timer(0.1, self._tick)

    def _build_header(self) -> None:
        c = COLORS
        with ui.row().classes("w-full items-center justify-between q-mb-sm no-wrap"):
            with ui.row().classes("items-center gap-3 no-wrap"):
                ui.icon("visibility", size="1.8rem").style(f"color:{c.accent}")
                with ui.column().classes("gap-0"):
                    ui.label("ARGUS PANOPTES").classes("argus-title text-lg")
                    ui.label("Industrial Perception - live monitor").classes("argus-caption")
            with ui.row().classes("items-center gap-4 no-wrap"):
                self._els["hdr_mode"] = ui.label("Standalone").classes("argus-caption")
                self._els["hdr_model"] = ui.label(self.orc.config.model).classes("argus-caption")
        ui.separator().style(f"background:{c.border}")

    # ------------------------------------------------------------- left drawer
    def _build_drawer(self) -> None:
        """Persistent global controls: mode, model, sim params, thresholds."""
        c = COLORS
        with ui.left_drawer(value=True, bordered=True).style(
            f"background:linear-gradient(180deg,{c.bg_alt} 0%,{c.bg} 100%);width:320px"
        ):
            ui.label("Global controls").classes("argus-title text-base")
            ui.separator().style(f"background:{c.border}")

            # ---- Operation mode ----
            ui.label("Operation mode").classes("argus-caption q-mt-sm")
            self._els["mode_toggle"] = ui.toggle(
                {"standalone": "Standalone", "api": "Connected to API"},
                value="api" if self.orc.config.use_api else "standalone",
                on_change=self._on_mode_change,
            ).props("no-caps dense").classes("w-full")
            self._els["api_url"] = ui.input(
                "API base URL", value=self.orc.config.api_base_url,
                on_change=lambda e: self.orc.set_mode(True, e.value),
            ).classes("w-full").bind_visibility_from(
                self._els["mode_toggle"], "value", lambda v: v == "api"
            )

            # ---- Model ----
            ui.label("Model").classes("argus-caption q-mt-sm")
            entries = orch.available_model_entries()
            opts = {e["name"]: ("[ok] " if e["available"] else "[--] ") + e["name"] for e in entries}
            self._els["model_select"] = ui.select(
                opts, value=self.orc.config.model, on_change=self._on_model_change,
            ).props("dense options-dense").classes("w-full")
            self._els["model_note"] = ui.label("").classes("argus-caption")
            self._refresh_model_note()

            self._els["chunk_slider"] = _labeled_slider(
                "Analysis chunk (s)", 0.25, 1.0, 0.05, float(self.orc.config.chunk_s),
                lambda e: self.orc.set_chunk_s(e.value),
            )

            ui.separator().style(f"background:{c.border}")

            # ---- Simulation parameters (live) ----
            ui.label("Simulation parameters").classes("argus-title text-sm q-mt-sm")
            p = self.orc.config.sim_params
            self._els["alloy"] = ui.select(
                list(orch.ALLOY_OPTIONS), value=p["alloy"], label="Alloy",
                on_change=lambda e: self.orc.update_params(alloy=e.value),
            ).props("dense").classes("w-full")
            self._els["sfpm"] = _labeled_slider(
                "Blade speed (SFPM)", 500.0, 1200.0, 10.0, float(p["blade_speed_sfpm"]),
                lambda e: self.orc.update_params(blade_speed_sfpm=float(e.value)),
            )
            self._els["feed"] = _labeled_slider(
                "Feed / tooth (mm)", 0.05, 0.40, 0.01, float(p["feed_per_tooth_mm"]),
                lambda e: self.orc.update_params(feed_per_tooth_mm=float(e.value)),
            )
            self._els["depth"] = _labeled_slider(
                "Depth of cut (mm)", 5.0, 50.0, 1.0, float(p["depth_mm"]),
                lambda e: self.orc.update_params(depth_mm=float(e.value)),
            )
            self._els["teeth"] = _labeled_slider(
                "Number of teeth", 40, 120, 1, int(p["num_teeth"]),
                lambda e: self.orc.update_params(num_teeth=int(e.value)),
            )
            self._els["wear"] = _labeled_slider(
                "Injected wear (manual)", 0.0, 1.0, 0.01, self.orc.manual_wear,
                lambda e: self.orc.set_manual_wear(float(e.value)),
            )
            self._els["noise"] = _labeled_slider(
                "Sensor noise (x rms)", 0.0, 0.5, 0.05, 0.0,
                lambda e: self.orc.set_manual_noise(float(e.value)),
            )
            self._els["kin"] = ui.label("").classes("argus-caption")
            self._refresh_kinematics()

            ui.separator().style(f"background:{c.border}")

            # ---- Alert thresholds ----
            with ui.expansion("Alert thresholds", icon="warning").classes("w-full"):
                th = self.orc.config.thresholds
                _labeled_slider("Wear alert level", 0.0, 1.0, 0.05, float(th["wear_alert"]),
                                lambda e: self.orc.set_thresholds(wear_alert=e.value))
                _labeled_slider("Min confidence for anomaly", 0.0, 1.0, 0.05,
                                float(th["anomaly_confidence"]),
                                lambda e: self.orc.set_thresholds(anomaly_confidence=e.value))

            # ---- Utilities ----
            with ui.expansion("Utilities", icon="tune").classes("w-full"):
                self._els["persist"] = ui.switch(
                    "Persist live predictions to Parquet",
                    value=self.orc.config.persist_logs,
                    on_change=lambda e: self.orc.set_persist(e.value, self.orc.config.log_dir),
                )
                self._els["log_dir"] = ui.input(
                    "Log directory", value=self.orc.config.log_dir,
                    on_change=lambda e: self.orc.set_persist(self._els["persist"].value, e.value),
                ).classes("w-full")
                ui.button("Reload models / caches", icon="refresh",
                          on_click=self._reload_caches).props("flat dense").classes("w-full")

    def _on_mode_change(self, e: Any) -> None:
        use_api = e.value == "api"
        self.orc.set_mode(use_api, self._els["api_url"].value)
        ui.notify(f"Mode: {_mode_label(use_api)}", type="info")

    def _on_model_change(self, e: Any) -> None:
        self.orc.set_model(e.value)
        self._refresh_model_note()

    def _refresh_model_note(self) -> None:
        name = self.orc.config.model
        entries = {x["name"]: x for x in orch.available_model_entries()}
        ent = entries.get(name, {})
        if ent and not ent.get("available", False):
            self._els["model_note"].set_text("Artifact missing - inference will error.")
            self._els["model_note"].style(f"color:{COLORS.warning}")
        else:
            self._els["model_note"].set_text(f"{ent.get('kind','?')} - {ent.get('description','')}")
            self._els["model_note"].style(f"color:{COLORS.text_muted}")

    def _refresh_kinematics(self) -> None:
        p = self.orc.config.sim_params
        try:
            kin = orch.kinematics_preview(p["alloy"], p["blade_speed_sfpm"],
                                          p["feed_per_tooth_mm"], p["depth_mm"], p["num_teeth"])
            self._els["kin"].set_text(
                f"Derived: {kin['rpm']:.0f} RPM - TPF {kin['tpf_hz']:.0f} Hz - "
                f"MRR {kin['mrr_mm3_s']:.0f} mm3/s"
            )
        except Exception:
            self._els["kin"].set_text("")

    def _reload_caches(self) -> None:
        orch.clear_caches()
        ui.notify("Caches cleared; models will reload on next inference.", type="positive")

    # =================================================================== live
    def _build_live_tab(self) -> None:
        c = COLORS
        # ---- Control bar ----
        with ui.row().classes("w-full items-center gap-2 no-wrap q-mb-sm"):
            self._els["btn_start"] = ui.button(
                "Start", icon="play_arrow", on_click=self._on_start_stop,
            ).props("color=positive").classes("text-white")
            self._els["btn_pause"] = ui.button(
                "Pause", icon="pause", on_click=self._on_pause,
            ).props("flat")
            ui.button("Step", icon="skip_next", on_click=self._on_step).props("flat")
            ui.button("+Wear", icon="trending_up", on_click=self._on_bump_wear).props("flat")
            ui.space()
            ui.select(
                {k: f"{SCENARIOS[k].icon} {SCENARIOS[k].name}" for k in SCENARIOS},
                value="progressive", label="Scenario",
            ).props("dense options-dense").classes("w-56").bind_value(self, "_scenario_key")
            ui.button("Launch scenario", icon="rocket_launch",
                      on_click=self._on_launch_scenario).props("color=primary").classes("text-white")
            _labeled_slider("Rate (s/chunk)", 0.05, 0.5, 0.01, float(self.orc.config.delay_s),
                            lambda e: self.orc.set_delay(float(e.value)))

        # ---- Status bar ----
        self._els["status"] = ui.html("").classes("w-full")

        # ---- Alert banner ----
        self._els["alert"] = ui.html(
            alert_banner_html("info", "STANDBY", "Press Start or launch a scenario to begin.")
        ).classes("w-full")

        # ---- KPI gauges + cards ----
        with ui.row().classes("w-full items-stretch gap-3 no-wrap"):
            with _panel(classes="col grow"):
                self._els["gauges"] = _live_plot(self._empty_gauges())
            with ui.column().classes("gap-2").style("min-width:260px"):
                self._els["kpi_health"] = ui.html(kpi_card_html("Health state", "-"))
                self._els["kpi_ctf"] = ui.html(kpi_card_html("Cycle-time factor", "-",
                                                             accent=c.info))
                self._els["kpi_temp"] = ui.html(kpi_card_html("Cut-zone temp", "-",
                                                              accent=c.warning))

        # ---- Live plots (2x2) ----
        with ui.grid(columns=2).classes("w-full gap-3 q-mt-sm"):
            with _panel():
                self._els["p_wave"] = _live_plot(
                    plots.build_waveform_figure(np.array([]), uirevision="wave"))
            with _panel():
                self._els["p_fft"] = _live_plot(
                    plots.build_fft_figure(np.array([]), 40960.0, uirevision="fft"))
            with _panel():
                self._els["p_stft"] = _live_plot(
                    plots.build_stft_heatmap(np.zeros((0, 0)), np.array([]), np.array([]),
                                             uirevision="stft"))
            with _panel():
                self._els["p_trend"] = _live_plot(
                    plots.build_trend_figure(pd.DataFrame(), [], title="Recent trend",
                                             uirevision="trend"))

        # ---- Recommendation ----
        self._els["rec"] = ui.html("").classes("w-full q-mt-sm")

    def _empty_gauges(self) -> Any:
        return plots.build_gauge_row_figure(
            [{"value": 0.0, "title": "Blade wear"},
             {"value": 0.0, "title": "Quality score"},
             {"value": 0.0, "title": "Confidence"}],
            uirevision="argus_gauges_live",
        )

    # ------------------------------------------------------------ control acts
    def _on_start_stop(self) -> None:
        snap = self.orc.snapshot()
        if snap.running:
            self.orc.stop()
        else:
            self.orc.start_manual()
        self.timer.active = True

    def _on_pause(self) -> None:
        paused = self.orc.toggle_pause()
        ui.notify("Paused" if paused else "Resumed", type="info")

    def _on_step(self) -> None:
        try:
            self.orc.step_once()
        except Exception as exc:  # pragma: no cover - surfaced in banner
            ui.notify(str(exc), type="negative")

    def _on_bump_wear(self) -> None:
        w = self.orc.bump_wear(0.1)
        self._els["wear"].value = w
        ui.notify(f"Injected wear -> {w:.2f}", type="info")

    def _on_launch_scenario(self) -> None:
        key = getattr(self, "_scenario_key", "progressive")
        self.orc.start(key)
        self.timer.active = True
        ui.notify(f"Launching: {SCENARIOS[key].name}", type="positive")

    # =============================================================== refresh
    def _tick(self) -> None:
        """Reactive refresh: cheap no-op unless the orchestrator produced a frame."""
        snap = self.orc.snapshot()
        # Header labels (cheap; keep in sync with live mode/model switches).
        self._els["hdr_mode"].set_text(_mode_label(self.orc.config.use_api))
        self._els["hdr_model"].set_text(self.orc.config.model)
        self._update_start_button(snap)

        if snap.generation == self._last_gen:
            return
        self._last_gen = snap.generation
        try:
            self._render_live(snap)
        except Exception as exc:  # pragma: no cover - defensive
            self._els["alert"].set_content(
                alert_banner_html("critical", "RENDER ERROR", str(exc)))

    def _update_start_button(self, snap: orch.Snapshot) -> None:
        btn = self._els["btn_start"]
        if snap.running:
            btn.set_text("Stop")
            btn.props("color=negative icon=stop")
        else:
            btn.set_text("Start")
            btn.props("color=positive icon=play_arrow")

    def _render_live(self, snap: orch.Snapshot) -> None:
        c = COLORS
        # ---- Status bar ----
        pill = status_pill_html("LIVE" if snap.running else ("PAUSED" if snap.paused else "IDLE"),
                                c.accent if snap.running else c.text_muted, live=snap.running)
        lat = f"{snap.latency_ms:.2f} ms" if snap.latency_ms is not None else "-"
        self._els["status"].set_content(
            f"<div class='argus-status-bar'>"
            f"<div class='sb-item'>{pill}</div>"
            f"<div class='sb-item'><span class='sb-label'>Model</span>"
            f"<span class='sb-value'>{snap.model}</span></div>"
            f"<div class='sb-item'><span class='sb-label'>Mode</span>"
            f"<span class='sb-value'>{snap.mode}</span></div>"
            f"<div class='sb-item'><span class='sb-label'>Scenario</span>"
            f"<span class='sb-value'>{snap.scenario_name}</span></div>"
            f"<div class='sb-item'><span class='sb-label'>Chunk</span>"
            f"<span class='sb-value'>{snap.step}/{snap.max_steps}</span></div>"
            f"<div class='sb-item'><span class='sb-label'>Latency</span>"
            f"<span class='sb-value'>{lat}</span></div></div>"
        )

        if snap.error:
            self._els["alert"].set_content(
                alert_banner_html("critical", "ERROR", snap.error))
            return

        result = snap.current_result
        if not result:
            return
        p = result.get("predictions", {})
        wear = float(p.get("wear_level", 0.0))
        quality = float(p.get("quality_score", 0.0))
        conf = float(p.get("confidence", 0.0))
        ctf = float(p.get("cycle_time_factor", 0.0))
        state = str(p.get("health_state", "-"))
        anomaly = bool(p.get("anomaly_flag", False))

        # ---- Gauges (single figure, animated in place) ----
        _update_plot(self._els["gauges"], plots.build_gauge_row_figure(
            [{"value": wear, "title": "Blade wear",
              "thresholds": (0.45, self.orc.config.thresholds["wear_alert"])},
             {"value": quality, "title": "Quality score", "thresholds": (0.5, 0.8),
              "colors": (c.critical, c.warning, c.accent)},
             {"value": conf, "title": "Confidence", "thresholds": (0.4, 0.7),
              "colors": (c.critical, c.warning, c.accent)}],
            uirevision="argus_gauges_live",
        ))

        # ---- KPI cards ----
        self._els["kpi_health"].set_content(kpi_card_html(
            "Health state", state.capitalize(), accent=c.health_color(state),
            sub=("anomaly" if anomaly else "nominal")))
        self._els["kpi_ctf"].set_content(kpi_card_html(
            "Cycle-time factor", f"{ctf:.3f}", accent=c.info, sub="1.0 = sharp baseline"))
        temp = snap.mean_temp_c
        temp_str = f"{temp:.0f} C" if temp == temp else "-"
        self._els["kpi_temp"].set_content(kpi_card_html(
            "Cut-zone temp", temp_str, accent=c.warning, sub="IR pyrometer"))

        # ---- Alert banner ----
        level, title, msg = orch.alert_level_for(result, self.orc.config.thresholds)
        self._els["alert"].set_content(alert_banner_html(level, title, msg))

        # ---- Recommendation ----
        rec = result.get("recommendations", {})
        self._els["rec"].set_content(recommendation_card_html(
            rec.get("action", "monitor"), rec.get("note", ""),
            color=c.health_color(state),
            blade_change=bool(rec.get("blade_change_suggested", False))))

        # ---- Waveform + FFT (every frame; aggressively downsampled) ----
        if snap.wave_accel is not None:
            _update_plot(self._els["p_wave"], plots.build_waveform_figure(
                snap.wave_accel, t=snap.wave_t, max_points=orch.LIVE_WAVE_POINTS,
                uirevision="wave"))
            fs = float((snap.meta or {}).get("fs_hz", 40960.0))
            tpf = float((snap.meta or {}).get("tooth_pass_freq_hz", 0.0)) or None
            _update_plot(self._els["p_fft"], plots.build_fft_figure(
                snap.wave_accel, fs, tpf_hz=tpf, max_points=orch.LIVE_FFT_POINTS,
                uirevision="fft"))

        # ---- STFT heatmap (throttled: heaviest to serialize) ----
        self._stft_tick += 1
        if snap.stft is not None and np.asarray(snap.stft["power"]).size and (
            not snap.running or self._stft_tick % 3 == 0
        ):
            s = snap.stft
            _update_plot(self._els["p_stft"], plots.build_stft_heatmap(
                s["power"], s["freqs"], s["times"],
                max_freq_bins=200, max_time_bins=180, uirevision="stft"))

        # ---- Trend (recent history) ----
        if snap.history:
            df = pd.DataFrame(snap.history).tail(40).reset_index()
            _update_plot(self._els["p_trend"], plots.build_trend_figure(
                df, ["wear_level", "quality_score"], x_col="index",
                title="Recent trend (wear / quality)", labels=_TREND_LABELS,
                color_map={"wear_level": c.warning, "quality_score": c.accent},
                height=300, uirevision="trend"))

    # ==================================================================== lab
    def _build_lab_tab(self) -> None:
        c = COLORS
        ui.label("Simulation Lab").classes("argus-title text-lg")
        ui.label("Controlled single / multi-model / robustness runs on synthetic chunks.") \
            .classes("argus-caption")

        # ---- Presets ----
        with _panel("Presets (from experiments)", classes="w-full q-mt-sm"):
            with ui.row().classes("w-full gap-3 no-wrap"):
                for key, preset in LAB_PRESETS.items():
                    with ui.column().classes("col gap-1"):
                        ui.label(preset["label"]).style(f"color:{c.text};font-weight:700")
                        ui.label(preset["note"]).classes("argus-caption")
                        ui.button("Load", icon="download",
                                  on_click=lambda _e, k=key: self._load_preset(k)) \
                            .props("flat dense color=primary")

        # ---- Form ----
        with _panel(classes="w-full q-mt-sm gap-2"):
            with ui.row().classes("w-full items-center gap-3 no-wrap"):
                self._els["lab_models"] = ui.select(
                    orch.known_model_names(), value=[self.orc.config.model], multiple=True,
                    label="Models to compare",
                ).props("dense use-chips").classes("col")
                self._els["lab_batch_n"] = ui.number("Batch N", value=50, min=5, max=300, step=5) \
                    .props("dense").classes("w-28")
            with ui.row().classes("gap-2 no-wrap"):
                ui.button("Run single", icon="bolt", on_click=self._lab_single).props("flat")
                ui.button("Run comparison", icon="compare_arrows",
                          on_click=self._lab_compare).props("color=primary").classes("text-white")
                ui.button("Run robustness batch", icon="analytics",
                          on_click=self._lab_batch).props("flat")
            ui.label("Uses the current sidebar operating point, injected wear, and sensor noise.") \
                .classes("argus-caption")

        self._els["lab_results"] = ui.column().classes("w-full q-mt-sm gap-2")

    def _load_preset(self, key: str) -> None:
        preset = LAB_PRESETS[key]
        pr = preset["params"]
        self.orc.update_params(**pr)
        self.orc.set_manual_wear(float(preset["wear"]))
        self.orc.set_manual_noise(float(preset["noise"]))
        # Reflect into the sidebar widgets.
        self._els["alloy"].value = pr["alloy"]
        self._els["sfpm"].value = pr["blade_speed_sfpm"]
        self._els["feed"].value = pr["feed_per_tooth_mm"]
        self._els["depth"].value = pr["depth_mm"]
        self._els["teeth"].value = pr["num_teeth"]
        self._els["wear"].value = float(preset["wear"])
        self._els["noise"].value = float(preset["noise"])
        self._refresh_kinematics()
        ui.notify(f"Loaded preset: {preset['label']}", type="positive")

    def _lab_context(self) -> tuple[dict, float, float]:
        p = dict(self.orc.config.sim_params)
        return p, self.orc.manual_wear, self.orc.manual_noise

    async def _lab_single(self) -> None:
        params, wear, noise = self._lab_context()
        cfg = self.orc.config

        def _work() -> dict:
            chunk = orch.generate_chunk(params, wear, 0, float(cfg.chunk_s))
            accel = _add_noise(chunk["accel"], noise, 0)
            result = orch.run_inference(
                accel, chunk["meta"], model=cfg.model, use_api=cfg.use_api,
                api_base_url=cfg.api_base_url, chunk_s=cfg.chunk_s, persist=False,
                log_dir=cfg.log_dir, return_features=True, wear_injected=wear)
            stft = orch.compute_stft_for_display(accel, float(chunk["meta"].get("fs_hz", 40960.0)))
            return {"chunk": chunk, "accel": accel, "result": result, "stft": stft}

        try:
            out = await run.io_bound(_work)
        except Exception as exc:
            ui.notify(str(exc), type="negative"); return
        self._render_lab_single(out, wear)

    def _render_lab_single(self, out: dict, wear: float) -> None:
        c = COLORS
        chunk, accel, result, stft = out["chunk"], out["accel"], out["result"], out["stft"]
        p = result["predictions"]
        cont = self._els["lab_results"]
        cont.clear()
        with cont:
            with ui.row().classes("w-full gap-2 no-wrap"):
                ui.html(kpi_card_html("Wear", f"{p['wear_level']:.3f}", accent=c.warning,
                                      sub=f"injected {wear:.2f}")).classes("col")
                ui.html(kpi_card_html("Health", str(p["health_state"]).capitalize(),
                                      accent=c.health_color(p["health_state"]))).classes("col")
                ui.html(kpi_card_html("Quality", f"{p['quality_score']:.3f}",
                                      accent=c.accent)).classes("col")
                ui.html(kpi_card_html("Latency", f"{result['latency_ms']:.2f} ms",
                                      accent=c.info)).classes("col")
            with ui.grid(columns=2).classes("w-full gap-3"):
                fs = float(chunk["meta"].get("fs_hz", 40960.0))
                tpf = float(chunk["meta"].get("tooth_pass_freq_hz", 0.0)) or None
                with _panel():
                    _static_plot(plots.build_waveform_figure(accel, t=chunk["t"],
                                                             title="Chunk waveform"))
                with _panel():
                    _static_plot(plots.build_fft_figure(accel, fs, tpf_hz=tpf, title="Chunk FFT"))
                with _panel():
                    _static_plot(plots.build_stft_heatmap(stft["power"], stft["freqs"],
                                                          stft["times"]))
                with _panel():
                    _static_plot(plots.build_health_prob_bar(p["health_probs"]))
            feats = result.get("features", {})
            if feats:
                with ui.expansion("DSP features (this chunk)", icon="functions").classes("w-full"):
                    rows = [{"feature": k, "value": round(float(v), 5)}
                            for k, v in sorted(feats.items())]
                    ui.table(columns=[{"name": "feature", "label": "feature", "field": "feature",
                                       "align": "left"},
                                      {"name": "value", "label": "value", "field": "value"}],
                             rows=rows, row_key="feature").classes("w-full").props("dense")

    async def _lab_compare(self) -> None:
        params, wear, noise = self._lab_context()
        cfg = self.orc.config
        models = list(self._els["lab_models"].value) or [cfg.model]

        def _work() -> list[dict]:
            chunk = orch.generate_chunk(params, wear, 0, float(cfg.chunk_s))
            accel = _add_noise(chunk["accel"], noise, 0)
            rows = []
            for m in models:
                try:
                    r = orch.run_inference(accel, chunk["meta"], model=m, use_api=cfg.use_api,
                                           api_base_url=cfg.api_base_url, chunk_s=cfg.chunk_s,
                                           persist=False, log_dir=cfg.log_dir, wear_injected=wear)
                except RuntimeError:
                    continue
                pr = r["predictions"]
                rows.append({"model": m, "wear_level": round(pr["wear_level"], 3),
                             "abs_err": round(abs(pr["wear_level"] - wear), 3),
                             "quality": round(pr["quality_score"], 3),
                             "health": pr["health_state"],
                             "confidence": round(pr["confidence"], 3),
                             "latency_ms": round(r["latency_ms"], 2)})
            return rows

        try:
            rows = await run.io_bound(_work)
        except Exception as exc:
            ui.notify(str(exc), type="negative"); return
        cont = self._els["lab_results"]; cont.clear()
        with cont:
            if not rows:
                ui.label("No models produced a result.").classes("argus-caption"); return
            ui.label(f"Injected wear = {wear:.2f}  -  noise x{noise:.2f} rms").classes("argus-caption")
            with ui.row().classes("w-full gap-3 no-wrap items-start"):
                cols = [{"name": k, "label": k, "field": k,
                         "align": "left" if k in ("model", "health") else "right"}
                        for k in rows[0]]
                ui.table(columns=cols, rows=rows, row_key="model").classes("col").props("dense")
                with _panel().classes("col"):
                    _static_plot(plots.build_grouped_bar(
                        [r["model"] for r in rows],
                        {"wear_level": [r["wear_level"] for r in rows],
                         "quality": [r["quality"] for r in rows]},
                        title="Wear / quality by model", y_title="value", height=320))

    async def _lab_batch(self) -> None:
        params, wear, noise = self._lab_context()
        cfg = self.orc.config
        n = int(self._els["lab_batch_n"].value or 50)

        def _work() -> list[dict]:
            rows = []
            for i in range(n):
                chunk = orch.generate_chunk(params, wear, i, float(cfg.chunk_s))
                accel = _add_noise(chunk["accel"], noise, i)
                r = orch.run_inference(accel, chunk["meta"], model=cfg.model, use_api=cfg.use_api,
                                       api_base_url=cfg.api_base_url, chunk_s=cfg.chunk_s,
                                       persist=False, log_dir=cfg.log_dir, wear_injected=wear)
                pr = r["predictions"]
                rows.append({"wear_level": pr["wear_level"], "quality_score": pr["quality_score"],
                             "confidence": pr["confidence"], "health_state": pr["health_state"],
                             "latency_ms": r["latency_ms"]})
            return rows

        ui.notify(f"Running robustness batch (N={n})...", type="info")
        try:
            rows = await run.io_bound(_work)
        except Exception as exc:
            ui.notify(str(exc), type="negative"); return
        df = pd.DataFrame(rows)
        cont = self._els["lab_results"]; cont.clear()
        c = COLORS
        with cont:
            with ui.row().classes("w-full gap-2 no-wrap"):
                ui.html(kpi_card_html("Wear mean", f"{df['wear_level'].mean():.3f}", accent=c.warning,
                                      sub=f"sd {df['wear_level'].std():.3f}")).classes("col")
                ui.html(kpi_card_html("Wear MAE", f"{(df['wear_level']-wear).abs().mean():.3f}",
                                      accent=c.critical, sub=f"vs injected {wear:.2f}")).classes("col")
                ui.html(kpi_card_html("Quality mean", f"{df['quality_score'].mean():.3f}",
                                      accent=c.accent)).classes("col")
                ui.html(kpi_card_html("Latency p50", f"{df['latency_ms'].median():.2f} ms",
                                      accent=c.info)).classes("col")
            with ui.grid(columns=2).classes("w-full gap-3"):
                with _panel():
                    _static_plot(plots.build_histogram(df["wear_level"],
                                 title="Wear prediction distribution", x_title="wear_level",
                                 color=c.warning))
                with _panel():
                    counts = df["health_state"].value_counts()
                    _static_plot(plots.build_pie(counts.index.tolist(), counts.values.tolist(),
                                 title="Health-state distribution", color_map=c.health))

    # ================================================================ history
    def _build_history_tab(self) -> None:
        ui.label("Historical Explorer").classes("argus-title text-lg")
        ui.label("Query the partitioned Parquet inference logs and drill into any record.") \
            .classes("argus-caption")
        with ui.row().classes("items-center gap-2 q-my-sm no-wrap"):
            ui.button("Load / Refresh logs", icon="refresh",
                      on_click=self._load_history).props("color=primary").classes("text-white")
            ui.label(f"Reading from '{self.orc.config.log_dir}'").classes("argus-caption")
        self._els["hist_body"] = ui.column().classes("w-full gap-2")
        self._hist_df: pd.DataFrame | None = None

    def _load_history(self) -> None:
        try:
            df = orch.load_logs(self.orc.config.log_dir)
        except RuntimeError as exc:
            ui.notify(str(exc), type="negative"); return
        body = self._els["hist_body"]; body.clear()
        if df is None or len(df) == 0:
            with body:
                ui.label("No inference logs found yet. Enable 'Persist live predictions' in the "
                         "sidebar and run a scenario, or: python scripts/stream_demo.py "
                         "--model 1dcnn_normnone --duration-s 5 --wear 0.6").classes("argus-caption")
            return
        df = df.copy()
        if "timestamp" in df:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
            df = df.sort_values("timestamp")
        self._hist_df = df.reset_index(drop=True)
        self._render_history(self._hist_df)

    def _render_history(self, df: pd.DataFrame) -> None:
        c = COLORS
        body = self._els["hist_body"]; body.clear()
        fdf = df.tail(400).reset_index(drop=True)
        with body:
            # ---- Summary ----
            with ui.row().classes("w-full gap-2 no-wrap"):
                ui.html(kpi_card_html("Records", f"{len(fdf):,}", accent=c.info)).classes("col")
                if "pred_anomaly_flag" in fdf:
                    ui.html(kpi_card_html("Anomaly rate",
                            f"{fdf['pred_anomaly_flag'].mean()*100:.1f}%",
                            accent=c.critical)).classes("col")
                if "pred_wear_level" in fdf:
                    ui.html(kpi_card_html("Avg wear", f"{fdf['pred_wear_level'].mean():.3f}",
                            accent=c.warning)).classes("col")
                if "latency_ms" in fdf:
                    ui.html(kpi_card_html("Latency p50", f"{fdf['latency_ms'].median():.2f} ms",
                            accent=c.accent)).classes("col")
            # ---- Trend + pie ----
            trend_cols = [x for x in ["pred_wear_level", "pred_cycle_time_factor",
                                      "pred_quality_score"] if x in fdf]
            x_col = "timestamp" if "timestamp" in fdf else None
            with ui.grid(columns=2).classes("w-full gap-3"):
                with _panel():
                    _static_plot(plots.build_trend_figure(
                        fdf, trend_cols, x_col=x_col, title="Prediction trends", range_slider=True,
                        labels={"pred_wear_level": "Wear",
                                "pred_cycle_time_factor": "Cycle-time factor",
                                "pred_quality_score": "Quality"},
                        color_map={"pred_wear_level": c.warning,
                                   "pred_cycle_time_factor": c.info,
                                   "pred_quality_score": c.accent}, uirevision=None))
                with _panel():
                    if "pred_health_state" in fdf:
                        counts = fdf["pred_health_state"].value_counts()
                        _static_plot(plots.build_pie(counts.index.tolist(),
                                     counts.values.tolist(), title="Health-state distribution",
                                     color_map=c.health))
            # ---- Records table ----
            show = [x for x in ["timestamp", "model", "pred_wear_level", "pred_health_state",
                                "pred_confidence", "pred_anomaly_flag", "action", "latency_ms"]
                    if x in fdf]
            tdf = fdf[show].tail(200).copy()
            if "timestamp" in tdf:
                tdf["timestamp"] = tdf["timestamp"].astype(str).str.slice(0, 19)
            for col in tdf.select_dtypes("float").columns:
                tdf[col] = tdf[col].round(3)
            rows = tdf.to_dict("records")
            cols = [{"name": x, "label": x, "field": x,
                     "align": "left" if x in ("timestamp", "model", "pred_health_state", "action")
                     else "right"} for x in show]
            ui.label("Records").classes("argus-title text-sm q-mt-sm")
            ui.table(columns=cols, rows=rows, row_key="timestamp",
                     pagination=10).classes("w-full").props("dense")

            # ---- Reconstruct selected record ----
            ui.label("Reconstruct & re-infer a record").classes("argus-title text-sm q-mt-md")
            ui.label("Raw waveforms aren't stored in Parquet; this re-generates the chunk from the "
                     "logged operating point and re-runs inference.").classes("argus-caption")
            labels = {i: (f"#{int(r.get('chunk_id', i) or i)} - {r.get('model', '?')} - "
                          f"wear={float(r.get('pred_wear_level', 0) or 0):.2f} - "
                          f"{r.get('pred_health_state', '?')}")
                      for i, r in fdf.iterrows()}
            with ui.row().classes("items-center gap-2 no-wrap w-full"):
                self._els["hist_sel"] = ui.select(labels, value=len(fdf) - 1,
                                                   label="Record").props("dense").classes("col")
                ui.button("Reconstruct", icon="insights",
                          on_click=self._reconstruct_record).props("color=primary").classes("text-white")
            self._els["hist_recon"] = ui.column().classes("w-full gap-2")

    async def _reconstruct_record(self) -> None:
        if self._hist_df is None:
            return
        fdf = self._hist_df.tail(400).reset_index(drop=True)
        idx = int(self._els["hist_sel"].value)
        row = fdf.iloc[idx]
        wear = (float(row["wear_injected"]) if "wear_injected" in row
                and pd.notna(row["wear_injected"]) else 0.5)
        params = orch.params_from_log_row(row)
        model = str(row.get("model", self.orc.config.model))
        seed = int(row.get("chunk_id", 0) or 0)
        cfg = self.orc.config

        def _work() -> dict:
            chunk = orch.generate_chunk(params, wear, seed, float(cfg.chunk_s))
            result = orch.run_inference(
                chunk["accel"], chunk["meta"], model=model, use_api=cfg.use_api,
                api_base_url=cfg.api_base_url, chunk_s=cfg.chunk_s, persist=False,
                log_dir=cfg.log_dir, return_features=True, wear_injected=wear)
            stft = orch.compute_stft_for_display(chunk["accel"],
                                                 float(chunk["meta"].get("fs_hz", 40960.0)))
            return {"chunk": chunk, "result": result, "stft": stft}

        try:
            out = await run.io_bound(_work)
        except Exception as exc:
            ui.notify(str(exc), type="negative"); return
        chunk, result, stft = out["chunk"], out["result"], out["stft"]
        cont = self._els["hist_recon"]; cont.clear()
        with cont:
            ui.label(f"Reconstructed from logged operating point (wear_injected={wear:.2f}, "
                     f"model={model}).").classes("argus-caption")
            fs = float(chunk["meta"].get("fs_hz", 40960.0))
            tpf = float(chunk["meta"].get("tooth_pass_freq_hz", 0.0)) or None
            with ui.grid(columns=2).classes("w-full gap-3"):
                with _panel():
                    _static_plot(plots.build_waveform_figure(chunk["accel"], t=chunk["t"],
                                 title="Reconstructed waveform"))
                with _panel():
                    _static_plot(plots.build_fft_figure(chunk["accel"], fs, tpf_hz=tpf,
                                 title="Reconstructed FFT"))
                with _panel():
                    _static_plot(plots.build_stft_heatmap(stft["power"], stft["freqs"],
                                 stft["times"], title="Reconstructed STFT"))
                with _panel():
                    _static_plot(plots.build_health_prob_bar(result["predictions"]["health_probs"],
                                 title="Re-inferred health probs"))
            with ui.expansion("Full re-inferred payload (JSON)", icon="data_object").classes("w-full"):
                ui.code(json.dumps(result, indent=2, default=str), language="json").classes("w-full")

    # =========================================================== optimization
    def _build_optimization_tab(self) -> None:
        c = COLORS
        ui.label("Optimization Sandbox").classes("argus-title text-lg")
        ui.label("Accurate perception (wear / cycle-time / quality) feeds a transparent "
                 "downstream production-planning, costing & nesting model.").classes("argus-caption")

        with ui.row().classes("items-center gap-2 q-my-sm no-wrap"):
            self._els["opt_src"] = ui.toggle(
                {"live": "Latest live result", "manual": "Manual override"},
                value="live").props("no-caps dense")
            ui.button("Compute impact", icon="calculate",
                      on_click=self._compute_optimization).props("color=primary").classes("text-white")

        # Manual override inputs (shown only in manual mode).
        self._els["opt_manual"] = ui.row().classes("w-full gap-3 no-wrap")
        with self._els["opt_manual"]:
            self._els["opt_wear"] = ui.number("Wear", value=0.6, min=0, max=1, step=0.01).props("dense")
            self._els["opt_ctf"] = ui.number("Cycle-time factor", value=1.25, min=1, max=2,
                                             step=0.01).props("dense")
            self._els["opt_quality"] = ui.number("Quality", value=0.75, min=0, max=1,
                                                 step=0.01).props("dense")
            self._els["opt_conf"] = ui.number("Confidence", value=0.8, min=0, max=1,
                                              step=0.01).props("dense")
            self._els["opt_state"] = ui.select(_HEALTH_ORDER, value="warning",
                                               label="Health").props("dense")
        self._els["opt_manual"].bind_visibility_from(self._els["opt_src"], "value",
                                                      lambda v: v == "manual")

        with ui.expansion("Shop-floor assumptions (editable)", icon="settings").classes("w-full"):
            with ui.row().classes("w-full gap-3 no-wrap"):
                self._els["opt_cycle"] = ui.number("Baseline cycle (s)", value=45.0).props("dense")
                self._els["opt_bladelife"] = ui.number("Blade life (cuts)", value=1000.0).props("dense")
                self._els["opt_rate"] = ui.number("Machine $/hr", value=90.0).props("dense")
                self._els["opt_mat"] = ui.number("Material $/part", value=12.0).props("dense")
                self._els["opt_bladecost"] = ui.number("Blade $", value=220.0).props("dense")

        self._els["opt_results"] = ui.column().classes("w-full q-mt-sm gap-2")

    def _compute_optimization(self) -> None:
        c = COLORS
        if self._els["opt_src"].value == "live":
            snap = self.orc.snapshot()
            if not snap.current_result:
                ui.notify("No live result yet - run the Live Monitor or use Manual override.",
                          type="warning"); return
            p = snap.current_result["predictions"]
            state = opt.PerceptionState(
                wear_level=p["wear_level"], cycle_time_factor=p["cycle_time_factor"],
                quality_score=p["quality_score"], health_state=p["health_state"],
                confidence=p["confidence"], anomaly_flag=p["anomaly_flag"])
        else:
            state = opt.PerceptionState(
                wear_level=float(self._els["opt_wear"].value),
                cycle_time_factor=float(self._els["opt_ctf"].value),
                quality_score=float(self._els["opt_quality"].value),
                confidence=float(self._els["opt_conf"].value),
                health_state=str(self._els["opt_state"].value))
        inp = opt.ProductionInputs(
            baseline_cycle_time_s=float(self._els["opt_cycle"].value),
            blade_life_cycles=float(self._els["opt_bladelife"].value),
            machine_rate_usd_per_hr=float(self._els["opt_rate"].value),
            material_cost_usd_per_part=float(self._els["opt_mat"].value),
            blade_cost_usd=float(self._els["opt_bladecost"].value))
        impact = opt.compute_production_impact(state, inp)
        cur, ref, delta = impact["current"], impact["reference"], impact["delta"]
        m = impact["maintenance"]
        payload = opt.downstream_payload(state, inp, impact, model=self.orc.config.model)

        cont = self._els["opt_results"]; cont.clear()
        with cont:
            ui.html(alert_banner_html(m["level"],
                    f"MAINTENANCE: {m['urgency'].replace('_', ' ').upper()}", m["message"])) \
                .classes("w-full")
            with ui.row().classes("w-full gap-2 no-wrap"):
                ui.html(kpi_card_html("Efficiency", f"{cur['efficiency_score']:.0f}", accent=c.accent,
                        sub=f"{delta['efficiency_score']:+.0f} vs sharp")).classes("col")
                ui.html(kpi_card_html("Good parts/hr", f"{cur['good_pph']:.1f}", accent=c.info,
                        sub=f"{delta['good_pph']:+.1f}")).classes("col")
                ui.html(kpi_card_html("Cost/good part", f"${cur['cost_per_good_part']:.2f}",
                        accent=c.warning, sub=f"${delta['cost_per_good_part']:+.2f}")).classes("col")
                ui.html(kpi_card_html("Blade life left", f"{cur['remaining_blade_life_cycles']:.0f}",
                        accent=c.purple, sub="cuts")).classes("col")
            with ui.grid(columns=2).classes("w-full gap-3"):
                with _panel():
                    _static_plot(plots.build_grouped_bar(
                        ["Cycle time (s)", "Good parts/hr", "Yield %"],
                        {"Sharp blade": [ref["cycle_time_s"], ref["good_pph"], ref["yield_pct"]],
                         "Current": [cur["cycle_time_s"], cur["good_pph"], cur["yield_pct"]]},
                        title="Sharp-blade baseline vs current",
                        colors=[c.neutral, c.accent_soft], y_title="value"))
                with _panel():
                    _static_plot(plots.build_gauge_figure(
                        cur["efficiency_score"], "Production efficiency", min_val=0, max_val=100,
                        thresholds=(60, 85), colors=(c.critical, c.warning, c.accent),
                        number_format=".0f", reference=ref["efficiency_score"], height=300))
            with ui.expansion("Downstream payload (clean contract)", icon="data_object") \
                    .classes("w-full"):
                ui.code(json.dumps(payload, indent=2), language="json").classes("w-full")

    # ================================================================= system
    def _build_system_tab(self) -> None:
        c = COLORS
        ui.label("System & Models").classes("argus-title text-lg")
        # ---- Model inventory ----
        entries = orch.available_model_entries()
        rows = []
        for e in entries:
            s = metrics.model_metric_summary(e["name"])
            rows.append({"model": e["name"], "kind": e["kind"],
                         "available": "yes" if e["available"] else "no",
                         "wear_MAE": _fmt(s["wear_mae"]), "health_F1": _fmt(s["health_f1"]),
                         "latency_ms": _fmt(s["latency_ms"], 2), "notes": e["description"]})
        with _panel("Model inventory", classes="w-full"):
            cols = [{"name": k, "label": k, "field": k,
                     "align": "left" if k in ("model", "kind", "notes", "available") else "right"}
                    for k in rows[0]]
            ui.table(columns=cols, rows=rows, row_key="model").classes("w-full").props("dense")

        with ui.grid(columns=2).classes("w-full gap-3 q-mt-sm"):
            with _panel("Edge latency (ONNX CPU, p50)"):
                lat_rows = [{"model": n, "p50": metrics.typical_latency_ms(n)}
                            for n in orch.known_model_names()
                            if metrics.typical_latency_ms(n) is not None]
                if lat_rows:
                    _static_plot(plots.build_grouped_bar(
                        [r["model"] for r in lat_rows], {"p50 ms": [r["p50"] for r in lat_rows]},
                        title="Single-chunk latency", y_title="ms", text_format=".2f", height=300))
            with _panel("Robustness & edge findings"):
                ui.markdown(metrics.robustness_notes()).classes("text-sm")

        with ui.expansion("Effective configuration", icon="settings").classes("w-full q-mt-sm"):
            cfg = self.orc.config
            ui.code(json.dumps({
                "mode": _mode_label(cfg.use_api), "model": cfg.model, "chunk_s": cfg.chunk_s,
                "sim_params": cfg.sim_params, "thresholds": cfg.thresholds,
                "persist_logs": cfg.persist_logs, "log_dir": cfg.log_dir,
                "api_base_url": cfg.api_base_url,
            }, indent=2), language="json").classes("w-full")

# =========================================================================== #
# Page + entry point
# =========================================================================== #
@ui.page("/")
def _index() -> None:
    """Build a fresh, isolated dashboard for each connected client."""
    dash = Dashboard()
    dash.build()
    # Tidy shutdown of this client's background thread on disconnect.
    ui.context.client.on_disconnect(dash.orc.shutdown)


def _dashboard_host() -> str:
    if "ARGUS_DASHBOARD_HOST" in os.environ:
        return os.environ["ARGUS_DASHBOARD_HOST"]
    # PaaS hosts (e.g. Render) set PORT; bind externally when present.
    if os.environ.get("PORT"):
        return "0.0.0.0"
    return "127.0.0.1"


def _dashboard_port() -> int:
    for key in ("PORT", "ARGUS_DASHBOARD_PORT"):
        if key in os.environ:
            return int(os.environ[key])
    return 8080


def main() -> None:
    ui.run(
        title="Argus Panoptes | Industrial Perception",
        host=_dashboard_host(),
        port=_dashboard_port(),
        dark=True,
        reload=False,
        show=_env_flag("ARGUS_DASHBOARD_SHOW", False),
        favicon="\U0001f441\ufe0f",
    )


# ``ui.run`` must be reachable at import time for NiceGUI's (re)start machinery.
if __name__ in {"__main__", "__mp_main__"}:
    main()
