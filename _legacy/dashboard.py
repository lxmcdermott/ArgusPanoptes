"""Argus Panoptes - Industrial Perception Dashboard (Day 5 ops layer).

A production-quality Streamlit + Plotly monitoring UI for the multi-modal
industrial perception stack. It turns live (simulated) saw / CNC vibration into
real-time wear / health / quality predictions and shows how those perception
outputs drive downstream production planning, costing, and optimization.

Operation modes
---------------
* **Standalone (direct)** - an in-process :class:`StreamingPerceptor` for the
  lowest-latency, dependency-free demo (default; recommended for recordings).
* **Connected to API** - the same contract over HTTP against the FastAPI service
  (``uvicorn app.main:app``) for a true client/server showcase.

Tabs
----
1. **Live Monitor** - real-time waveform / FFT / STFT + KPI gauges, alerts, and
   recommendations, driven by a self-refreshing ``st.fragment`` loop.
2. **Simulation Lab** - single / multi-model / batch runs with distributions.
3. **Historical Explorer** - query + filter the partitioned Parquet inference logs.
4. **Optimization Sandbox** - transparent downstream production-impact model.
5. **System Health & Models** - model inventory, latency, robustness, config.

Run
---
    pip install -e ".[ml,dl,app,dashboard]"
    streamlit run dashboard.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

# Robust imports: ensure the repo root is importable whether the package was
# installed editable (``pip install -e .``) or the script is run in-place.
_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

st.set_page_config(
    page_title="Argus Panoptes | Industrial Perception Dashboard",
    page_icon="\U0001f441\ufe0f",
    layout="wide",
    initial_sidebar_state="expanded",
)

import infra  # noqa: E402  # legacy: now lives beside this file in _legacy/
from dashviz import metrics, plots  # noqa: E402
from dashviz import optimization as opt  # noqa: E402
from dashviz.scenarios import SCENARIOS  # noqa: E402
from dashviz.theme import (  # noqa: E402
    COLORS,
    PLOTLY_CONFIG,
    PLOTLY_CONFIG_LIVE,
    alert_banner_html,
    build_css,
    kpi_card_html,
    recommendation_card_html,
    status_pill_html,
)

#: Simulation Lab presets (known operating points + expected behavior notes).
LAB_PRESETS: dict[str, dict[str, Any]] = {
    "clean_sharp": {
        "label": "Clean sharp blade",
        "params": {"alloy": "6061", "blade_speed_sfpm": 800.0, "feed_per_tooth_mm": 0.12,
                   "depth_mm": 25.0, "num_teeth": 80},
        "wear": 0.12,
        "noise": 0.0,
        "seed": 42,
        "note": "Nominal cut; expect healthy/monitor states and low wear MAE on clean data.",
    },
    "high_wear": {
        "label": "High wear (0.85)",
        "params": {"alloy": "7075", "blade_speed_sfpm": 900.0, "feed_per_tooth_mm": 0.18,
                   "depth_mm": 32.0, "num_teeth": 72},
        "wear": 0.85,
        "noise": 0.0,
        "seed": 7,
        "note": "End-of-life blade; expect warning/critical health and blade-change flag.",
    },
    "noisy_robustness": {
        "label": "Noisy input (sd=0.5·rms)",
        "params": {"alloy": "6061", "blade_speed_sfpm": 800.0, "feed_per_tooth_mm": 0.12,
                   "depth_mm": 25.0, "num_teeth": 80},
        "wear": 0.55,
        "noise": 0.5,
        "seed": 99,
        "note": "Robustness ablation case: compare 1dcnn vs 1dcnn_noisy — noisy variant "
        "should hold ~3× better wear MAE under corruption.",
    },
}
_TREND_LABELS = {
    "wear_level": "Wear level",
    "cycle_time_factor": "Cycle-time factor",
    "quality_score": "Quality score",
    "confidence": "Confidence",
    "mean_temp_c": "Cut-zone temp (\u00b0C)",
}


# =========================================================================== #
# Small shared render helpers
# =========================================================================== #
def _plotly(fig: Any, *, key: str) -> None:
    """Render a Plotly figure with the shared responsive dark config."""
    st.plotly_chart(fig, width="stretch", config=PLOTLY_CONFIG, key=key)


def _plotly_live(fig: Any, *, key: str) -> None:
    """Render a live-view figure (no mode bar) for the smoothest refresh."""
    st.plotly_chart(fig, width="stretch", config=PLOTLY_CONFIG_LIVE, key=key)


def _md(html: str) -> None:
    st.markdown(html, unsafe_allow_html=True)


def _mode_label() -> str:
    return "API" if st.session_state.use_api_mode else "Standalone"


# =========================================================================== #
# Sidebar
# =========================================================================== #
def render_sidebar() -> None:
    """Persistent global controls: mode, model, sim params, thresholds, demos."""
    ss = st.session_state
    with st.sidebar:
        _md(
            f"<div style='display:flex;align-items:center;gap:10px;margin-bottom:2px'>"
            f"<span style='font-size:1.7rem'>\U0001f441\ufe0f</span>"
            f"<div><div style='font-weight:800;font-size:1.08rem;color:{COLORS.text}'>"
            f"ARGUS PANOPTES</div>"
            f"<div style='color:{COLORS.text_muted};font-size:0.72rem;letter-spacing:0.08em'>"
            f"INDUSTRIAL PERCEPTION</div></div></div>"
        )
        st.divider()

        # ---- Operation mode ----
        st.markdown("##### Operation mode")
        mode = st.radio(
            "Operation mode",
            options=["Standalone (direct)", "Connected to API"],
            index=1 if ss.use_api_mode else 0,
            label_visibility="collapsed",
            help="Standalone runs an in-process perceptor (lowest latency). API "
            "mode calls the FastAPI service over HTTP.",
        )
        ss.use_api_mode = mode == "Connected to API"
        if ss.use_api_mode:
            ss.api_base_url = st.text_input(
                "API base URL", value=ss.api_base_url,
                help="Start it with `uvicorn app.main:app --reload`.",
            )

        # ---- Model selector ----
        st.markdown("##### Model")
        entries = infra.available_model_entries()
        avail = {e["name"]: e for e in entries}
        names = [e["name"] for e in entries]
        default_idx = names.index(ss.selected_model) if ss.selected_model in names else 0

        def _fmt(name: str) -> str:
            ok = avail.get(name, {}).get("available", False)
            mark = "\u2713" if ok else "\u25cb"
            return f"{mark}  {name}"

        chosen = st.selectbox(
            "Model", options=names, index=default_idx, format_func=_fmt,
            label_visibility="collapsed",
            help="\u2713 = artifact available on disk. \u25cb = missing (train/export first).",
        )
        ss.selected_model = chosen
        entry = avail.get(chosen, {})
        if not entry.get("available", False):
            st.warning("Artifact missing for this model - inference will error.", icon="\u26a0\ufe0f")
        else:
            st.caption(f"{entry.get('kind','?')} \u00b7 {entry.get('description','')}")

        with st.expander("Custom / advanced model"):
            custom = st.text_input("Custom model name", value="", placeholder="e.g. fusion_noisy")
            if custom.strip():
                ss.selected_model = custom.strip()
            ss.chunk_s = st.slider(
                "Analysis chunk (s)", 0.25, 1.0, float(ss.chunk_s), 0.05,
                help="Longer chunks = more spectral resolution; shorter = snappier.",
            )

        st.divider()

        # ---- Simulation parameters ----
        with st.expander("\u2699\ufe0f  Simulation parameters", expanded=True):
            p = ss.sim_params
            p["alloy"] = st.selectbox(
                "Alloy", infra.ALLOY_OPTIONS,
                index=infra.ALLOY_OPTIONS.index(p.get("alloy", "6061")),
            )
            p["blade_speed_sfpm"] = st.slider("Blade speed (SFPM)", 500.0, 1200.0,
                                              float(p["blade_speed_sfpm"]), 10.0)
            p["feed_per_tooth_mm"] = st.slider("Feed / tooth (mm)", 0.05, 0.40,
                                               float(p["feed_per_tooth_mm"]), 0.01)
            p["depth_mm"] = st.slider("Depth of cut (mm)", 5.0, 50.0, float(p["depth_mm"]), 1.0)
            p["num_teeth"] = st.slider("Number of teeth", 40, 120, int(p["num_teeth"]), 1)
            ss.manual_wear = st.slider(
                "Injected wear", 0.0, 1.0, float(ss.manual_wear), 0.01,
                help="Ground-truth blade wear driving the simulator (0 sharp \u2192 1 end-of-life).",
            )
            kin = infra.kinematics_preview(
                p["alloy"], p["blade_speed_sfpm"], p["feed_per_tooth_mm"],
                p["depth_mm"], p["num_teeth"],
            )
            st.caption(
                f"Derived: **{kin['rpm']:.0f}** RPM \u00b7 TPF **{kin['tpf_hz']:.0f} Hz** "
                f"\u00b7 MRR **{kin['mrr_mm3_s']:.0f} mm\u00b3/s**"
            )

        # ---- Thresholds ----
        with st.expander("\U0001f6a8  Alert thresholds"):
            th = ss.thresholds
            th["wear_alert"] = st.slider("Wear alert level", 0.0, 1.0, float(th["wear_alert"]), 0.05)
            th["anomaly_confidence"] = st.slider(
                "Min confidence for anomaly alert", 0.0, 1.0, float(th["anomaly_confidence"]), 0.05
            )
            th["min_confidence"] = st.slider(
                "Min confidence to display", 0.0, 1.0, float(th["min_confidence"]), 0.05
            )

        # ---- Demo scenarios ----
        with st.expander("\U0001f3ac  Demo scenarios", expanded=True):
            keys = list(SCENARIOS)
            sel = st.selectbox(
                "Scenario", keys,
                format_func=lambda k: f"{SCENARIOS[k].icon}  {SCENARIOS[k].name}",
            )
            st.caption(SCENARIOS[sel].description)
            if st.button("\u25b6  Launch scenario", type="primary", use_container_width=True):
                _launch_scenario(sel)

        st.divider()

        # ---- Utilities ----
        with st.expander("\U0001f9f0  Utilities"):
            ss.persist_logs = st.checkbox(
                "Persist live predictions to Parquet", value=ss.persist_logs,
                help="Writes to the inference log so the Historical Explorer sees live runs.",
            )
            ss.log_dir = st.text_input("Log directory", value=ss.log_dir)
            c1, c2 = st.columns(2)
            if c1.button("Clear history", use_container_width=True):
                infra.reset_run_state()
                st.toast("History cleared", icon="\U0001f9f9")
            if c2.button("Reload models", use_container_width=True):
                st.cache_data.clear()
                st.cache_resource.clear()
                st.toast("Caches cleared", icon="\U0001f504")
            st.download_button(
                "\u2b07 Export session (JSON)",
                data=_session_export_json(),
                file_name="argus_session.json",
                mime="application/json",
                use_container_width=True,
            )

        st.caption(f"perceptor v{_perceptor_version()} \u00b7 mode: {_mode_label()}")


def _launch_scenario(key: str) -> None:
    """Configure session state for a demo scenario and arm the live run."""
    ss = st.session_state
    sc = SCENARIOS[key]
    infra.reset_run_state()
    ss.sim_params = dict(sc.params)
    ss.active_scenario = sc
    ss.demo_scenario = key
    ss.delay_s = sc.delay_s
    ss.max_steps = sc.n_chunks
    ss.run_seed = sc.seed
    ss.step = 0
    ss.running = True
    st.toast(f"Launching: {sc.name}", icon=sc.icon)


def _session_export_json() -> str:
    ss = st.session_state
    payload = {
        "model": ss.selected_model,
        "mode": _mode_label(),
        "chunk_s": ss.chunk_s,
        "sim_params": ss.sim_params,
        "thresholds": ss.thresholds,
        "scenario": ss.demo_scenario,
        "history": ss.history,
        "current_result": ss.current_result,
    }
    return json.dumps(payload, indent=2, default=str)


def _perceptor_version() -> str:
    try:
        from models.streaming_perceptor import __version__

        return __version__
    except Exception:
        return "?"


# =========================================================================== #
# Tab 1: Live Monitor
# =========================================================================== #
def render_live_tab() -> None:
    st.markdown("### \U0001f4e1 Live Monitor")
    with st.expander("How to use this dashboard", expanded=False):
        st.markdown(
            "- **Launch a scenario** from the sidebar for a one-click, repeatable "
            "live run, or tune **Simulation parameters** and press **Start**.\n"
            "- Gauges show **wear**, **quality**, and **model confidence**; the banner "
            "flags threshold breaches. Plots update every chunk.\n"
            "- Switch **Standalone \u2194 API** and the **model** in the sidebar at any time.\n"
            "- Explore the other tabs for batch experiments, historical logs, the "
            "downstream production-impact model, and system/model details."
        )
    # Control bar is now OUTSIDE the fragment. Button clicks trigger a clean
    # full rerun that re-evaluates run_every without fighting the auto-refresh timer.
    _render_control_bar()

    # Dynamic run_every fragment: only the live content subtree repaints on timer.
    # When idle, run_every=None disables the timer completely (zero background cost).
    run_every = float(st.session_state.delay_s) if st.session_state.running else None
    st.fragment(_live_monitor_body, run_every=run_every)()


def _live_monitor_body() -> None:
    """Self-contained live view; advances one chunk per fragment tick."""
    ss = st.session_state

    if ss.running:
        try:
            _advance_one_step()
        except RuntimeError as exc:
            ss.running = False
            ss.live_error = str(exc)
        # NOTE: We deliberately do NOT call st.rerun() here when the run ends.
        # The outer render_live_tab() will naturally disarm the timer on next
        # evaluation because running is now False. This eliminates the full-page
        # flash that previously occurred at scenario/manual run completion.

    if ss.get("live_error"):
        st.error(ss.live_error, icon="\U0001f6d1")

    _render_status_bar()
    _render_kpi_row(ss.current_result)
    _render_live_alert(ss.current_result)
    _render_live_plots()
    _render_recommendation(ss.current_result)


def _render_control_bar() -> None:
    ss = st.session_state
    c1, c2, c3, c4, c5 = st.columns([1.6, 1, 1, 1.3, 1.6])
    if c1.button(
        "\u23f9  Stop" if ss.running else "\u25b6  Start Live Simulation",
        type="primary", use_container_width=True, key="btn_startstop",
    ):
        # Toggle run state, then a full rerun so the live tab recomputes
        # ``run_every`` (arming/disarming the fragment refresh timer).
        if ss.running:
            ss.running = False
        else:
            infra.reset_run_state()
            ss.active_scenario = None
            ss.demo_scenario = None
            ss.max_steps = 24
            ss.delay_s = 0.10
            ss.running = True
        st.rerun()

    if c2.button("\u23ed  Step", use_container_width=True, key="btn_step",
                 help="Process a single chunk"):
        ss.running = False
        ss.live_error = ""
        try:
            _advance_one_step(single=True)
        except RuntimeError as exc:
            ss.live_error = str(exc)

    if c3.button("\u2795  Wear", use_container_width=True, key="btn_wear",
                 help="Inject a step increase in blade wear"):
        ss.manual_wear = float(min(1.0, ss.manual_wear + 0.1))
        st.toast(f"Injected wear \u2192 {ss.manual_wear:.2f}", icon="\U0001f4c8")

    models_list = infra.known_model_names()
    if "quick_model" not in st.session_state:
        st.session_state.quick_model = (
            ss.selected_model if ss.selected_model in models_list else models_list[0]
        )
    quick = c4.selectbox("Model", options=models_list, label_visibility="collapsed",
                         key="quick_model")
    # Change-detection avoids ping-pong with the sidebar model selector. The
    # in-fragment selectbox change already reruns the fragment, so no explicit
    # rerun is needed - downstream rendering just picks up the new model.
    if quick != ss.get("_quick_prev", quick):
        ss.selected_model = quick
    ss._quick_prev = quick

    prog = ss.step / max(1, ss.max_steps)
    c5.progress(min(1.0, prog), text=f"chunk {ss.step}/{ss.max_steps}")


def _advance_one_step(single: bool = False) -> None:
    """Generate one chunk, run inference, and update all live state."""
    ss = st.session_state
    sc = ss.active_scenario

    if sc is not None:
        if ss.step >= sc.n_chunks:
            ss.running = False
            return
        wear = sc.wear_for_step(ss.step)
        params = dict(sc.params)
        seed = sc.seed + ss.step
    else:
        if not single and ss.step >= ss.max_steps:
            ss.running = False
            return
        wear = float(ss.manual_wear)
        params = dict(ss.sim_params)
        seed = int(ss.run_seed) + ss.step

    chunk = infra.generate_chunk(params, wear, seed, float(ss.chunk_s), with_thermal=True)
    accel = chunk["accel"]
    noise_sd = float(getattr(sc, "noise_sd", 0.0) if sc is not None else 0.0)
    if noise_sd > 0:
        rng = np.random.default_rng(seed + 777)
        rms = float(np.sqrt(np.mean(accel**2))) or 1.0
        accel = accel + rng.normal(0.0, noise_sd * rms, size=accel.shape)
    result = infra.run_inference(
        accel, chunk["meta"],
        model=ss.selected_model, use_api=ss.use_api_mode,
        api_base_url=ss.api_base_url, return_features=False, wear_injected=wear,
    )
    ss.live_error = ""
    ss.current_result = result
    ss.last_waveform = (chunk["t"], accel)
    ss.last_meta = chunk["meta"]
    ss.last_latency_ms = float(result.get("latency_ms", 0.0))
    ss.last_stft = infra.compute_stft_for_display(
        accel, float(chunk["meta"].get("fs_hz", 40960.0))
    )
    infra.push_history(infra.summarize_result(
        result, injected_wear=wear, mean_temp_c=chunk["mean_temp_c"]
    ))
    ss.step += 1


def _render_status_bar() -> None:
    ss = st.session_state
    running = ss.running
    pill = status_pill_html(
        "LIVE" if running else "IDLE",
        COLORS.accent if running else COLORS.text_muted,
        live=running,
    )
    lat = f"{ss.last_latency_ms:.2f} ms" if ss.last_latency_ms is not None else "\u2014"
    model = ss.selected_model
    scen = SCENARIOS[ss.demo_scenario].name if ss.demo_scenario else "Manual"
    _md(
        f"<div class='argus-status-bar'>"
        f"<div class='sb-item'>{pill}</div>"
        f"<div class='sb-item'><span class='sb-label'>Model</span>"
        f"<span class='sb-value'>{model}</span></div>"
        f"<div class='sb-item'><span class='sb-label'>Mode</span>"
        f"<span class='sb-value'>{_mode_label()}</span></div>"
        f"<div class='sb-item'><span class='sb-label'>Scenario</span>"
        f"<span class='sb-value'>{scen}</span></div>"
        f"<div class='sb-item'><span class='sb-label'>Latency</span>"
        f"<span class='sb-value'>{lat}</span></div>"
        f"<div class='sb-item'><span class='sb-label'>Chunk</span>"
        f"<span class='sb-value'>{ss.chunk_s:.2f}s</span></div>"
        f"</div>"
    )


def _render_kpi_row(result: dict[str, Any] | None, *, key_prefix: str = "live") -> None:
    p = (result or {}).get("predictions", {})
    wear = float(p.get("wear_level", 0.0))
    quality = float(p.get("quality_score", 0.0))
    conf = float(p.get("confidence", 0.0))
    ctf = float(p.get("cycle_time_factor", 0.0))
    state = str(p.get("health_state", "\u2014")) if result else "\u2014"
    anomaly = bool(p.get("anomaly_flag", False))
    temp = st.session_state.history[-1]["mean_temp_c"] if st.session_state.history else float("nan")

    # Three gauges in a SINGLE Plotly figure with a stable uirevision: one
    # component that animates needles in place instead of three that each tear
    # down and rebuild every fragment tick (the main source of live flicker).
    col_g, col_k = st.columns([3, 1.3])
    with col_g:
        _plotly_live(plots.build_gauge_row_figure(
            [
                {"value": wear, "title": "Blade wear",
                 "thresholds": (0.45, st.session_state.thresholds["wear_alert"])},
                {"value": quality, "title": "Quality score", "thresholds": (0.5, 0.8),
                 "colors": (COLORS.critical, COLORS.warning, COLORS.accent)},
                {"value": conf, "title": "Confidence", "thresholds": (0.4, 0.7),
                 "colors": (COLORS.critical, COLORS.warning, COLORS.accent)},
            ],
            uirevision=f"argus_gauges_{key_prefix}",
        ), key=f"{key_prefix}_g_row")
    with col_k:
        color = COLORS.health_color(state)
        anomaly_sub = "\u26a0 anomaly" if anomaly else "nominal"
        _md(kpi_card_html("Health state", state.capitalize(), accent=color, sub=anomaly_sub))
        st.write("")
        _md(kpi_card_html("Cycle-time factor", f"{ctf:.3f}", accent=COLORS.info,
                          sub="1.0 = sharp-blade baseline"))
        st.write("")
        temp_str = f"{temp:.0f} \u00b0C" if temp == temp else "\u2014"
        _md(kpi_card_html("Cut-zone temp", temp_str, accent=COLORS.warning,
                          sub="IR pyrometer (thermal)"))


def _render_live_alert(result: dict[str, Any] | None) -> None:
    if not result:
        _md(alert_banner_html("info", "STANDBY", "Press Start or launch a scenario to begin."))
        return
    level, title, msg = infra.alert_level_for(result, st.session_state.thresholds)
    _md(alert_banner_html(level, title, msg))


def _render_live_plots() -> None:
    ss = st.session_state
    r1c1, r1c2 = st.columns(2)
    with r1c1:
        if ss.last_waveform is not None:
            t, accel = ss.last_waveform
            _plotly_live(
                plots.build_waveform_figure(accel, t=t, uirevision="argus_live_wave"),
                key="p_wave",
            )
        else:
            _plotly_live(
                plots.build_waveform_figure(np.array([]), uirevision="argus_live_wave"),
                key="p_wave",
            )
    with r1c2:
        if ss.last_waveform is not None:
            _, accel = ss.last_waveform
            fs = float((ss.last_meta or {}).get("fs_hz", 40960.0))
            tpf = float((ss.last_meta or {}).get("tooth_pass_freq_hz", 0.0)) or None
            _plotly_live(
                plots.build_fft_figure(accel, fs, tpf_hz=tpf, uirevision="argus_live_fft"),
                key="p_fft",
            )
        else:
            _plotly_live(
                plots.build_fft_figure(np.array([]), 40960.0, uirevision="argus_live_fft"),
                key="p_fft",
            )

    r2c1, r2c2 = st.columns(2)
    with r2c1:
        if ss.last_stft and np.asarray(ss.last_stft["power"]).size:
            s = ss.last_stft
            _plotly_live(
                plots.build_stft_heatmap(
                    s["power"], s["freqs"], s["times"], uirevision="argus_live_stft"
                ),
                key="p_stft",
            )
        else:
            _plotly_live(
                plots.build_stft_heatmap(
                    np.zeros((0, 0)), np.array([]), np.array([]), uirevision="argus_live_stft"
                ),
                key="p_stft",
            )
    with r2c2:
        df = pd.DataFrame(ss.history)
        if not df.empty and "mean_temp_c" in df:
            # Cap history for the trend plot to keep render cost constant
            df_trend = df.tail(30)
            _plotly_live(
                plots.build_trend_figure(
                    df_trend.reset_index(),
                    ["wear_level", "quality_score"],
                    x_col="index",
                    title="Recent trend (wear / quality)",
                    labels=_TREND_LABELS,
                    color_map={"wear_level": COLORS.warning, "quality_score": COLORS.accent},
                    height=300,
                    uirevision="argus_live_trend",
                ),
                key="p_trend",
            )
        else:
            _plotly_live(
                plots.build_trend_figure(
                    pd.DataFrame(), [], title="Recent trend", uirevision="argus_live_trend"
                ),
                key="p_trend",
            )


def _render_recommendation(result: dict[str, Any] | None) -> None:
    if not result:
        return
    rec = result.get("recommendations", {})
    state = str(result.get("predictions", {}).get("health_state", "healthy"))
    _md(recommendation_card_html(
        rec.get("action", "monitor"), rec.get("note", ""),
        color=COLORS.health_color(state),
        blade_change=bool(rec.get("blade_change_suggested", False)),
    ))


# =========================================================================== #
# Tab 2: Simulation Lab
# =========================================================================== #
def render_lab_tab() -> None:
    st.markdown("### \U0001f9ea Simulation Lab")
    st.caption("Run controlled single / multi-model / batch experiments on synthetic chunks.")
    ss = st.session_state

    with st.expander("Presets (from experiments)", expanded=False):
        st.caption("Load known parameter sets and expected behavior from ablation results.")
        pc1, pc2, pc3 = st.columns(3)
        for col, (key, preset) in zip((pc1, pc2, pc3), LAB_PRESETS.items()):
            with col:
                st.markdown(f"**{preset['label']}**")
                st.caption(preset["note"])
                if st.button(f"Load {preset['label']}", key=f"preset_{key}", use_container_width=True):
                    _LAB_KEY_MAP = {
                        "alloy": "lab_alloy",
                        "blade_speed_sfpm": "lab_sfpm",
                        "feed_per_tooth_mm": "lab_feed",
                        "depth_mm": "lab_depth",
                        "num_teeth": "lab_teeth",
                    }
                    for k, v in preset["params"].items():
                        st.session_state[_LAB_KEY_MAP.get(k, f"lab_{k}")] = v
                    st.session_state["lab_wear"] = float(preset["wear"])
                    st.session_state["lab_noise"] = float(preset["noise"])
                    st.session_state["lab_seed"] = int(preset["seed"])
                    st.toast(f"Loaded preset: {preset['label']}", icon="\U0001f4cb")
                    st.rerun()

    with st.form("lab_form"):
        c1, c2, c3 = st.columns(3)
        alloy = c1.selectbox("Alloy", infra.ALLOY_OPTIONS, key="lab_alloy")
        wear = c1.slider("Injected wear", 0.0, 1.0, 0.6, 0.01, key="lab_wear")
        sfpm = c2.slider("Blade speed (SFPM)", 500.0, 1200.0, 800.0, 10.0, key="lab_sfpm")
        feed = c2.slider("Feed / tooth (mm)", 0.05, 0.40, 0.12, 0.01, key="lab_feed")
        depth = c3.slider("Depth (mm)", 5.0, 50.0, 25.0, 1.0, key="lab_depth")
        teeth = c3.slider("Teeth", 40, 120, 80, 1, key="lab_teeth")
        seed = c1.number_input("Seed", 0, 10_000, 0, 1, key="lab_seed")
        noise = c2.slider("Robustness noise (\u00d7rms)", 0.0, 0.5, 0.0, 0.05, key="lab_noise")
        models_sel = st.multiselect(
            "Models to compare", options=infra.known_model_names(),
            default=[ss.selected_model], key="lab_models",
        )
        b1, b2, b3 = st.columns(3)
        run_single = b1.form_submit_button("Run single inference", use_container_width=True)
        run_compare = b2.form_submit_button("Run comparison", type="primary",
                                            use_container_width=True)
        run_batch = b3.form_submit_button("Run robustness batch (N=50)", use_container_width=True)

    params = {"alloy": alloy, "blade_speed_sfpm": sfpm, "feed_per_tooth_mm": feed,
              "depth_mm": depth, "num_teeth": teeth}

    if run_single:
        _lab_single(params, wear, int(seed), noise)
    if run_compare:
        _lab_compare(params, wear, int(seed), noise, models_sel or [ss.selected_model])
    if run_batch:
        _lab_batch(params, wear, int(seed), noise)


def _maybe_add_noise(accel: np.ndarray, noise: float, seed: int) -> np.ndarray:
    if noise <= 0:
        return accel
    rng = np.random.default_rng(seed + 999)
    rms = float(np.sqrt(np.mean(accel**2))) or 1.0
    return accel + rng.normal(0.0, noise * rms, size=accel.shape)


def _lab_single(params: dict, wear: float, seed: int, noise: float) -> None:
    ss = st.session_state
    with st.spinner("Running inference\u2026"):
        chunk = infra.generate_chunk(params, wear, seed, float(ss.chunk_s))
        accel = _maybe_add_noise(chunk["accel"], noise, seed)
        try:
            result = infra.run_inference(accel, chunk["meta"], model=ss.selected_model,
                                         use_api=ss.use_api_mode, api_base_url=ss.api_base_url,
                                         return_features=True, wear_injected=wear)
        except RuntimeError as exc:
            st.error(str(exc), icon="\U0001f6d1")
            return
    _render_kpi_row(result, key_prefix="lab")
    c1, c2 = st.columns(2)
    with c1:
        _plotly(plots.build_waveform_figure(accel, t=chunk["t"],
                                            title="Chunk waveform"), key="lab_wave")
        fs = float(chunk["meta"].get("fs_hz", 40960.0))
        tpf = float(chunk["meta"].get("tooth_pass_freq_hz", 0.0)) or None
        _plotly(plots.build_fft_figure(accel, fs, tpf_hz=tpf, title="Chunk FFT"), key="lab_fft")
    with c2:
        stft = infra.compute_stft_for_display(accel, float(chunk["meta"].get("fs_hz", 40960.0)))
        _plotly(plots.build_stft_heatmap(stft["power"], stft["freqs"], stft["times"]),
                key="lab_stft")
        _plotly(plots.build_health_prob_bar(result["predictions"]["health_probs"]),
                key="lab_probs")

    feats = result.get("features", {})
    if feats:
        with st.expander("DSP features (this chunk)"):
            fdf = pd.DataFrame(sorted(feats.items()), columns=["feature", "value"])
            st.dataframe(fdf, use_container_width=True, height=280, hide_index=True)
    st.download_button(
        "\u2b07 Download result JSON", data=json.dumps(result, indent=2, default=str),
        file_name="argus_inference.json", mime="application/json",
    )


def _lab_compare(params: dict, wear: float, seed: int, noise: float, models: list[str]) -> None:
    ss = st.session_state
    chunk = infra.generate_chunk(params, wear, seed, float(ss.chunk_s))
    accel = _maybe_add_noise(chunk["accel"], noise, seed)
    rows: list[dict[str, Any]] = []
    with st.spinner(f"Comparing {len(models)} models\u2026"):
        for m in models:
            try:
                r = infra.run_inference(accel, chunk["meta"], model=m,
                                        use_api=ss.use_api_mode, api_base_url=ss.api_base_url,
                                        wear_injected=wear)
            except RuntimeError as exc:
                st.warning(f"{m}: {exc}", icon="\u26a0\ufe0f")
                continue
            pr = r["predictions"]
            rows.append({
                "model": m, "wear_level": pr["wear_level"],
                "abs_err_vs_injected": abs(pr["wear_level"] - wear),
                "cycle_time_factor": pr["cycle_time_factor"], "quality_score": pr["quality_score"],
                "health_state": pr["health_state"], "confidence": pr["confidence"],
                "latency_ms": r["latency_ms"],
            })
    if not rows:
        return
    df = pd.DataFrame(rows)
    st.caption(f"Injected wear = **{wear:.2f}** \u00b7 noise \u00d7{noise:.2f}rms")
    c1, c2 = st.columns([1.2, 1])
    with c1:
        st.dataframe(df, use_container_width=True, hide_index=True,
                     column_config={
                         "wear_level": st.column_config.NumberColumn(format="%.3f"),
                         "abs_err_vs_injected": st.column_config.NumberColumn(format="%.3f"),
                         "quality_score": st.column_config.NumberColumn(format="%.3f"),
                         "confidence": st.column_config.ProgressColumn(min_value=0, max_value=1),
                         "latency_ms": st.column_config.NumberColumn(format="%.2f ms"),
                     })
    with c2:
        _plotly(plots.build_grouped_bar(
            df["model"].tolist(),
            {"wear_level": df["wear_level"].tolist(), "quality_score": df["quality_score"].tolist()},
            title="Wear / quality by model", y_title="value", height=320,
        ), key="cmp_bar")
    st.download_button("\u2b07 Download comparison CSV", data=df.to_csv(index=False),
                       file_name="argus_model_comparison.csv", mime="text/csv")


def _lab_batch(params: dict, wear: float, seed: int, noise: float, n: int = 50) -> None:
    ss = st.session_state
    rows: list[dict[str, Any]] = []
    progress = st.progress(0.0, text="Running robustness batch\u2026")
    for i in range(n):
        chunk = infra.generate_chunk(params, wear, seed + i, float(ss.chunk_s))
        accel = _maybe_add_noise(chunk["accel"], noise, seed + i)
        try:
            r = infra.run_inference(accel, chunk["meta"], model=ss.selected_model,
                                    use_api=ss.use_api_mode, api_base_url=ss.api_base_url,
                                    wear_injected=wear)
        except RuntimeError as exc:
            st.error(str(exc), icon="\U0001f6d1")
            progress.empty()
            return
        pr = r["predictions"]
        rows.append({"wear_level": pr["wear_level"], "quality_score": pr["quality_score"],
                     "cycle_time_factor": pr["cycle_time_factor"], "confidence": pr["confidence"],
                     "health_state": pr["health_state"], "latency_ms": r["latency_ms"]})
        progress.progress((i + 1) / n, text=f"Chunk {i+1}/{n}")
    progress.empty()
    df = pd.DataFrame(rows)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Wear mean", f"{df['wear_level'].mean():.3f}", f"\u03c3 {df['wear_level'].std():.3f}")
    m2.metric("Wear MAE vs injected", f"{(df['wear_level'] - wear).abs().mean():.3f}")
    m3.metric("Quality mean", f"{df['quality_score'].mean():.3f}")
    m4.metric("Latency p50", f"{df['latency_ms'].median():.2f} ms")

    c1, c2 = st.columns(2)
    with c1:
        _plotly(plots.build_histogram(df["wear_level"], title="Wear prediction distribution",
                                      x_title="wear_level", color=COLORS.warning), key="batch_hist")
    with c2:
        counts = df["health_state"].value_counts()
        _plotly(plots.build_pie(counts.index.tolist(), counts.values.tolist(),
                                title="Health-state distribution",
                                color_map=COLORS.health), key="batch_pie")
    st.download_button("\u2b07 Download batch CSV", data=df.to_csv(index=False),
                       file_name="argus_robustness_batch.csv", mime="text/csv")


# =========================================================================== #
# Tab 3: Historical Explorer
# =========================================================================== #
def render_history_tab() -> None:
    st.markdown("### \U0001f5c4\ufe0f Historical Explorer")
    st.caption("Query and filter the partitioned Parquet inference logs.")
    ss = st.session_state

    c1, c2 = st.columns([1, 3])
    if c1.button("\U0001f504 Load / Refresh logs", type="primary"):
        infra.load_logs_cached.clear()
    c2.caption(f"Reading from `{ss.log_dir}` \u00b7 generate data via "
               "`python scripts/stream_demo.py` or enable live-run persistence in Utilities.")

    try:
        df = infra.load_logs_cached(ss.log_dir)
    except RuntimeError as exc:
        st.error(str(exc), icon="\U0001f6d1")
        return
    if df is None or len(df) == 0:
        st.info("No inference logs found yet. Run `python scripts/stream_demo.py "
                "--model 1dcnn_normnone --duration-s 5 --wear 0.6` once, then Refresh.",
                icon="\U0001f4a1")
        return

    df = df.copy()
    if "timestamp" in df:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
        df = df.sort_values("timestamp")

    # ---- Filters ----
    with st.expander("Filters", expanded=True):
        f1, f2, f3 = st.columns(3)
        models = sorted(df["model"].dropna().unique()) if "model" in df else []
        sel_models = f1.multiselect("Model", models, default=models)
        states = sorted(df["pred_health_state"].dropna().unique()) if "pred_health_state" in df else []
        sel_states = f2.multiselect("Health state", states, default=states)
        recent_n = f3.slider("Show most recent N", 20, min(2000, max(20, len(df))),
                             min(300, len(df)), 10)
        wmin, wmax = f1.slider("Wear range", 0.0, 1.0, (0.0, 1.0), 0.05)
        search = f2.text_input("Search action contains", "")

    mask = pd.Series(True, index=df.index)
    if sel_models and "model" in df:
        mask &= df["model"].isin(sel_models)
    if sel_states and "pred_health_state" in df:
        mask &= df["pred_health_state"].isin(sel_states)
    if "pred_wear_level" in df:
        mask &= df["pred_wear_level"].between(wmin, wmax)
    if search and "action" in df:
        mask &= df["action"].fillna("").str.contains(search, case=False)
    fdf = df[mask].tail(recent_n)

    if fdf.empty:
        st.warning("No records match the current filters.", icon="\U0001f50d")
        return

    # ---- Summary row ----
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Records", f"{len(fdf):,}")
    if "pred_anomaly_flag" in fdf:
        s2.metric("Anomaly rate", f"{fdf['pred_anomaly_flag'].mean()*100:.1f}%")
    if "pred_wear_level" in fdf:
        s3.metric("Avg wear", f"{fdf['pred_wear_level'].mean():.3f}")
    if "latency_ms" in fdf:
        s4.metric("Latency p50", f"{fdf['latency_ms'].median():.2f} ms")

    # ---- Trend ----
    trend_cols = [c for c in ["pred_wear_level", "pred_cycle_time_factor", "pred_quality_score"]
                  if c in fdf]
    x_col = "timestamp" if "timestamp" in fdf else None
    _plotly(plots.build_trend_figure(
        fdf, trend_cols, x_col=x_col, title="Prediction trends", range_slider=True,
        labels={"pred_wear_level": "Wear", "pred_cycle_time_factor": "Cycle-time factor",
                "pred_quality_score": "Quality"},
        color_map={"pred_wear_level": COLORS.warning, "pred_cycle_time_factor": COLORS.info,
                   "pred_quality_score": COLORS.accent},
    ), key="hist_trend")

    c1, c2 = st.columns(2)
    with c1:
        if "pred_health_state" in fdf:
            counts = fdf["pred_health_state"].value_counts()
            _plotly(plots.build_pie(counts.index.tolist(), counts.values.tolist(),
                                    title="Health-state distribution", color_map=COLORS.health),
                    key="hist_pie")
    with c2:
        if "pred_wear_level" in fdf:
            colors = [COLORS.health_color(s) for s in fdf.get("pred_health_state", [])]
            xvals = fdf["timestamp"] if x_col else list(range(len(fdf)))
            _plotly(plots.build_scatter_timeline(
                xvals, fdf["pred_wear_level"].tolist(), colors,
                text=fdf.get("pred_health_state", pd.Series([""] * len(fdf))).tolist(),
                title="Wear timeline (colored by health)", y_title="wear_level",
            ), key="hist_scatter")

    st.markdown("##### Records")
    show_cols = [c for c in ["timestamp", "model", "pred_wear_level", "pred_cycle_time_factor",
                             "pred_quality_score", "pred_health_state", "pred_confidence",
                             "pred_anomaly_flag", "action", "latency_ms"] if c in fdf]
    st.dataframe(fdf[show_cols], use_container_width=True, height=340, hide_index=True)
    st.download_button("\u2b07 Download filtered CSV", data=fdf.to_csv(index=False),
                       file_name="argus_logs_filtered.csv", mime="text/csv")

    # ---- Visualize selected record (re-generate chunk; raw waveforms not stored) ----
    st.markdown("##### Visualize selected record")
    labels = [
        f"#{int(r.get('chunk_id', i))} · {r.get('model', '?')} · "
        f"wear={float(r.get('pred_wear_level', 0)):.2f} · {r.get('pred_health_state', '?')}"
        for i, r in fdf.reset_index(drop=True).iterrows()
    ]
    sel_idx = st.selectbox(
        "Record", range(len(labels)), format_func=lambda i: labels[i],
        key="hist_sel_idx",
    )
    if st.button("Visualize selected record", type="primary", key="hist_viz_btn"):
        row = fdf.reset_index(drop=True).iloc[int(sel_idx)]
        wear = float(row["wear_injected"]) if "wear_injected" in row and pd.notna(row["wear_injected"]) else 0.5
        params = infra.params_from_log_row(row)
        model = str(row.get("model", ss.selected_model))
        seed = int(row.get("chunk_id", 0) or 0)
        chunk = infra.generate_chunk(params, wear, seed, float(ss.chunk_s))
        try:
            result = infra.run_inference(
                chunk["accel"], chunk["meta"], model=model,
                use_api=ss.use_api_mode, api_base_url=ss.api_base_url,
                return_features=True, wear_injected=wear,
            )
        except RuntimeError as exc:
            st.error(str(exc), icon="\U0001f6d1")
            return
        with st.expander("Selected record — waveform / FFT / STFT", expanded=True):
            st.caption(
                f"Re-generated from logged operating point (wear_injected={wear:.2f}). "
                "Raw waveforms are not stored in Parquet logs."
            )
            c1, c2 = st.columns(2)
            with c1:
                _plotly(plots.build_waveform_figure(chunk["accel"], t=chunk["t"],
                                                    title="Reconstructed waveform"), key="hv_wave")
                fs = float(chunk["meta"].get("fs_hz", 40960.0))
                tpf = float(chunk["meta"].get("tooth_pass_freq_hz", 0.0)) or None
                _plotly(plots.build_fft_figure(chunk["accel"], fs, tpf_hz=tpf,
                                                title="Reconstructed FFT"), key="hv_fft")
            with c2:
                stft = infra.compute_stft_for_display(chunk["accel"], fs)
                _plotly(plots.build_stft_heatmap(stft["power"], stft["freqs"], stft["times"],
                                                 title="Reconstructed STFT"), key="hv_stft")
                _plotly(plots.build_health_prob_bar(result["predictions"]["health_probs"],
                                                    title="Re-inferred health probs"), key="hv_probs")
            st.json(result, expanded=False)


# =========================================================================== #
# Tab 4: Optimization Sandbox
# =========================================================================== #
def render_optimization_tab() -> None:
    st.markdown("### \U0001f3ed Optimization Sandbox")
    st.caption(
        "Transparent demonstration: accurate perception (wear / cycle-time / quality) "
        "feeds a downstream production planning, costing & nesting optimizer. Every "
        "formula below is plain, auditable Python."
    )
    ss = st.session_state

    src = st.radio("Input source", ["Use latest live result", "Manual override", "Paste JSON payload"],
                   horizontal=True)
    state = opt.PerceptionState()

    if src == "Use latest live result" and ss.current_result:
        p = ss.current_result["predictions"]
        state = opt.PerceptionState(
            wear_level=p["wear_level"], cycle_time_factor=p["cycle_time_factor"],
            quality_score=p["quality_score"], health_state=p["health_state"],
            confidence=p["confidence"], anomaly_flag=p["anomaly_flag"],
        )
    elif src == "Manual override":
        c1, c2, c3 = st.columns(3)
        state.wear_level = c1.slider("Wear level", 0.0, 1.0, 0.6, 0.01)
        state.cycle_time_factor = c2.slider("Cycle-time factor", 1.0, 2.0, 1.25, 0.01)
        state.quality_score = c3.slider("Quality score", 0.0, 1.0, 0.75, 0.01)
        state.confidence = c1.slider("Confidence", 0.0, 1.0, 0.8, 0.01)
        state.health_state = c2.selectbox("Health state", _HEALTH_ORDER, index=2)
        state.anomaly_flag = c3.checkbox("Anomaly flag", value=False)
    elif src == "Paste JSON payload":
        txt = st.text_area("Paste an InferenceResponse (or its predictions block)", height=140)
        if txt.strip():
            try:
                data = json.loads(txt)
                p = data.get("predictions", data)
                state = opt.PerceptionState(
                    wear_level=float(p["wear_level"]),
                    cycle_time_factor=float(p["cycle_time_factor"]),
                    quality_score=float(p["quality_score"]),
                    health_state=str(p.get("health_state", "warning")),
                    confidence=float(p.get("confidence", 1.0)),
                    anomaly_flag=bool(p.get("anomaly_flag", False)),
                )
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                st.error(f"Invalid payload: {exc}", icon="\U0001f6d1")
    if src == "Use latest live result" and not ss.current_result:
        st.info("No live result yet - run the Live Monitor first, or pick another input source.",
                icon="\U0001f4a1")

    with st.expander("Shop-floor assumptions (editable, transparent)"):
        a1, a2, a3 = st.columns(3)
        inp = opt.ProductionInputs(
            baseline_cycle_time_s=a1.number_input("Baseline cycle time (s)", 1.0, 600.0, 45.0, 1.0),
            blade_life_cycles=a1.number_input("Blade life (cuts)", 50.0, 5000.0, 1000.0, 50.0),
            machine_rate_usd_per_hr=a2.number_input("Machine rate ($/hr)", 10.0, 500.0, 90.0, 5.0),
            material_cost_usd_per_part=a2.number_input("Material $/part", 0.0, 500.0, 12.0, 1.0),
            blade_cost_usd=a3.number_input("Blade cost ($)", 10.0, 2000.0, 220.0, 10.0),
            scrap_disposal_usd_per_part=a3.number_input("Scrap $/part", 0.0, 100.0, 3.0, 0.5),
            shift_hours=a1.number_input("Shift hours", 1.0, 24.0, 8.0, 0.5),
        )

    impact = opt.compute_production_impact(state, inp)
    cur, ref, delta = impact["current"], impact["reference"], impact["delta"]

    m = impact["maintenance"]
    _md(alert_banner_html(m["level"], f"MAINTENANCE: {m['urgency'].replace('_', ' ').upper()}",
                          m["message"]))

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        _md(kpi_card_html("Efficiency score", f"{cur['efficiency_score']:.0f}",
                          accent=COLORS.accent, sub=f"\u0394 {delta['efficiency_score']:+.0f} vs sharp"))
    with k2:
        _md(kpi_card_html("Good parts / hr", f"{cur['good_pph']:.1f}",
                          accent=COLORS.info, sub=f"\u0394 {delta['good_pph']:+.1f}"))
    with k3:
        _md(kpi_card_html("Cost / good part", f"${cur['cost_per_good_part']:.2f}",
                          accent=COLORS.warning, sub=f"\u0394 ${delta['cost_per_good_part']:+.2f}"))
    with k4:
        _md(kpi_card_html("Blade life left", f"{cur['remaining_blade_life_cycles']:.0f}",
                          accent=COLORS.purple, sub="cuts remaining"))

    c1, c2 = st.columns(2)
    with c1:
        _plotly(plots.build_grouped_bar(
            ["Cycle time (s)", "Good parts/hr", "Yield %"],
            {"Sharp blade": [ref["cycle_time_s"], ref["good_pph"], ref["yield_pct"]],
             "Current": [cur["cycle_time_s"], cur["good_pph"], cur["yield_pct"]]},
            title="Sharp-blade baseline vs current", colors=[COLORS.neutral, COLORS.accent_soft],
            y_title="value",
        ), key="opt_bar")
    with c2:
        _plotly(plots.build_gauge_figure(
            cur["efficiency_score"], "Production efficiency", min_val=0, max_val=100,
            thresholds=(60, 85), colors=(COLORS.critical, COLORS.warning, COLORS.accent),
            number_suffix="", number_format=".0f", reference=ref["efficiency_score"], height=300,
        ), key="opt_gauge")

    st.markdown("##### Downstream payload (clean contract)")
    payload = opt.downstream_payload(state, inp, impact, model=ss.selected_model)
    st.code(json.dumps(payload, indent=2), language="json")
    st.download_button("\u2b07 Download payload JSON", data=json.dumps(payload, indent=2),
                       file_name="argus_production_plan.json", mime="application/json")


# =========================================================================== #
# Tab 5: System Health & Models
# =========================================================================== #
def render_system_tab() -> None:
    st.markdown("### \U0001f6e0\ufe0f System Health & Models")
    ss = st.session_state

    if ss.use_api_mode:
        st.markdown("##### API health")
        try:
            h = infra.api_health(ss.api_base_url)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Status", h.get("status", "?"))
            c2.metric("Default model", h.get("default_model", "?"))
            c3.metric("Available models", h.get("n_available_models", 0))
            c4.metric("Logged", h.get("n_logged", 0))
        except RuntimeError as exc:
            st.error(str(exc), icon="\U0001f6d1")
    else:
        st.markdown("##### Standalone perceptor")
        c1, c2, c3 = st.columns(3)
        c1.metric("Perceptor version", _perceptor_version())
        c2.metric("Active model", ss.selected_model)
        c3.metric("Chunk size", f"{ss.chunk_s:.2f} s")

    # ---- Model inventory ----
    st.markdown("##### Model inventory")
    entries = infra.available_model_entries()
    rows = []
    for e in entries:
        summ = metrics.model_metric_summary(e["name"])
        rows.append({
            "model": e["name"], "kind": e["kind"], "available": e["available"],
            "wear_MAE": summ["wear_mae"], "health_F1": summ["health_f1"],
            "latency_ms": summ["latency_ms"], "params": summ["n_params"],
            "notes": e["description"],
        })
    idf = pd.DataFrame(rows)
    st.dataframe(idf, use_container_width=True, hide_index=True, height=360,
                 column_config={
                     "available": st.column_config.CheckboxColumn("avail"),
                     "wear_MAE": st.column_config.NumberColumn(format="%.3f"),
                     "health_F1": st.column_config.NumberColumn(format="%.3f"),
                     "latency_ms": st.column_config.NumberColumn(format="%.2f ms"),
                 })

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("##### Edge latency (ONNX CPU, p50)")
        lat_rows = []
        for name in infra.known_model_names():
            lm = metrics.typical_latency_ms(name)
            if lm is not None:
                lat_rows.append({"model": name, "p50_ms": lm})
        if lat_rows:
            ldf = pd.DataFrame(lat_rows)
            _plotly(plots.build_grouped_bar(
                ldf["model"].tolist(), {"p50 ms": ldf["p50_ms"].tolist()},
                title="Single-chunk latency", y_title="ms", text_format=".2f", height=300,
            ), key="sys_lat")
    with c2:
        st.markdown("##### Quick tester")
        with st.form("sys_test"):
            tw = st.slider("Injected wear", 0.0, 1.0, 0.5, 0.01, key="sys_wear")
            go = st.form_submit_button("Run one inference", type="primary", use_container_width=True)
        if go:
            chunk = infra.generate_chunk(ss.sim_params, tw, 0, float(ss.chunk_s))
            try:
                r = infra.run_inference(chunk["accel"], chunk["meta"], model=ss.selected_model,
                                        use_api=ss.use_api_mode, api_base_url=ss.api_base_url,
                                        wear_injected=tw)
                st.json(r, expanded=False)
            except RuntimeError as exc:
                st.error(str(exc), icon="\U0001f6d1")

    # ---- Robustness notes ----
    with st.expander("Robustness & edge findings"):
        st.markdown(metrics.robustness_notes())
        bench = metrics.load_benchmark()
        xgb = bench.get("xgboost", {})
        if xgb:
            st.markdown(
                f"- **XGBoost** end-to-end p50: **{xgb.get('end_to_end_p50_ms', 0):.2f} ms** "
                f"(DSP extract dominates at ~{xgb.get('dsp_extract', {}).get('p50_ms', 0):.1f} ms).\n"
                f"- **DL variants** run torch-free on ONNX Runtime at ~0.3-0.4 ms/chunk (p50)."
            )

    # ---- Config inspector ----
    with st.expander("Config inspector (effective settings)"):
        st.json({
            "mode": _mode_label(), "model": ss.selected_model, "chunk_s": ss.chunk_s,
            "sim_params": ss.sim_params, "thresholds": ss.thresholds,
            "persist_logs": ss.persist_logs, "log_dir": ss.log_dir,
            "api_base_url": ss.api_base_url,
        }, expanded=False)


# =========================================================================== #
# Main
# =========================================================================== #
def main() -> None:
    _md(build_css())
    infra.init_session_state()

    _md(
        f"<div style='display:flex;justify-content:space-between;align-items:center;"
        f"margin-bottom:6px'><div><span style='font-size:1.5rem;font-weight:800'>"
        f"Argus Panoptes</span> <span style='color:{COLORS.text_muted}'>\u2014 "
        f"Industrial Perception Dashboard</span></div>"
        f"<div class='argus-caption'>Vibration \u00b7 Thermal \u2192 wear / health / quality "
        f"\u2192 production planning</div></div>"
    )

    render_sidebar()

    tabs = st.tabs([
        "\U0001f4e1  Live Monitor", "\U0001f9ea  Simulation Lab",
        "\U0001f5c4\ufe0f  Historical Explorer", "\U0001f3ed  Optimization Sandbox",
        "\U0001f6e0\ufe0f  System & Models",
    ])
    with tabs[0]:
        render_live_tab()
    with tabs[1]:
        render_lab_tab()
    with tabs[2]:
        render_history_tab()
    with tabs[3]:
        render_optimization_tab()
    with tabs[4]:
        render_system_tab()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # global graceful degradation
        st.error(f"Dashboard error: {exc}", icon="\U0001f6d1")
        with st.expander("Traceback"):
            import traceback

            st.code(traceback.format_exc())
