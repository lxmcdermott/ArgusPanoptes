# Argus Panoptes — Session Change Log

**Date:** 2026-07-07
**Scope:** Day-5 ops layer, **performance overhaul** — replacing the Streamlit +
Plotly operator dashboard with a high-performance **NiceGUI + Plotly** dashboard
(`app/nicegui_dashboard.py`) backed by a decoupled background
`SimulationOrchestrator` (`dashviz/orchestrator.py`). Goal: eliminate the live
flicker / lag / slow-load problems of the Streamlit UI while preserving 100% of
the perception, ML, data-pipeline, and integration value.
**No changes** to `sensors/`, `dsp/`, `models/`, or `app/main.py`/`app/logging.py`
runtime code — the dashboard only *consumes* the existing `StreamingPerceptor`,
`InferenceLogger`, FastAPI service, simulators, and experiment artifacts. All 86
tests still pass; the Parquet schema and trained models are untouched.
**Environment:** Python 3.13.9 (Windows / PowerShell). **nicegui 2.x**, plotly,
httpx (the `dashboard-nicegui` extra).

---

## 1. Why the rewrite

Even after the previous session's `st.fragment` + dynamic `run_every` + dark-theme
+ CSS hardening, the Streamlit live view (waveform + FFT + STFT heatmap + thermal +
several gauges/KPIs updating together) still suffered from lag, occasional flicker
on complex updates, and slow initial loads. Root causes were architectural:

1. **Full-subtree repaints** on every rerun (whole fragment re-executes).
2. **JSON figure (re)serialization** each rerun — the STFT heatmap dominates.
3. **No true background data production** with *partial* in-place UI updates.

NiceGUI addresses all three: WebSocket-backed reactive updates, a background
thread producing frames, and per-element in-place updates driven by one
`ui.timer` — only the changed figures/labels are pushed.

## 2. Architecture

```
                 ┌───────────────────────────────────────────────┐
   controls ───▶ │  SimulationOrchestrator  (dashviz/orchestrator)│
 (drawer/live)   │  background thread: simulate → DSP → infer      │
                 │  thread-safe Snapshot (RLock) + live params     │
                 └───────────────┬───────────────────────────────┘
                                 │ snapshot() (immutable copy)
                 ┌───────────────▼───────────────────────────────┐
   ui.timer ───▶ │  Dashboard (app/nicegui_dashboard.py)          │
   (0.1 s)       │  reads snapshot; if generation changed, pushes │
                 │  only changed Plotly figures / HTML / labels   │
                 └───────────────────────────────────────────────┘
```

- **`dashviz/orchestrator.py`** (~760 lines, UI-agnostic): cached, thread-safe
  resources (perceptor per model/chunk, optional httpx client, simulators, DSP
  `SignalProcessor`); `SessionConfig` (mutable) + `Snapshot` (immutable); the
  unified `run_inference` (direct **and** API modes, identical payload); ported
  `generate_chunk`, `compute_stft_for_display`, `alert_level_for`,
  `params_from_log_row`, `load_logs`, `kinematics_preview`; background `_loop`
  with `start`/`start_manual`/`stop`/`toggle_pause`/`step_once`/`shutdown`,
  waking via a `threading.Event`. Display caps `LIVE_WAVE_POINTS` / `LIVE_FFT_POINTS`
  keep serialization cheap.
- **`app/nicegui_dashboard.py`** (~1220 lines): one `Dashboard` per client
  (`@ui.page('/')`), its own orchestrator, cleaned up on `client.on_disconnect`.
  Left drawer = global controls (mode, model, live sim params, thresholds,
  utilities). Five tabs: **Live Monitor**, **Simulation Lab**, **Historical
  Explorer**, **Optimization Sandbox**, **System & Models**. Dark industrial
  theme reproduced as NiceGUI/Quasar CSS + reused `dashviz.theme` HTML helpers.

### Smoothness strategy
- One `ui.timer(0.1)`; the callback is a **cheap no-op** unless `snapshot.generation`
  advanced (generation guard) — no wasted repaints when idle/paused.
- Waveform/FFT downsampled to ≤~1–2k points; **STFT throttled** (updated every 3rd
  live tick + capped freq/time bins) since it is the heaviest to serialize.
- In-place figure swaps (`el.figure = ...; el.update()`) with stable `uirevision`,
  and live charts carry a `config` with the mode bar hidden (nothing to repaint on
  hover). Gauges are a single indicator figure animated in place.

## 3. Files created / modified / archived

**Created**
- `dashviz/orchestrator.py` — `SimulationOrchestrator`, `SessionConfig`, `Snapshot`,
  cached resources, unified inference, DSP/STFT/log helpers.
- `app/nicegui_dashboard.py` — the NiceGUI + Plotly dashboard (entry point).
- `_legacy/README.md` — why the Streamlit UI was superseded + how to run it.
- `logs/2026-07-07_argus-v1-nicegui-dashboard.md` — this log.

**Modified**
- `pyproject.toml`, `requirements.txt` — new `dashboard-nicegui` extra
  (`nicegui>=2.0`, `plotly>=5.22`, `httpx>=0.27`); `dashboard` (Streamlit) marked legacy.
- `dashviz/__init__.py`, `dashviz/metrics.py` — docstrings updated (orchestrator
  replaces the Streamlit-coupled `infra`).
- `deployment/Dockerfile` — installs `.[ml,dl,app,dashboard-nicegui]`; default CMD =
  API; exposes 8000/8080.
- `deployment/docker-compose.yml` — real `api` + `dashboard` (+ `datagen`) services;
  dashboard runs `python -m app.nicegui_dashboard` in API mode against `api`.
- `README.md`, `app/README.md` — new install/run instructions, tab overview, env
  overrides, legacy note.

**Archived (via `git mv` → `_legacy/`)**
- `dashboard.py`, `dashviz/infra.py`, `_smoke_dashboard.py`,
  `_apptest_dashboard.py`, `_apptest_step.py`. The reusable helpers
  (`theme`, `plots`, `optimization`, `scenarios`, `metrics`) stayed in `dashviz/`
  and are reused directly by the new app.

## 4. Verification

- **Perception regression gate:** `pytest -q` → **86 passed** (unchanged), before
  and after the migration.
- **Orchestrator (headless):** ran a scenario + manual steps + model switching;
  consistent structured output matching the `infer_chunk` contract (~20 ms latency).
- **Page build:** every one of the 5 tabs builds; `GET /` returns 200.
- **Reactive pipeline (headless, NiceGUI user simulation, no browser):** opened `/`,
  launched a scenario, and confirmed the background thread + `ui.timer` actually
  mutate the live DOM — the Start control flips to **Stop** and the status shows the
  **LIVE** pill (proves per-tick in-place updates run without error).
- **API mode parity:** with `uvicorn app.main:app` running, standalone vs
  `use_api=True` inference on an identical chunk were bit-for-bit equal
  (wear/health/quality/confidence).

## 5. How to run

```bash
pip install -e ".[ml,dl,app,dashboard-nicegui]"
python -m app.nicegui_dashboard            # http://127.0.0.1:8080
# API mode: uvicorn app.main:app --reload  (then toggle "Connected to API" in the drawer)
# Docker:   docker compose -f deployment/docker-compose.yml up api dashboard
```

## 6. Remaining TODOs / notes

- Add captured screenshots/GIFs (`docs/dashboard-live.png`, …) to the READMEs.
- Optional: a NiceGUI `nicegui.testing` pytest test mirroring the headless check
  (needs a `pytest-asyncio`/`user` fixture dev dependency).
- Multi-client note: each browser tab gets its own orchestrator/thread (isolated);
  fine for demos. For many concurrent operators, consider a shared read-only
  producer.
- Env knobs for Docker: `ARGUS_DASHBOARD_HOST/PORT/USE_API/MODEL`,
  `ARGUS_API_BASE_URL`, `ARGUS_LOG_DIR`.
