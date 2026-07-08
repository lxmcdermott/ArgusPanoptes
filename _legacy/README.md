# `_legacy/` — archived Streamlit dashboard

This folder holds the **original Streamlit operator dashboard** (Step 5, v1). It
has been **superseded** by the high-performance **NiceGUI + Plotly** dashboard at
`app/nicegui_dashboard.py` and is kept only for reference / comparison.

## Why it was replaced

The Streamlit UI worked, but could not deliver a smooth, production-grade *live*
multi-stream experience. Even after `st.fragment` + dynamic `run_every` + a
hardened dark theme, the live view (waveform + FFT + STFT heatmap + thermal +
several gauges/KPIs refreshing together) still suffered from:

- **Full-subtree repaints** on every rerun → flicker on complex updates.
- **JSON figure serialization** cost per rerun (the STFT heatmap especially).
- **No true background data production** + partial UI updates.

The NiceGUI rewrite fixes these at the architecture level: a background
`SimulationOrchestrator` thread produces frames while WebSocket-backed reactive
updates and a single `ui.timer` push only the changed figures/labels in place
(with aggressive waveform/FFT downsampling and a throttled STFT). Result:
stable, flicker-free ~5–10 Hz live updates. See
`logs/2026-07-07_argus-v1-nicegui-dashboard.md` for the full migration write-up.

## What's here

| File | Role |
| --- | --- |
| `dashboard.py` | The Streamlit operator app (5 tabs). |
| `infra.py` | Streamlit-coupled session-state + cached resources + unified `run_inference` (the ancestor of `dashviz/orchestrator.py`). |
| `_smoke_dashboard.py` | Import/build smoke check. |
| `_apptest_dashboard.py`, `_apptest_step.py` | Streamlit `AppTest` harnesses. |

The reusable, framework-neutral helpers (`dashviz/theme.py`, `plots.py`,
`optimization.py`, `scenarios.py`, `metrics.py`) were **kept in `dashviz/`** —
the new NiceGUI app reuses them directly.

## Running the legacy dashboard (if you must)

`infra.py` moved out of the `dashviz` package, so run from *inside* this folder
(Streamlit puts the script's directory on `sys.path`, which resolves
`import infra`):

```bash
pip install -e ".[ml,dl,app,dashboard]"   # the legacy Streamlit extra
cd _legacy
streamlit run dashboard.py                 # http://localhost:8501
```

The `_apptest_*.py` harnesses likewise expect to be run from within `_legacy/`.

> Prefer the current dashboard: `pip install -e ".[ml,dl,app,dashboard-nicegui]"`
> then `python -m app.nicegui_dashboard` (see the root `README.md`).
