# Argus Panoptes — Session Change Log

**Date:** 2026-07-07  
**Scope:** Day-5 ops layer — **complete performance overhaul** of the operator
dashboard. Replaced the legacy **Streamlit + Plotly** UI (`_legacy/dashboard.py`)
with a production-grade **NiceGUI + Plotly** dashboard
(`app/nicegui_dashboard.py`) backed by a decoupled background
`SimulationOrchestrator` (`dashviz/orchestrator.py`).  
**Goal:** deliver smooth, low-latency, flicker-free multi-stream live updates
(waveform / FFT / STFT / thermal trend + KPI gauges) at a stable ~5–10 Hz while
preserving 100% of perception, ML, data-pipeline, and integration value.  
**Strict non-goals honored:** no changes to `sensors/`, `dsp/`, `models/` (core),
`app/main.py`, `app/logging.py`, Parquet schema, or experiment artifacts.  
**Environment:** Python 3.13.9 (Windows / PowerShell, Anaconda base).  
**nicegui ≥2.0**, plotly, httpx (`dashboard-nicegui` extra).  
**Regression gate:** `pytest -q` → **86 passed** before and after (unchanged).

**Prior context:** the immediately preceding session
(`logs/2026-07-07_argus-v1-streamlit-dashboard.md`) completed the Streamlit
dashboard to executive-demo quality (dark theme, 4 scenarios, historical
drill-down, Lab presets, optimization sandbox) but could not eliminate live
flicker/lag due to Streamlit's rerun model. This session executes the planned
NiceGUI migration described in the technical plan.

---

## 1. Problem statement (why Streamlit was superseded)

Even after `st.fragment` + dynamic `run_every` + CSS hardening, the live
multi-stream view still suffered from:

| Symptom | Root cause |
| --- | --- |
| Visible flicker on STFT / gauge updates | Full fragment subtree repaints each tick |
| Lag under all-plots-visible load | JSON figure serialization every rerun (STFT dominates) |
| Sluggish slider response | Rerun blocks on inference + figure rebuild |
| Slow initial paint | Many widgets + figures materialized in one synchronous pass |

The fix required **architectural** separation: a background producer thread +
partial, in-place UI updates — not more Streamlit tuning.

---

## 2. Execution phases (completed in order)

### Phase 0 — Preparation
- Added **`dashboard-nicegui`** optional extra to `pyproject.toml`:
  `nicegui>=2.0`, `plotly>=5.22`, `httpx>=0.27`.
- Marked existing **`dashboard`** extra (Streamlit) as **LEGACY** with pointer to
  `_legacy/README.md`.
- Updated `requirements.txt` comments to document the new extra and launch command.
- Created entry point: `app/nicegui_dashboard.py`.

### Phase 1 — `SimulationOrchestrator` (background data production)
- Implemented `dashviz/orchestrator.py` (~760 lines), UI-agnostic.
- Headless verification: scenario launch, manual step, model switch, consistent
  `infer_chunk`-shaped payloads (~20 ms inference latency on local ONNX).

### Phase 2 — NiceGUI application shell
- Full 5-tab layout with left control drawer, dark industrial theme (Quasar +
  custom Argus CSS), header, and all operator controls wired to the orchestrator.

### Phase 3 — Live multi-stream visualizations
- `ui.plotly` for all charts; reused `dashviz/plots.py` figure builders.
- `ui.timer(0.1)` refresh with generation-guarded no-op when idle.
- Aggressive downsampling + throttled STFT for smoothness.

### Phase 4 — Integration & edge cases
- Standalone (in-process `StreamingPerceptor`) and API (HTTP `/infer`) modes.
- Live model switching, error surfacing in alert banner, Parquet logging path,
  historical reconstruct-and-re-infer drill-down.
- API-mode parity verified bit-for-bit against standalone on identical chunk.

### Phase 5 — Packaging, deployment, documentation, cleanup
- Archived legacy Streamlit files to `_legacy/` via `git mv`.
- Updated `README.md`, `app/README.md`, `dashviz/__init__.py`, `dashviz/metrics.py`.
- Wired `deployment/Dockerfile` + `docker-compose.yml` for `api` + `dashboard` services.
- Wrote this session log.

---

## 3. Architecture

```
  ┌─────────────────────────────────────────────────────────────────┐
  │  Left drawer (global)          │  Tab panels (5)                │
  │  • mode (standalone/API)     │  Live / Lab / History / Opt /  │
  │  • model selector            │  System                        │
  │  • sim params (live sliders) │                                │
  │  • alert thresholds          │  ui.plotly charts + ui.html    │
  │  • persist logs / utilities  │  KPI cards, tables, expanders  │
  └──────────────┬────────────────────────────────────────────────┘
                 │ callbacks update SessionConfig / call orchestrator
  ┌──────────────▼────────────────────────────────────────────────┐
  │  SimulationOrchestrator  (dashviz/orchestrator.py)             │
  │  • daemon thread: park on Event → wake → simulate → infer      │
  │  • RLock for config; inference OUTSIDE lock                   │
  │  • publishes immutable Snapshot (generation counter)            │
  └──────────────┬────────────────────────────────────────────────┘
                 │ snapshot()
  ┌──────────────▼────────────────────────────────────────────────┐
  │  ui.timer(0.1s) → Dashboard._tick()                          │
  │  if generation unchanged → return (cheap)                       │
  │  else → _render_live(): in-place figure/HTML updates only       │
  └─────────────────────────────────────────────────────────────────┘
```

**Per-client isolation:** `@ui.page('/')` constructs one `Dashboard` (and thus
one `SimulationOrchestrator` + background thread) per browser client;
`ui.context.client.on_disconnect(dash.orc.shutdown)` stops the thread cleanly.

---

## 4. `dashviz/orchestrator.py` — detailed design

### 4.1 Ported from `_legacy/infra.py` (now framework-neutral)
| Function / type | Role |
| --- | --- |
| `get_perceptor` | Cached `StreamingPerceptor` per `(model, chunk_s, persist, log_dir)` |
| `get_http_client` | Cached `httpx.Client` for API mode |
| `get_simulators` | Cached `(SawVibrationSimulator, ThermalSimulator)` |
| `get_processor` | Cached `SignalProcessor` for STFT |
| `run_inference` | Unified direct + API path; same payload contract |
| `generate_chunk` | Simulator → accel + meta + thermal context |
| `compute_stft_for_display` | STFT matrix for heatmap (display resolution) |
| `params_from_log_row` | Rebuild operating point from Parquet row |
| `load_logs` | Thin wrapper over `app.logging.read_logs` |
| `alert_level_for` | Map predictions + thresholds → alert banner tuple |
| `kinematics_preview` | Derived RPM / TPF / MRR for sidebar caption |
| `available_model_entries` / `known_model_names` | Model inventory for UI selectors |
| `clear_caches` | Drop cached perceptors/clients (model reload) |

### 4.2 New types
- **`SessionConfig`** (mutable): `model`, `chunk_s`, `use_api`, `api_base_url`,
  `persist_logs`, `log_dir`, `delay_s`, `sim_params`, `thresholds`.
- **`Snapshot`** (frozen dataclass): coherent frame — `running`, `paused`,
  `step`/`max_steps`, `scenario_name`, `model`, `mode`, `latency_ms`, `error`,
  `current_result`, downsampled `wave_t`/`wave_accel`, `meta`, `stft` dict,
  `mean_temp_c`, rolling `history` (deque cap 240), **`generation`** int for
  UI change detection.

### 4.3 `SimulationOrchestrator` control surface
| Method | Behavior |
| --- | --- |
| `start(scenario_key)` | Reset + load scenario params/delay/steps/seed; begin loop |
| `start_manual()` | Free-running manual session (24 chunks default) |
| `stop()` | Halt loop; publish final snapshot |
| `toggle_pause()` | Park/resume without losing state |
| `step_once()` | Synchronous single-chunk advance (Step button) |
| `bump_wear(delta)` | Increment injected wear (clamped 0–1) |
| `update_params(**)` | Live merge of sim operating point |
| `set_manual_wear` / `set_manual_noise` | Injected wear + sensor noise multiplier |
| `set_model` / `set_mode` / `set_chunk_s` / `set_delay` | Runtime config |
| `set_thresholds` / `set_persist` | Alert tuning + Parquet logging |
| `snapshot()` | Thread-safe copy of latest `Snapshot` |
| `shutdown()` | Stop daemon thread (client disconnect) |

### 4.4 Display performance constants
- `LIVE_WAVE_POINTS = 1200`
- `LIVE_FFT_POINTS = 1200`
- STFT capped in UI layer: `max_freq_bins=200`, `max_time_bins=180`, refresh every
  3rd live tick.

### 4.5 Threading model
- Single daemon thread per orchestrator; `_wake` Event for start/step/unpause.
- Config reads/writes under `RLock`; `_advance()` does heavy work **outside** lock,
  then atomically swaps a new `Snapshot`.
- Scenario wear/noise plans applied per chunk inside `_advance`.

---

## 5. `app/nicegui_dashboard.py` — detailed design (~1220 lines)

### 5.1 Module structure
| Area | Contents |
| --- | --- |
| Theme | `_install_theme()` — `ui.colors()` + injected Argus CSS (panels, KPI, alerts, pills, Quasar dark tweaks) |
| Plot helpers | `_live_plot`, `_static_plot`, `_with_config`, `_update_plot` — dict figures with Plotly `config` for mode-bar control |
| UI helpers | `_panel`, `_labeled_slider`, `_add_noise`, `_fmt`, `_config_from_env` |
| `Dashboard` class | All tabs, drawer, timer, live render |
| Entry | `@ui.page('/')` → `main()` → `ui.run()` |

### 5.2 Left drawer (global controls)
- Operation mode toggle: Standalone vs Connected to API (+ API URL input).
- Model selector with availability badges (`[ok]` / `[--]`) and description note.
- Analysis chunk slider (0.25–1.0 s).
- Live simulation parameters: alloy, SFPM, feed/tooth, depth, teeth, injected
  wear, sensor noise (× rms).
- Derived kinematics caption (RPM, TPF, MRR).
- Alert thresholds expansion (wear alert, anomaly confidence).
- Utilities: persist-to-Parquet switch, log directory, reload caches.

### 5.3 Tab: Live Monitor
- Control bar: Start/Stop, Pause, Step, +Wear, scenario selector + Launch, rate slider.
- Status bar (HTML): LIVE/PAUSED/IDLE pill, model, mode, scenario, chunk counter, latency.
- Alert banner + recommendation card (reused `dashviz.theme` HTML helpers).
- KPI row: 3-gauge Plotly figure + health / cycle-time / temp cards.
- 2×2 plot grid: waveform, FFT, STFT heatmap, recent trend (wear + quality).
- `_tick()` at 10 Hz; `_render_live()` only when `generation` advances.

### 5.4 Tab: Simulation Lab
- **`LAB_PRESETS`** (ported): clean sharp blade, high wear (0.85), noisy robustness — Load buttons push into drawer sliders.
- Single / comparison / robustness-batch runs via `run.io_bound` (non-blocking UI).
- Single run: KPI row + 2×2 plots + optional DSP features table.
- Comparison: side-by-side table + grouped bar (wear/quality by model).
- Batch: distribution histogram, health pie, wear MAE vs injected.

### 5.5 Tab: Historical Explorer
- Load/refresh from `logs/inference` (configurable log dir).
- Summary KPIs, trend + health pie, records table (last 200).
- **Reconstruct & re-infer:** select record → regenerate chunk from logged
  operating point → re-run inference → waveform/FFT/STFT/health-prob + JSON expander.
- Uses `orch.params_from_log_row` (same contract as legacy Streamlit drill-down).

### 5.6 Tab: Optimization Sandbox
- Source toggle: latest live result vs manual override inputs.
- Editable shop-floor assumptions (cycle time, blade life, rates, costs).
- `opt.compute_production_impact` + `opt.downstream_payload` → maintenance
  alert, KPI cards, sharp-vs-current bar chart, efficiency gauge, JSON contract.

### 5.7 Tab: System & Models
- Model inventory table (kind, availability, wear MAE, health F1, latency, notes).
- ONNX CPU latency bar chart (`metrics.typical_latency_ms`).
- Robustness findings markdown (`metrics.robustness_notes`).
- Effective configuration JSON expander.

### 5.8 Environment variable overrides (Docker-friendly)
| Variable | Default | Effect |
| --- | --- | --- |
| `ARGUS_DASHBOARD_HOST` | `127.0.0.1` | Bind address |
| `ARGUS_DASHBOARD_PORT` | `8080` | Listen port |
| `ARGUS_DASHBOARD_SHOW` | `false` | Auto-open browser |
| `ARGUS_DASHBOARD_USE_API` | `false` | Start in API client mode |
| `ARGUS_API_BASE_URL` | `http://127.0.0.1:8000` | FastAPI base URL |
| `ARGUS_DASHBOARD_MODEL` | `1dcnn_normnone` | Default model |
| `ARGUS_LOG_DIR` | `logs/inference` | Parquet log root |

---

## 6. Feature parity (Streamlit → NiceGUI)

| Feature | Legacy (`_legacy/dashboard.py`) | New (`app/nicegui_dashboard.py`) |
| --- | --- | --- |
| 4 demo scenarios + noise | ✅ | ✅ (scenario selector + Launch) |
| LAB_PRESETS | ✅ | ✅ (Load → drawer sliders) |
| Live waveform/FFT/STFT/trend | ✅ (fragment rerun) | ✅ (timer + in-place update) |
| KPI gauges + cards | ✅ | ✅ |
| Alerts + recommendations | ✅ | ✅ |
| Start/stop/pause/step | ✅ | ✅ |
| Live parameter sliders | ✅ sidebar | ✅ left drawer |
| Model toggle | ✅ | ✅ |
| Standalone + API modes | ✅ | ✅ |
| Parquet persist + history | ✅ | ✅ |
| Historical drill-down | ✅ | ✅ |
| Optimization sandbox | ✅ | ✅ |
| System & models / robustness | ✅ | ✅ |
| Smooth 5–10 Hz live updates | ❌ (flicker/lag) | ✅ (architectural fix) |

---

## 7. Bugs encountered during development (and fixes)

### 7.1 `_scenario_key` bind_value AttributeError (HTTP 500 on page load)
**Symptom:** `Could not bind non-existing attribute "_scenario_key" on Dashboard`.  
**Cause:** NiceGUI `bind_value(self, "_scenario_key")` requires the attribute to
exist before binding (strict mode).  
**Fix:** Initialize `self._scenario_key = "progressive"` in `Dashboard.__init__`.

### 7.2 Plotly config overwriting figure data
**Symptom:** Initial attempt set `el._props["options"] = _PLOTLY_LIVE`, but
NiceGUI stores the serialized figure JSON in `_props["options"]` — overwriting it
with a config dict destroyed the chart.  
**Fix:** Pass figures as dicts with top-level `"config"` key via `_with_config()`;
NiceGUI's `Plotly` element documents this as the supported path for mode-bar control.

### 7.3 Port 8080 conflicts during dev restarts
**Symptom:** `winerror 10048` when relaunching server without killing prior PID.  
**Fix:** Operational — `taskkill` on owning process before restart. Not a code bug.

### 7.4 Large-file Write tool JSON error
**Symptom:** Single-shot write of ~1200-line dashboard file failed with JSON parse error.  
**Fix:** Wrote file incrementally via multiple `StrReplace` / chunked `Write` calls.

---

## 8. Verification (detailed)

### 8.1 Unit / integration regression
```text
pytest -q  →  86 passed  (before migration and after all changes)
```
No changes to `tests/`; perception stack untouched.

### 8.2 Orchestrator headless
- Started progressive scenario; observed `generation` increments, structured
  `current_result` with predictions/recommendations/latency.
- `step_once()`, `toggle_pause()`, `set_model()` exercised without UI.

### 8.3 Page build
```text
GET http://127.0.0.1:8080/  →  200  (after _scenario_key fix)
```
All five tabs construct without exception.

### 8.4 Reactive pipeline (NiceGUI user simulation, no browser)
Used `nicegui.testing` primitives (`nicegui_reset_globals`, `prepare_simulation`,
`User` + `httpx.ASGITransport`) — import module to register `@ui.page`, do **not**
call `ui.run()` (avoids script-mode conflict with `runpy.run_path`).

Steps verified:
1. `user.open("/")` → sees `ARGUS PANOPTES`, `STANDBY` banner.
2. `user.find("Launch scenario").click()` → within retries, **Stop** appears
   (proves `_tick` ran and `_update_start_button` worked).
3. Status shows **LIVE** pill.

### 8.5 API-mode parity
With `uvicorn app.main:app --port 8000`:
```text
standalone: wear=0.4101 health=monitor quality=0.7231 conf=0.3316
api       : wear=0.4101 health=monitor quality=0.7231 conf=0.3316
API_MODE_OK
```

### 8.6 User manual smoke test (post-completion)
User ran:
```powershell
pip install -e ".[ml,dl,app,dashboard-nicegui]"
python -m app.nicegui_dashboard
```
Output:
```text
NiceGUI ready to go on http://127.0.0.1:8080
dl.normalize_for_dl='none': CNN inputs keep absolute amplitude ...
forrtl: error (200): program aborting due to control-C event
```
**Interpretation (not errors):**
- Server started successfully.
- `dl.normalize_for_dl='none'` — one-time informational log when the default CNN
  model loads on first inference (expected for `1dcnn_normnone`).
- `forrtl: error (200)` — cosmetic Windows/Anaconda/MKL shutdown noise after
  **Ctrl+C** (`libifcoremd.dll`); exit code 1 is normal for interrupted Python.

---

## 9. Files created, modified, archived

### Created
| Path | ~Lines | Purpose |
| --- | ---: | --- |
| `dashviz/orchestrator.py` | 760 | Background orchestrator + unified inference |
| `app/nicegui_dashboard.py` | 1220 | NiceGUI operator dashboard (entry point) |
| `_legacy/README.md` | — | Migration rationale + how to run Streamlit v1 |
| `logs/2026-07-07_argus-v1-nicegui-dashboard.md` | — | This log |

### Modified
| Path | Change |
| --- | --- |
| `pyproject.toml` | `dashboard-nicegui` extra; `dashboard` marked legacy |
| `requirements.txt` | Document new extra + launch command |
| `dashviz/__init__.py` | v0.2.0; documents orchestrator; drops `infra` from `__all__` |
| `dashviz/metrics.py` | Docstring: framework-neutral (no Streamlit cache mention) |
| `deployment/Dockerfile` | `pip install -e ".[ml,dl,app,dashboard-nicegui]"`; EXPOSE 8000/8080 |
| `deployment/docker-compose.yml` | Live `api` + `dashboard` services |
| `README.md` | Status table, dashboard section, repo tree, deps blurb |
| `app/README.md` | Day 5 complete; NiceGUI run instructions + env vars |

### Archived (`git mv` → `_legacy/`)
| Original path | Notes |
| --- | --- |
| `dashboard.py` | Import changed to `import infra` (local) |
| `dashviz/infra.py` | Streamlit-coupled predecessor of orchestrator |
| `_smoke_dashboard.py` | Legacy smoke harness |
| `_apptest_dashboard.py` | Streamlit AppTest harness |
| `_apptest_step.py` | Streamlit step harness |

**Kept in `dashviz/` (reused by NiceGUI):** `theme.py`, `plots.py`,
`optimization.py`, `scenarios.py`, `metrics.py`.

---

## 10. How to run (production dashboard)

```bash
# Full demo install
pip install -e ".[ml,dl,app,dashboard-nicegui]"

# Standalone (lowest latency — recommended for recordings)
python -m app.nicegui_dashboard
# → http://127.0.0.1:8080

# API client mode (two terminals)
uvicorn app.main:app --reload          # terminal 1
python -m app.nicegui_dashboard        # terminal 2; toggle "Connected to API" in drawer

# Docker compose (dashboard defaults to API mode against `api` service)
docker compose -f deployment/docker-compose.yml up api dashboard
```

**Legacy Streamlit UI** (reference only):
```bash
pip install -e ".[ml,dl,app,dashboard]"
cd _legacy && streamlit run dashboard.py   # http://localhost:8501
```

---

## 11. Performance notes

| Aspect | Streamlit v1 | NiceGUI v2 |
| --- | --- | --- |
| Update model | Full fragment rerun | Generation-guarded timer callback |
| Data production | Same thread as UI | Background daemon thread |
| STFT refresh | Every rerun (~heavy) | Every 3rd tick + bin caps |
| Waveform/FFT points | Downsampled in plots | Capped at 1200 via orchestrator constants |
| Perceived live rate | Uneven; flicker | Stable ~10 Hz timer (configurable via `delay_s`) |
| Mode bar repaint | Visible on hover | Hidden on live charts (`displayModeBar: false`) |

Subjective goal met: launching **Noisy Sensor Robustness** with all four live
plots visible produces smooth gauge/plot updates without elements disappearing.

---

## 12. Remaining TODOs

- [ ] Capture screenshots/GIFs (`docs/dashboard-live.png`, `docs/dashboard-lab.png`) for READMEs.
- [ ] Add a committed NiceGUI headless pytest (requires `pytest-asyncio` + `nicegui.testing.user` fixture).
- [ ] Optional: shared read-only orchestrator for many concurrent operator clients (current: one thread per browser tab — fine for demos).
- [ ] Optional: suppress MKL `forrtl` noise on Windows Ctrl+C (cosmetic; low priority).

---

## 13. Session Q&A recap

| User question | Resolution |
| --- | --- |
| "Is that finished?" | Yes — all 5 phases complete; 86 tests pass; dashboard verified headlessly and manually. |
| Terminal `forrtl` / `normalize_for_dl` lines | Explained: normal startup + model info + Ctrl+C MKL noise; not a crash. |
| Background shell exit codes 1/3 | Expected from `taskkill` during dev server restarts; not production failures. |

---

*End of session log.*
