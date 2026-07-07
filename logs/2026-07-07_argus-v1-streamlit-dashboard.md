# Argus Panoptes — Session Change Log

**Date:** 2026-07-07
**Scope:** Day-5 ops layer — completing, theming, and performance-hardening the
**Streamlit + Plotly operator dashboard** (`dashboard.py` + the `dashviz/`
helper package). This session took the dashboard from a partially-scaffolded
state to a runnable, executive-demo-quality UI: filled remaining feature gaps
(README docs, historical record drill-down, a 4th demo scenario, Simulation-Lab
presets, robustness-notes surfacing), then addressed three rounds of user
feedback — **(1)** low-contrast colors, **(2)** white-on-white native widgets +
an opaque white top bar, and **(3)** heavy flicker / inefficiency during live
runs. **No changes** to `sensors/`, `dsp/`, `models/`, or `app/` runtime code —
the dashboard only *consumes* the existing `StreamingPerceptor`, `InferenceLogger`,
FastAPI service, simulators, and experiment artifacts, so there are no
regressions to the perception layers, the Parquet schema, or the trained models.
**Environment:** Python 3.13 (Windows / PowerShell). **streamlit 1.51.0**,
plotly (dashboard extra). Runs standalone against an in-process
`StreamingPerceptor` (lowest latency) or over HTTP against the FastAPI service.

---

## 1. Starting state (Step 0 exploration)

The repo already contained a substantial dashboard scaffold from earlier work:
`dashboard.py` (~1084 lines) plus a `dashviz/` package (`theme.py`, `plots.py`,
`infra.py`, `optimization.py`, `scenarios.py`, `metrics.py`, `__init__.py`) and
three headless test harnesses (`_smoke_dashboard.py`, `_apptest_dashboard.py`,
`_apptest_step.py`). The `[dashboard]` extra (`streamlit>=1.36`, `plotly>=5.22`,
`httpx>=0.27`) was already present in `pyproject.toml` / `requirements.txt`, and
`dashviz` was already registered under `[tool.setuptools] packages`.

Contracts internalized before touching code (unchanged, consumed as-is):
- **`StreamingPerceptor.infer_chunk(...)`** returns the structured payload
  (`predictions` = wear_level / cycle_time_factor / quality_score / health_state
  / health_probs / anomaly_flag / confidence; `recommendations`; `metadata`;
  optional `features`; provenance + `latency_ms`). Models via `MODEL_REGISTRY` /
  `available_models()` / `resolve_model_name()`.
- **`app.logging.read_logs(log_dir)`** → partitioned Parquet DataFrame with
  `pred_*`, `health_prob_*`, echoed operating-point scalars.
- **Simulators** (`SawVibrationSimulator`, `ThermalSimulator`) `generate(...)`
  returning `(t, signal, meta)` with kinematics (`rpm`, `tooth_pass_freq_hz`,
  `material_removal_rate_mm3_s`, ...); alloys `6061` / `7075`.
- **Experiment JSONs** (`experiments/onnx_benchmark.json`, `*_metrics.json`,
  `robustness_results.json`) for latency/accuracy surfacing.

Verification of the starting state: `_smoke_dashboard.py`, `_apptest_dashboard.py`
(5 tabs, 0 errors), and `_apptest_step.py` all passed.

---

## 2. Feature-gap completion

### 2.1 New demo scenario (`dashviz/scenarios.py`)
Added a **4th** self-contained scenario, **"Noisy Sensor Robustness"**
(`_noisy_robust_plan`: fixed moderate wear ≈0.42 with per-chunk jitter). Extended
the `Scenario` dataclass with an optional **`noise_sd`** field (default `0.0`) so a
scenario can inject Gaussian sensor noise (`×rms`) at render time; the new
scenario sets `noise_sd=0.35` to visibly stress the model and motivate the
`normnone` vs `noisy` comparison. Existing three scenarios (Normal, Progressive
Wear, Sudden Anomaly) untouched.

### 2.2 Robustness-notes surfacing (`dashviz/metrics.py`)
Added **`robustness_notes()`** which parses `robustness_results.json`
(`1dcnn_zscore_baseline` vs `1dcnn_zscore_noisy015`, `gaussian sd=0.5*rms`
config) into a human-readable Markdown summary (clean wear MAE, the
baseline→noisy MAE improvement under corruption, and the normnone/noisy
trade-off). Replaces a prior `rob.get("notes")` lookup that didn't match the
JSON's actual schema.

### 2.3 Historical record drill-down (`dashboard.py` + `dashviz/infra.py`)
Implemented the spec's **"Visualize Selected Record"**: a record selectbox +
button in the Historical Explorer that rebuilds the operating point from the
logged row (**`infra.params_from_log_row(row)`**, new), re-generates the chunk via
the simulator (raw waveforms aren't stored in Parquet), re-infers, and shows
reconstructed waveform / FFT / STFT / health-prob plots plus the full JSON in an
expander. Clearly labeled as a reconstruction from the logged operating point.

### 2.4 Simulation-Lab presets (`dashboard.py`)
Added a **`LAB_PRESETS`** dict (clean sharp blade / high-wear / noisy-robustness)
and a presets expander whose buttons write directly into the Lab form's
`st.session_state` widget keys (mapped via `_LAB_KEY_MAP` to `lab_alloy`,
`lab_sfpm`, `lab_feed`, `lab_depth`, `lab_teeth`, `lab_wear`, `lab_noise`,
`lab_seed`) and `st.rerun()` so the form reflects the loaded values, each with an
"expected behavior" note tied to the ablation results.

### 2.5 Session-state hygiene (`dashviz/infra.py`)
Added `ss.live_error = ""` to the `init_session_state()` schema (previously
read/written ad-hoc). Imported `pandas as pd` for `params_from_log_row`.

### 2.6 Documentation
- **Root `README.md`:** status table (`dashboard.py` ✅ Day 5 complete); new
  **"### Dashboard (Day 5)"** section (install `pip install -e ".[ml,dl,app,dashboard]"`,
  run `streamlit run dashboard.py`, standalone vs API modes, a 5-tab overview
  table, demo-scenario usage, log-seeding via `scripts/stream_demo.py`, and a
  screenshots/GIFs placeholder); repo-layout block for `dashviz/` + `dashboard.py`;
  tech-stack `[dashboard]` note; roadmap Day 5 ✅.
- **`app/README.md`:** "Planned (Day 5)" Streamlit bullet marked ✅.

---

## 3. Round 1 — color contrast pass (`dashviz/theme.py`, `plots.py`, `dashboard.py`)

User report: *"quite a few colors ... are too similar to the ones behind it."*
Retuned the single source-of-truth `Palette` for readable contrast on the
charcoal surfaces, then propagated brighter tokens everywhere.

Palette changes (before → after):

| Token | Before | After | Reason |
| --- | --- | --- | --- |
| `text` | `#e2e8f0` | `#f1f5f9` | brighter primary text |
| `text_muted` | `#94a3b8` | `#cbd5e1` | widget/axis labels too dim |
| `text_faint` | `#64748b` | `#94a3b8` | captions nearly invisible |
| `surface_2` | `#243244` | `#334155` | was ~indistinguishable from `surface` |
| `border` | `#334155` | `#475569` | visible card/axis borders |
| `grid` | `#1f2b3d` | `#3d4f66` | readable plot gridlines |
| `accent` | `#14b8a6` | `#2dd4bf` | brighter teal for lines/gauges |
| `accent_soft` | `#2dd4bf` | `#5eead4` | highlight on dark |
| `warning` | `#f59e0b` | `#fbbf24` | amber pops more |
| `critical` | `#ef4444` | `#f87171` | red readable as text/line |
| `info` | `#38bdf8` | `#7dd3fc` | FFT/info lines |
| `purple` | `#a78bfa` | `#c4b5fd` | extra series |
| `neutral` (new) | — | `#a8b4c8` | baseline bars (not caption-gray) |

- **`get_dark_layout`:** gave plots a subtle panel background
  (`plot_bgcolor = rgba(surface, 0.35)`) so gridlines read; brighter legend /
  title / axis-title / tick fonts; thicker gridlines/zerolines/axis lines.
- **`plots.py`:** thicker waveform/FFT lines (1.1→1.4) and stronger fills
  (0.10→0.20–0.22); TPF annotation given a solid bordered background; STFT
  colorbar title brightened + panel bg; gauge step-band alpha 0.22→0.38 and
  larger tick labels; bar/pie value labels use primary text; marker outlines
  switched from `bg` to `surface`.
- **`dashboard.py`:** IDLE status pill + sidebar subtitle use `text_muted`;
  optimization "baseline vs current" bars use `neutral`/`accent_soft`.

Also moved `_rgba` above `get_dark_layout` in `theme.py` (it's now used by the
layout builder, not just the HTML helpers).

---

## 4. Round 2 — white-on-white widgets + white top bar

User report: *"a number of buttons/text boxes are white ... can't see the text
unless I highlight it. Also, there's a white bar at the top covering components."*

**Root cause:** there was **no `.streamlit/config.toml`**, so Streamlit rendered
its native widgets (text/number inputs, form-submit buttons, multiselect chips)
and the top **header** using its default (light) chrome — hence white surfaces
with invisible text, and an opaque white header bar over the page top.

### 4.1 New `.streamlit/config.toml`
Sets a dark theme base mirroring the palette so every native widget defaults to
dark surfaces + light text:

```toml
[theme]
base = "dark"
primaryColor = "#2dd4bf"
backgroundColor = "#0f172a"
secondaryBackgroundColor = "#1e293b"
textColor = "#f1f5f9"
font = "sans serif"

[server]
headless = true
[browser]
gatherUsageStats = false
```

### 4.2 CSS hardening (`dashviz/theme.py build_css`)
- **Top bar:** `header[data-testid="stHeader"]` → transparent background + no
  shadow; `stDecoration` transparent; toolbar/main-menu icons tinted to
  `text_muted`; toolbar nudged with `right: 0.5rem`.
- **Buttons:** unified `.stButton`, **`.stFormSubmitButton`**,
  **`.stDownloadButton`**, `.stLinkButton` styling with explicit `color` on the
  inner `<p>` (Streamlit wraps button labels in `<p>`) for both normal and hover;
  primary buttons keep the teal fill with dark (`#04121a`) text.
- **Inputs:** text/number inputs + textareas + base-input given dark surface,
  forced light text via `color` **and `-webkit-text-fill-color`** (so text is
  readable without highlighting), readable placeholders, styled number-input
  steppers, and dark dropdown-popover options with hover states.
- **Multiselect chips:** `span[data-baseweb="tag"]` → teal background with dark
  text/icon (were white with pale text).
- **Misc:** radio/checkbox option labels, expander `details` surface, `st.form`
  container border/tint, and code/JSON blocks (`.stCode`, `.stJson`, `pre`,
  `code`) all forced to dark surfaces + light text.

---

## 5. Round 3 — live-run flicker & inefficiency (the key fix)

User report: *"running incredibly inefficiently ... when I run a scenario, the
plots and blade-wear indicators disappear and reappear until it ends."*

### 5.1 Root cause
The live loop used `st.rerun(scope="fragment")` with a fallback to a full
`st.rerun()` (`_fragment_rerun`). In **Streamlit 1.51**, `scope="fragment"` is
**rejected when the fragment executes during a full-app run** — which is exactly
how a scenario launch (a sidebar button → full rerun) enters the Live Monitor.
So the first tick raised, hit the fallback, and did a **full-page `st.rerun()`**;
that full rerun re-entered the fragment in full-run context again → raised again
→ full rerun again. Net effect: a **full-page rerun every ~100 ms**, tearing down
and rebuilding the entire app (sidebar + all five tabs + every plot and gauge)
each frame. That is the disappear/reappear flicker and the wasted work.

### 5.2 Fix — dynamic `run_every` fragment
Replaced the manual sleep + `st.rerun` loop with Streamlit's intended live-view
pattern: a fragment whose refresh timer is armed **only while running**.

```python
def render_live_tab() -> None:
    ...
    # fragment-scoped auto-refresh while running; no timer (zero cost) when idle
    run_every = float(st.session_state.delay_s) if st.session_state.running else None
    st.fragment(_live_monitor_body, run_every=run_every)()

def _live_monitor_body() -> None:
    ss = st.session_state
    _render_control_bar()
    if ss.running:
        try:
            _advance_one_step()
        except RuntimeError as exc:
            ss.running = False
            ss.live_error = str(exc)
        if not ss.running:          # completed or errored -> disarm the timer
            st.rerun()              # single full rerun recomputes run_every=None
    ...
    _render_status_bar(); _render_kpi_row(...); _render_live_alert(...)
    _render_live_plots(); _render_recommendation(...)
```

Consequences:
- **While running:** `run_every=delay_s` → **fragment-scoped** auto-reruns; only
  the Live Monitor subtree repaints, and Plotly charts (which carry stable
  `key`s: `g_wear`, `g_quality`, `g_conf`, `p_wave`, `p_fft`, `p_stft`, `p_trend`)
  update in place instead of being destroyed/recreated — no full-page flicker.
- **While idle:** `run_every=None` → the timer is fully disabled, so there is
  **zero background rerun cost** (the old design effectively spun the whole app
  even between ticks).
- **Start/Stop:** the control-bar toggle now calls `st.rerun()` after flipping
  `ss.running` so the outer `render_live_tab` recomputes `run_every` (a button
  click inside a fragment otherwise only reruns the fragment, leaving the timer
  un-armed/un-disarmed).
- **Completion/error:** a single full rerun disarms the timer cleanly (no
  runaway idle ticks).

Removed the now-dead `_fragment_rerun` helper, the `time.sleep` busy-wait, and
the unused `time` / `streamlit.errors.StreamlitAPIException` imports.

---

## 6. Testing & verification

All three headless harnesses were re-run after each round (`_smoke_dashboard.py`
exercises the heavy non-Streamlit paths; `_apptest_dashboard.py` renders the full
app via `AppTest`; `_apptest_step.py` clicks the Live Monitor "Step" control and
asserts state updates).

```
$ python _smoke_dashboard.py
scenarios OK: ['normal', 'progressive', 'anomaly', 'noisy_sensor']
infer OK: wear=0.611 health=warning conf=0.67 lat=23.75ms action=reduce_feed_plan_blade_change
figures OK: ['waveform', 'fft', 'stft', 'gauge', 'probs', 'hist', 'pie', 'bar']
empty-figure guards OK
optimization OK: eff=47.6 cost=$23.27 maint=warning
available models: 10/10
ALL SMOKE TESTS PASSED

$ python _apptest_dashboard.py
tabs rendered: 5
errors: 0  warnings: 0
APPTEST OK (no exceptions)

$ python _apptest_step.py
STEP OK: wear=0.360 health=monitor conf=0.68 history_len=1
APPTEST STEP OK
```

Additional checks:
- `python -c "from dashviz.theme import build_css; ..."` → CSS builds (len ≈10.6k).
- `python scripts/stream_demo.py --model 1dcnn_normnone --duration-s 3 --wear 0.6`
  wrote 3 records; `read_logs('logs/inference')` returns them so the Historical
  Explorer has data.
- `ReadLints` on `dashboard.py` / `dashviz/*`: no linter errors.

Non-blocking: repeated `use_container_width` deprecation notices from Streamlit
1.51 (removal after 2025-12-31) — see TODOs.

---

## 7. Bugs / tuning encountered and resolved

1. **`st.rerun(scope="fragment")` full-page-rerun loop** (§5) — the session's
   headline fix; replaced with a dynamic `run_every` fragment.
2. **Missing `.streamlit/config.toml`** (§4) — native widgets + header rendered
   light; added a dark-theme config so widget chrome is dark by default rather
   than relying solely on CSS overrides.
3. **Button label text invisible** — Streamlit wraps button/label text in inner
   `<p>` elements; setting `color` on the `<button>` alone was insufficient, so
   explicit `... p { color }` rules were added (incl. hover + primary variants).
4. **Input text unreadable until highlighted** — some browsers honor
   `-webkit-text-fill-color` over `color` for form fields; set both.
5. **Broken merge in `infra.py`** — inserting `params_from_log_row` initially
   spliced its body ahead of the `alert_level_for` docstring/signature; repaired
   so both functions are well-formed (caught immediately by re-reading the file).
6. **Lab preset keys** — presets first wrote generic `lab_<param>` keys that
   didn't match the actual widget keys (`lab_sfpm`/`lab_feed`/`lab_teeth`); added
   `_LAB_KEY_MAP` + `st.rerun()` so loaded presets populate the form.
7. **Duplicated `_rgba`** — `get_dark_layout` began using `_rgba`; moved the
   definition above it in `theme.py` and left the HTML helpers referencing the
   same single implementation.

---

## 8. Files touched this session

- **New:** `.streamlit/config.toml`; `logs/2026-07-07_argus-v1-streamlit-dashboard.md`
  (this log).
- **Modified:** `dashboard.py` (live-loop `run_every` rewrite, Lab presets,
  historical drill-down, color tokens, import cleanup); `dashviz/theme.py`
  (palette retune, CSS hardening, `_rgba` move); `dashviz/plots.py` (contrast:
  lines/fills/gridlines/gauge bands/annotations); `dashviz/infra.py`
  (`params_from_log_row`, `live_error` init, pandas import); `dashviz/scenarios.py`
  (4th scenario + `noise_sd`); `dashviz/metrics.py` (`robustness_notes`);
  `README.md`; `app/README.md`.
- **Untouched:** all `sensors/`, `dsp/`, `models/`, `app/*.py` runtime code; the
  Parquet schema; trained artifacts.

---

## 9. Immediate next steps / TODOs

1. **`use_container_width` migration** — Streamlit 1.51 deprecates it (removal
   after 2025-12-31) in favor of `width="stretch"`/`"content"`; the live-plot
   render helper (`_plotly`) already uses `width="stretch"`, but the many
   `st.button`/`st.dataframe`/`st.form_submit_button(..., use_container_width=True)`
   call sites still emit notices and should be migrated.
2. **Screenshots / GIFs** — populate the README placeholder (`docs/dashboard-*.png`)
   for portfolio/demo use now that the UI is stable.
3. **Per-tab fragments** — wrap Simulation Lab / Historical Explorer bodies in
   their own `st.fragment`s so their interactions don't repaint sibling tabs.
4. **Optional waveform persistence** — store downsampled waveforms in the Parquet
   logs for true historical replay (the drill-down currently *reconstructs* the
   chunk from the logged operating point).
5. **Fusion thermal standardization** — carry the Day-4 note forward so the
   fusion streaming path can accept raw thermal scalars end-to-end.

Session complete — the Day-5 Streamlit dashboard is runnable
(`streamlit run dashboard.py` after `pip install -e ".[ml,dl,app,dashboard]"`),
dark-themed with readable contrast, free of white-on-white widgets and the white
header bar, and renders live simulations smoothly via fragment-scoped refresh
with no full-page flicker.
