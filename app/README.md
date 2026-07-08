# `app/` — Inference, Integration & Dashboard

> **Status:** ✅ **Day 4 + Day 5 complete** — streaming perceptor, Parquet
> inference logging, a FastAPI service, and the high-performance **NiceGUI +
> Plotly** operator dashboard (`app/nicegui_dashboard.py`). Mock PLC/OPC-UA
> remains scaffolded.

## What's implemented

- **`StreamingPerceptor`** (`models/streaming_perceptor.py`) — real-time chunk
  processing over a bounded ring buffer. Reuses `dsp.SignalProcessor` for the
  DSP front-end and either the XGBoost baseline or an exported ONNX variant
  (`models.onnx_inference.ONNXPerceptor`, torch-free). Supports the hardened
  variants (`*_normnone`, `*_noisy`, `*_noisy01`) via friendly name mapping.
- **`InferenceLogger`** (`app/logging.py`) — buffers predictions and flushes to a
  partitioned Parquet dataset (`records/` + `manifest.parquet`, Hive-partitioned
  by `date` / `model`), queryable with `pyarrow.dataset`.
- **FastAPI service** (`app/main.py`) — `POST /infer`, `POST /batch`,
  `GET /health`, `GET /models`. Structured JSON payloads (`wear_level`,
  `cycle_time_factor`, `quality_score`, `health_state` + probabilities,
  `anomaly_flag`, `confidence`, `recommendations`, `latency_ms`, provenance) for
  the downstream cost / cycle-time / nesting optimizer. Models are pooled and
  lazily loaded (default pre-loaded on startup); config via `ARGUS_*` env vars.

## Install

```bash
pip install -e ".[ml,dl,app]"   # xgboost/joblib + onnxruntime + fastapi/uvicorn
```

The service itself is **torch-free**: DL variants run through `onnxruntime`, the
baseline through `joblib`/`xgboost`.

## Run the service

```bash
uvicorn app.main:app --reload
# docs at http://127.0.0.1:8000/docs
```

Configuration (env vars, all optional):

| Variable | Default | Meaning |
| --- | --- | --- |
| `ARGUS_DEFAULT_MODEL` | `1dcnn_normnone` | Model used when a request omits one |
| `ARGUS_PRELOAD_MODELS` | `1dcnn_normnone` | Comma list pre-loaded on startup |
| `ARGUS_MODEL_DIR` | `experiments/models` | Artifact directory |
| `ARGUS_FS_HZ` | `40960` | Sample rate (Hz) |
| `ARGUS_CHUNK_S` | `1.0` | Analysis chunk length (s) |
| `ARGUS_LOG_ENABLED` | `true` | Persist predictions to Parquet |
| `ARGUS_LOG_DIR` | `logs/inference` | Parquet log root |
| `ARGUS_CORS_ORIGINS` | `*` | Comma list of allowed origins |

## Example requests

```bash
# List models / variants and artifact availability
curl http://127.0.0.1:8000/models

# Single-chunk inference (vibration is a raw waveform array in g)
curl -X POST "http://127.0.0.1:8000/infer" \
  -H "Content-Type: application/json" \
  -d '{"model": "1dcnn_normnone", "vibration": [0.1, 0.2, -0.1, 0.05], "fs_hz": 40960}'

# XGBoost path with operating-point context (no waveform array required to run,
# but DSP features come from the waveform when supplied)
curl -X POST "http://127.0.0.1:8000/infer?model=xgboost" \
  -H "Content-Type: application/json" \
  -d '{"vibration": [/* ... */], "params": {"alloy": "6061", "num_teeth": 80}}'
```

Available model names: `xgboost`, `1dcnn`, `1dcnn_normnone`, `1dcnn_noisy`,
`1dcnn_noisy01`, `fusion`, `fusion_normnone`, `fusion_noisy`, `fusion_noisy01`,
`spectrogram` (aliases: `xgb`, `cnn1d`, `cnn2d`, ...).

## Live streaming demo (no HTTP)

```bash
python scripts/stream_demo.py --model 1dcnn_normnone --duration-s 5 --wear 0.6
python scripts/stream_demo.py --model xgboost --with-thermal --wear 0.9
```

Simulator → `StreamingPerceptor` → DSP + model → Parquet log → printed payloads.

## Query the inference logs

```python
from app.logging import read_logs
df = read_logs("logs/inference")          # partition cols (date/model) restored
df.groupby("model")["pred_wear_level"].mean()

# Or with pyarrow directly:
import pyarrow.dataset as ds
d = ds.dataset("logs/inference/records", format="parquet", partitioning="hive")
recent = d.to_table(filter=ds.field("model") == "1dcnn_normnone").to_pandas()
```

## Notes

- **Fusion thermal branch:** the fusion models were trained on *standardized*
  thermal scalars; those standardization stats aren't persisted with the ONNX
  artifact, so `infer_chunk` defaults the thermal input to a neutral zero vector
  (the standardized dataset mean). Pass an already-prepared `thermal` vector to
  override. The waveform-only (`1dcnn*`) and `xgboost` paths need no such prep.
- **`*_normnone` vs `*_noisy`:** `normnone` keeps absolute amplitude (best clean
  accuracy); `noisy`/`noisy01` are z-score + noise-augmented for sensor
  robustness. The perceptor configures a matching DSP front-end automatically.

## Operator dashboard (Day 5)

The dashboard is a high-performance **NiceGUI + Plotly** app in
`app/nicegui_dashboard.py`. A background `dashviz.orchestrator.SimulationOrchestrator`
thread runs the simulate → DSP → infer loop and publishes thread-safe `Snapshot`s;
the UI refreshes via a single `ui.timer` that pushes only changed figures/labels
in place (aggressive waveform/FFT downsampling + a throttled STFT heatmap) for
smooth, flicker-free ~5–10 Hz live updates.

```bash
pip install -e ".[ml,dl,app,dashboard-nicegui]"
python -m app.nicegui_dashboard        # http://127.0.0.1:8080
```

Modes: **Standalone** (in-process `StreamingPerceptor`, lowest latency) or
**Connected to API** (calls this service's `/infer` over HTTP). Env overrides
(used by `deployment/docker-compose.yml`): `ARGUS_DASHBOARD_HOST`,
`ARGUS_DASHBOARD_PORT`, `ARGUS_DASHBOARD_USE_API`, `ARGUS_API_BASE_URL`,
`ARGUS_DASHBOARD_MODEL`, `ARGUS_LOG_DIR`.

Tabs: **Live Monitor** (waveform/FFT/STFT/trend, KPI gauges, alerts,
recommendations, start/stop/pause/step, scenarios), **Simulation Lab** (presets +
single/multi-model/robustness-batch runs), **Historical Explorer** (query Parquet
logs + reconstruct/re-infer a record), **Optimization Sandbox** (downstream
production-impact model), **System & Models** (inventory, ONNX latency, robustness
notes). The reusable, framework-neutral helpers live in the `dashviz/` package.

> The original Streamlit dashboard is archived under `_legacy/` (see
> `_legacy/README.md`); it was superseded for the reasons above.

## Planned

- Mock PLC / OPC-UA tags and ML feature store.
