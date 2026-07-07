# Argus Panoptes — Session Change Log

**Date:** 2026-07-07
**Scope:** Day-4 integration layer — a stateful **`StreamingPerceptor`** (ring
buffer + DSP + model) for real-time chunked inference, a **`InferenceLogger`**
that persists predictions to a partitioned Parquet dataset, and a **FastAPI**
service (`/infer`, `/batch`, `/health`, `/models`) emitting structured JSON
payloads for the downstream cost / cycle-time / nesting optimizer. A live-demo
script, new tests, and docs. **No changes** to `sensors/`, `dsp/`, or the
existing `models/` code — the new layer only *consumes* them, so there are no
regressions to the simulators, the Parquet schema, or the trained artifacts.
**Environment:** Python 3.13.9 (Windows / PowerShell). Core + `ml` + `dl` deps
unchanged (onnxruntime 1.27.0, xgboost 3.3.0, scikit-learn 1.7.2). New **`app`
extra** installed via `python -m pip install "fastapi>=0.110" "uvicorn[standard]>=0.29"
"python-multipart>=0.0.9" "httpx>=0.27"`: **fastapi 0.139.0**, **uvicorn 0.50.2**,
**starlette 1.3.1**, python-multipart 0.0.32 (+ httptools / watchfiles /
websockets). The service is **torch-free at runtime** — DL variants run on
onnxruntime, the baseline on joblib/xgboost.

---

## 1. Design decisions

- **Reuse, don't reimplement.** The perceptor is pure glue: it calls the *exact*
  same `dsp.SignalProcessor` (feature / `get_normalized_waveform` /
  `compute_spectrogram`) and `models.onnx_inference.ONNXPerceptor` used offline,
  so streaming inputs are byte-identical to training inputs. Nothing in
  `sensors/`, `dsp/`, or `models/` was modified.
- **Ring buffer over a `deque`.** Live sensors deliver irregular bursts, not tidy
  analysis windows. `ingest()` appends to a bounded `collections.deque`
  (`maxlen = buffer_capacity_chunks × chunk_samples`, default 8 chunks → memory is
  O(chunk)); `process_next_chunk()` pops exactly one `chunk_samples` window off the
  **front** when enough has accumulated and returns `None` otherwise (partial tail
  retained). `infer_chunk()` runs the same pipeline on a caller-supplied chunk
  directly — that's the path the FastAPI `/infer` endpoint uses.
- **Friendly-name model registry.** `MODEL_REGISTRY` maps `xgboost` + every
  hardened DL variant (`1dcnn`, `1dcnn_normnone`, `1dcnn_noisy`, `1dcnn_noisy01`,
  `fusion*`, `spectrogram`) to `ModelSpec(kind, onnx, normalize_for_dl,
  description)`. Crucially each spec carries the **`normalize_for_dl` recipe the
  artifact was trained with** (`"none"` for `*_normnone`, else `"zscore"`) so the
  perceptor auto-configures a *matching* DSP front-end — otherwise a `normnone`
  (amplitude-preserving) model would be fed z-scored inputs and produce garbage.
  Aliases (`xgb`, `cnn1d`, `cnn2d`, ...) resolve via `resolve_model_name`.
- **Lazy, torch-free loading.** No model is loaded until first inference (or an
  explicit `.load()`). ONNX kind is read back from the session's input names
  (`waveform` / `spectrogram` / `waveform`+`thermal`); XGBoost feature order is
  read from `clf.feature_names_in_` (42 features: 13 `vib_td_*`, 15 `vib_fd_*`,
  5 thermal, 9 kinematics context) and fed as a named `DataFrame` to avoid
  sklearn's feature-name-mismatch warning. `import models` stays torch-free.
- **Finite, physically-bounded outputs.** All predictions are `nan_to_num`'d and
  clamped: `wear_level`/`quality_score` → `[0,1]`, `cycle_time_factor` → `≥0`;
  health probs re-aligned to the canonical class order and re-normalized. Missing
  TPF / context degrade gracefully (XGBoost eats `NaN` natively; DSP returns
  `NaN` TPF-relative features as designed).
- **Structured payload for downstream optimizers.** Every result carries
  `predictions` (wear/cycle/quality + `health_state` + per-class `health_probs` +
  `anomaly_flag` + `confidence`), a **`recommendations`** block (`action` mapped
  from health state — `continue` / `monitor_schedule_inspection` /
  `reduce_feed_plan_blade_change` / `stop_replace_blade` — plus
  `blade_change_suggested`, cycle/quality), echoed operating-point `metadata`,
  optional DSP `features`, and provenance (`model`, `model_variant`,
  `latency_ms`, `timestamp`).
- **Partitioned Parquet logging, dataset-gen parity.** `InferenceLogger` mirrors
  the `records/` + `manifest.parquet` layout of `scripts/generate_dataset.py`
  (Hive-partitioned by `date` / `model`, `pq.write_to_dataset` with
  `existing_data_behavior="overwrite_or_ignore"`), so logs are queryable with the
  same `pyarrow.dataset` tooling and are retraining-ready. Count- **or**
  time-based flushing; thread-safe (`threading.Lock`) for concurrent API requests.
- **Import hygiene / no cycles.** `app/logging.py` depends only on core deps
  (pandas / pyarrow / pydantic) — never on `models` or FastAPI — so the perceptor
  can import it lazily and stay light. `app/__init__.py` uses `__getattr__` to
  expose `create_app` / `app` / `InferenceLogger` **lazily**, so `import app`
  (and `import app.logging`) never eagerly pulls in FastAPI/uvicorn. Note:
  `app/logging.py` is a submodule and does **not** shadow the stdlib `logging`
  (absolute imports resolve `import logging` to the stdlib).

---

## 2. `StreamingPerceptor` (`models/streaming_perceptor.py`, new, v0.1.0)

Registry + variant recipe (excerpt):

```python
@dataclass(frozen=True)
class ModelSpec:
    name: str; kind: str; onnx: str | None; normalize_for_dl: str; description: str

MODEL_REGISTRY = {
    "xgboost": ModelSpec("xgboost", "xgboost", None, "none", "XGBoost baseline ..."),
    "1dcnn_normnone": ModelSpec("1dcnn_normnone", "waveform",
                                "dl_1dcnn_normnone.onnx", "none", "amplitude-preserving"),
    "fusion_normnone": ModelSpec("fusion_normnone", "fusion",
                                 "dl_fusion_normnone.onnx", "none", "..."),
    # ... 1dcnn / fusion (+_noisy/_noisy01), spectrogram ...
}
```

DL front-end is auto-matched to the artifact by deriving a processor whose
`dl.normalize_for_dl` equals the spec's:

```python
base = self.processor.config.model_dump()
base["dl"] = {**base.get("dl", {}), "normalize_for_dl": self.spec.normalize_for_dl}
self._dl_processor = SignalProcessor(ProcessorConfig.model_validate(base))
```

Public API: `ingest(samples, metadata=None)`, `process_next_chunk() -> dict|None`,
`infer_chunk(vibration, *, thermal=None, metadata=None, params=None, chunk_id=None,
wear_injected=None, return_features=True) -> dict`, `get_latest_prediction()`,
`stream_from_simulator(simulator, *, duration_s, chunk_s, params, wear, seed,
thermal_simulator=None, ...)`, plus `load()`, `reset()`, `buffered_samples()`,
`flush()`, `close()`, and context-manager support. Typed **`StreamingConfig`**
(pydantic v2, `extra="forbid"`, `protected_namespaces=()` to allow the `model` /
`model_dir` fields): `fs_hz=40960`, `chunk_s=1.0`, `model="1dcnn_normnone"`,
`target_len=None` (CNNs global-pool → variable length OK),
`buffer_capacity_chunks=8`, `min_chunk_samples=64`, `providers=None`.

Inference dispatch:
- **XGBoost:** `process(x)` → `vib_*` features, join thermal/context from
  `metadata`, predict 3 regressors + `predict_proba`.
- **waveform (1dcnn):** `get_normalized_waveform` → `(1,1,L)` → `ONNXPerceptor.predict`.
- **spectrogram:** `compute_spectrogram` → `(1,1,F,T)` (F=513 fixed, T dynamic).
- **fusion:** waveform `(1,1,L)` + thermal `(1,dim)`.

`stream_from_simulator` truly uses `simulator.stream()` for the chunks and fetches
duration-independent operating-point context once via a short `generate()` call
(TPF / kinematics don't depend on the streamed window); with a `thermal_simulator`
it adds observable thermal stats (`mean/max/temp_rise` + `therm_std_c`/
`therm_slope_c_per_s` via `np.polyfit`) so the XGBoost/fusion context is complete.

Also exported: `available_models(model_dir=None)` (registry annotated with on-disk
artifact availability), `resolve_model_name`, `REGRESSION_TARGETS`,
`HEALTH_CLASS_NAMES`.

---

## 3. `InferenceLogger` (`app/logging.py`, new, v0.1.0)

Typed **`LoggerConfig`** (`extra="forbid"`): `enabled`, `log_dir="logs/inference"`,
`flush_every=50`, `flush_interval_s=30`, `partition_cols=["date","model"]`,
`include_features=True`, `write_manifest=True`, `compression="snappy"`.

`log_prediction(result)` flattens a payload into a row (timestamp, chunk id,
model provenance, `pred_*`, `health_prob_<class>` × 4, `latency_ms`, echoed
operating-point scalars, and `vib_*` DSP features when enabled), buffers it, and
flushes on the count/time threshold. `flush()` writes the buffered rows to
`records/` via `pq.write_to_dataset` (partitioned, `overwrite_or_ignore`) and
rewrites the feature-free scalar `manifest.parquet` atomically (`.tmp` →
`replace`). Thread-safe. Convenience `read_logs(log_dir)` reads the partitioned
`records/` back into a DataFrame (Hive partition columns restored). `close()` /
context-manager flush the tail.

---

## 4. FastAPI service (`app/main.py` + `app/config.py`, new)

- **`app/config.py` — `AppConfig`** (pydantic v2, `extra="forbid"`,
  `protected_namespaces=()`). `AppConfig.from_env(**overrides)` reads `ARGUS_*`
  env vars (no `pydantic-settings` dependency): `ARGUS_DEFAULT_MODEL`,
  `ARGUS_PRELOAD_MODELS` (CSV), `ARGUS_MODEL_DIR`, `ARGUS_FS_HZ`, `ARGUS_CHUNK_S`,
  `ARGUS_LOG_ENABLED`, `ARGUS_LOG_DIR`, `ARGUS_LOG_FLUSH_EVERY`,
  `ARGUS_CORS_ORIGINS` (CSV).
- **Schemas** (all `extra="forbid"` except the passthrough response models):
  `InferenceRequest` (`model?`, `vibration?`, `thermal?`, `fs_hz?`, `params?`,
  `metadata?`, `return_features`), `BatchRequest` (`model?`, `items[]`),
  `Predictions`, `Recommendations`, `InferenceResponse`, `BatchResponse`,
  `HealthResponse`.
- **`PerceptorPool`** — thread-safe cache of one `StreamingPerceptor` per model,
  all sharing a single `InferenceLogger` so every model's predictions land in the
  same dataset. `get(model)` lazily builds + `.load()`s on first use.
- **Endpoints:** `POST /infer` (body or `?model=` query override), `POST /batch`,
  `GET /health` (status + version + loaded/available models + `n_logged`),
  `GET /models` (registry + artifact availability + aliases). DL models without a
  `vibration` array → **422**; unknown model → **404**; missing artifact → **503**;
  bad chunk (too short / non-finite) → **422**.
- **Lifespan** pre-loads `preload_models` on startup (low first-request latency)
  and flushes the logger on shutdown; **CORS** middleware from `cors_origins`.
- Module-level `app = create_app()` so `uvicorn app.main:app` just works.

`app/__init__.py` bumped to **0.1.0** with lazy `__getattr__` exports
(`create_app`, `app`, `InferenceLogger`, `LoggerConfig`, `StreamingPerceptor`).

---

## 5. Live demo (`scripts/stream_demo.py`, new)

End-to-end without HTTP: start a simulator → `stream_from_simulator` → DSP +
model → Parquet log → printed structured payloads. Flags: `--model`,
`--duration-s`, `--chunk-s`, `--wear`, `--seed`, `--with-thermal`, `--log-dir`,
`--no-log`, `--show-features`. Sample run (`--model xgboost --wear 0.9
--with-thermal`, 2×1 s chunks):

```
chunk 0 | health=critical conf=0.73 | wear=0.868 cycle=1.405 quality=0.564 | anomaly=True | 48.4 ms | -> stop_replace_blade
chunk 1 | health=critical conf=0.66 | wear=0.867 cycle=1.398 quality=0.574 | anomaly=True | 26.6 ms | -> stop_replace_blade
```

The physics tracks: injected wear=0.9 → predicted wear ≈0.87, `health=critical`,
`anomaly=True`, action `stop_replace_blade`.

---

## 6. Testing & verification

`tests/test_streaming_perceptor.py` (new, **32 tests**, graceful skips when
onnxruntime / xgboost / fastapi or the artifacts are missing):

- **Registry/config:** alias resolution, unknown-name `ValueError`, per-variant
  `normalize_for_dl`, `available_models` availability, `StreamingConfig` rejects
  unknown keys.
- **ONNX waveform path:** structured + finite output, ranges/prob-sum, DSP reuse
  (`td_`/`fd_` features present), provenance + recommendations.
- **Ring buffer:** `ingest` → `process_next_chunk` returns `None` under a chunk,
  drains exactly `chunk_samples` when full, `get_latest_prediction` identity;
  non-finite `ingest` and sub-`min_chunk_samples` chunk both raise.
- **Streaming:** `stream_from_simulator` yields ≥2 results with monotonic ids.
- **XGBoost path + switching:** `model_kind=="xgboost"`, prob-sum, high-wear sanity.
- **Fusion:** default zero thermal runs; explicit prepared thermal vector runs & stays finite.
- **Logging:** partitioned round-trip (`model=1dcnn_normnone` partition dir,
  `vib_*` cols, `read_logs` ≥3 rows, manifest length matches & is feature-free);
  disabled logger is a no-op.
- **FastAPI (`TestClient`, lifespan-triggered preload):** `/health` (default
  preloaded), `/models`, `/infer`, `/batch` (count=2), 422 for DL-without-vibration,
  404 for unknown model.

Results:

```
$ python -m pytest tests/ -q --tb=no
86 passed, 1 warning in 6.07s          # 54 prior + 32 new; warning is the benign
                                       # StarletteDeprecationWarning (httpx TestClient)
$ python -m ruff check models/streaming_perceptor.py app/ scripts/stream_demo.py \
      tests/test_streaming_perceptor.py
All checks passed!
$ python -c "from app.main import app; print(sorted({r.path for r in app.routes if getattr(r,'methods',None)}))"
['/batch', '/docs', '/docs/oauth2-redirect', '/health', '/infer', '/models', '/openapi.json', '/redoc']
```

Existing suites (simulators, DSP, baseline, DL) are untouched and still pass.

---

## 7. Bugs / tuning encountered and resolved

1. **Pydantic protected-namespace warnings.** Fields like `model_dir`,
   `model_kind`, `model_variant`, `model_version` collide with pydantic v2's
   reserved `model_` namespace (noisy `UserWarning`). Set
   `protected_namespaces=()` on the affected models (`StreamingConfig`,
   `AppConfig`, `InferenceRequest`/`BatchRequest`, `InferenceResponse`) — clean.
2. **Unknown-model → 500.** `resolve_model_name`'s `ValueError` initially bubbled
   up as a 500. Wrapped `pool.get` in `_infer_one` to map `ValueError` → **404**
   and `FileNotFoundError` (known model, missing artifact) → **503**; updated the
   test accordingly.
3. **Fusion thermal standardization (documented limitation).** The fusion models
   were trained on *standardized* thermal scalars, but those per-dataset mean/std
   stats aren't persisted with the ONNX artifact. Passing **raw** thermal values
   (e.g. 300 °C) lands far outside the standardized range and yields degenerate
   output (observed `wear=0.0`). Resolution: `infer_chunk` defaults the thermal
   branch to a **neutral zero vector** (the standardized dataset mean), and callers
   may pass an already-prepared `thermal` vector to override. The waveform-only
   (`1dcnn*`) and `xgboost` paths need no such prep and are the recommended
   streaming defaults. Documented in `app/README.md`.
4. **`TestClient` lifespan.** Startup preload only fires when the client is used
   as a context manager; the `client` fixture uses `with TestClient(app) as c:` so
   `/health` reports the preloaded model.
5. **OpenMP duplicate-runtime notice.** Same pre-existing Anaconda
   `libiomp5md.dll` conflict as Day 3 when torch/xgboost co-import; unrelated to
   this layer (the service is torch-free). `KMP_DUPLICATE_LIB_OK=TRUE` remains the
   workaround and was used for the verification runs.

---

## 8. Documentation & repo updates

- `pyproject.toml`: new `[project.optional-dependencies] app = [fastapi>=0.110,
  uvicorn[standard]>=0.29, python-multipart>=0.0.9, httpx>=0.27]`;
  `requirements.txt` updated with the `app` extra + `pip install -e ".[ml,dl,app]"`
  serving note.
- `.gitignore`: ignore runtime `/logs/inference/` (tracked `logs/*.md` session
  logs are unaffected; `*.parquet` was already ignored globally).
- `app/README.md`: rewritten "scaffold" → **"Day 4 complete"** — install, run
  (`uvicorn app.main:app --reload`), `ARGUS_*` config table, `curl` examples,
  model-name list, the streaming demo, log-query snippets, and the fusion-thermal
  note.
- Root `README.md`: status table (`app/` ✅ Day 4), quickstart step 7
  (streaming demo + uvicorn + `curl`), a "simulator → perceptor → API" snippet,
  repo layout (`streaming_perceptor.py`, `app/main.py`/`logging.py`/`config.py`,
  `scripts/stream_demo.py`), tech-stack `[app]` note, and roadmap (Day 4 ✅).

---

## 9. Verification snapshot (commands + key stdout)

```
$ python -m pip install "fastapi>=0.110" "uvicorn[standard]>=0.29" \
      "python-multipart>=0.0.9" "httpx>=0.27"
Successfully installed fastapi-0.139.0 uvicorn-0.50.2 starlette-1.3.1 python-multipart-0.0.32 ...

$ python -m pytest tests/ -q --tb=no
86 passed, 1 warning in 6.07s

# programmatic smoke test
$ python -c "from sensors.vibration_simulator import SawVibrationSimulator; \
    from models.streaming_perceptor import StreamingPerceptor; \
    p=StreamingPerceptor(model='1dcnn_normnone', chunk_s=1.0); s=SawVibrationSimulator(); \
    [None for _ in p.stream_from_simulator(s, duration_s=2.0, wear=0.5, seed=0)]; \
    print('StreamingPerceptor smoke test passed')"
StreamingPerceptor smoke test passed

# FastAPI via TestClient
health: {'status': 'ok', 'default_model': '1dcnn_normnone', 'n_available_models': 10, ...}
infer status 200 -> 1dcnn_normnone warning wear=0.759 latency 26.1 ms
batch status 200 count 2 ; missing vibration -> 422 ; unknown model -> 404

$ python scripts/stream_demo.py --model xgboost --duration-s 2 --wear 0.9 --with-thermal
chunk 0 health=critical wear=0.868 anomaly=True -> stop_replace_blade
StreamingPerceptor smoke test passed

$ python -c "from app.main import app; print(app.title, 'v'+app.version)"
Argus Panoptes Perception API v0.1.0
```

---

## 10. Immediate next steps / TODOs for Day 5

1. **Streamlit dashboard** — live waveform / FFT / STFT plots, KPI gauges
   (wear %, RUL, cycle factor, anomaly score), alerts, a historical explorer over
   the Parquet inference logs, and a model/variant toggle driving `/infer`.
2. **Cost / nesting integration mock** — consume the `recommendations` +
   `cycle_time_factor` payload to demo downstream job-costing / nesting.
3. **Persist fusion thermal standardization stats** with (or alongside) the ONNX
   artifact so the fusion streaming path can accept raw thermal scalars end-to-end.
4. **Mock PLC / OPC-UA tags** feeding the perceptor's `ingest` for a realistic
   push-stream demo; consider a WebSocket `/stream` endpoint.
5. **Edge packaging** — carry the Day-3 TensorRT/OpenVINO notes into a Docker
   image (Day 6) that serves this API.

Session complete — streaming perceptor + Parquet logging + FastAPI ready for the
Day-5 dashboard and cost/nesting integration.
