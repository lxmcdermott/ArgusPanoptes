# `app/` — Inference, Integration & Dashboard

> **Status:** scaffold (implemented in Day 4–5 of the execution plan).

## Planned scope (per technical plan §5–6)

- **`StreamingPerceptor`:** real-time chunk processing over a ring buffer.
- **FastAPI service:** `/infer`, `/batch` endpoints returning structured JSON
  payloads for WAYNE (cycle-time factor, quality score, recommendations).
- **Mock integrations:** PLC tags, ML feature store.
- **Streamlit dashboard:** live simulation controls (param/wear sliders),
  real-time waveform / FFT / STFT / thermal plots, KPI gauges (wear %, RUL,
  cycle factor, anomaly score), alerts, historical explorer, optimization lab,
  and a PyTorch↔ONNX model toggle.
