# `deployment/` — Docker & Edge

> **Status:** scaffold (implemented in Day 6 of the execution plan).

Contains the containerization and edge-deployment assets for the full stack.

## Planned scope (per technical plan §7)

- `Dockerfile` + `docker-compose.yml` for the FastAPI API + Streamlit dashboard.
- ONNX Runtime benchmarks (CPU; notes for Jetson / TensorRT / OpenVINO).
- Monitoring hooks and retraining stubs.

The placeholder `Dockerfile` and `docker-compose.yml` here are non-functional
stubs to be fleshed out once `app/` exists.
