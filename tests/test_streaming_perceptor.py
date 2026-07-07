"""Tests for the Day-4 integration layer.

Covers :class:`models.streaming_perceptor.StreamingPerceptor` (ingest / chunked
processing, DSP + ONNX / XGBoost reuse, streaming from a simulator, edge cases
and output finiteness), :class:`app.logging.InferenceLogger` (partitioned Parquet
round-trip), and the FastAPI service (:mod:`app.main`) via ``TestClient``.

DL / ONNX tests are skipped when ``onnxruntime`` (or the ONNX artifacts) are
missing; XGBoost tests when the ``ml`` extra / joblib artifacts are missing; API
tests when ``fastapi`` is missing - so the suite degrades gracefully.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest

from models.streaming_perceptor import (
    MODEL_REGISTRY,
    StreamingConfig,
    StreamingPerceptor,
    available_models,
    resolve_model_name,
)

_MODEL_DIR = Path(__file__).resolve().parents[1] / "experiments" / "models"
_ONNX_1DCNN = _MODEL_DIR / "dl_1dcnn_normnone.onnx"
_XGB_CLF = _MODEL_DIR / "xgb_clf_health_state.joblib"

_HAS_ONNX = pytest.importorskip  # alias for readability in guards below


def _onnx_available() -> bool:
    try:
        import onnxruntime  # noqa: F401
    except Exception:
        return False
    return _ONNX_1DCNN.is_file()


def _xgb_available() -> bool:
    try:
        import joblib  # noqa: F401
        import xgboost  # noqa: F401
    except Exception:
        return False
    return _XGB_CLF.is_file()


requires_onnx = pytest.mark.skipif(not _onnx_available(), reason="onnxruntime / ONNX artifact missing")
requires_xgb = pytest.mark.skipif(not _xgb_available(), reason="xgboost / joblib artifact missing")


@pytest.fixture()
def chunk(vib):
    """A 1 s wear=0.6 vibration chunk + its simulator metadata."""
    _, accel, meta = vib.generate(duration_s=1.0, wear=0.6, seed=0)
    return accel, meta


# --------------------------------------------------------------------------- #
# Registry / config
# --------------------------------------------------------------------------- #
def test_registry_and_aliases():
    assert resolve_model_name("xgb") == "xgboost"
    assert resolve_model_name("cnn1d") == "1dcnn"
    assert resolve_model_name("FUSION_normnone") == "fusion_normnone"
    with pytest.raises(ValueError):
        resolve_model_name("nope")
    assert "fusion_normnone" in MODEL_REGISTRY
    assert MODEL_REGISTRY["fusion_normnone"].normalize_for_dl == "none"
    assert MODEL_REGISTRY["1dcnn_noisy"].normalize_for_dl == "zscore"


def test_available_models_reports_availability():
    entries = {m["name"]: m for m in available_models()}
    assert set(entries) == set(MODEL_REGISTRY)
    # The committed artifacts should be present for the showcased variants.
    if _onnx_available():
        assert entries["1dcnn_normnone"]["available"]
    if _xgb_available():
        assert entries["xgboost"]["available"]


def test_config_forbids_unknown_keys():
    with pytest.raises(Exception):
        StreamingConfig.model_validate({"bogus_key": 1})
    cfg = StreamingConfig(model="1dcnn", chunk_s=0.5, fs_hz=40960.0)
    assert cfg.chunk_s == 0.5


# --------------------------------------------------------------------------- #
# StreamingPerceptor - ONNX (waveform) path
# --------------------------------------------------------------------------- #
@requires_onnx
def test_infer_chunk_waveform_finite_and_structured(chunk):
    accel, meta = chunk
    perc = StreamingPerceptor(model="1dcnn_normnone", chunk_s=1.0).load()
    res = perc.infer_chunk(accel, metadata=meta, return_features=True)

    p = res["predictions"]
    assert 0.0 <= p["wear_level"] <= 1.0
    assert p["cycle_time_factor"] >= 0.0
    assert 0.0 <= p["quality_score"] <= 1.0
    assert p["health_state"] in ("critical", "healthy", "monitor", "warning")
    assert set(p["health_probs"]) == {"critical", "healthy", "monitor", "warning"}
    assert np.isclose(sum(p["health_probs"].values()), 1.0, atol=1e-5)
    assert 0.0 <= p["confidence"] <= 1.0
    assert isinstance(p["anomaly_flag"], bool)
    # All numeric outputs finite.
    assert np.isfinite(p["wear_level"]) and np.isfinite(res["latency_ms"])
    # DSP reuse: time/frequency features present with the documented prefixes.
    assert any(k.startswith("td_") for k in res["features"])
    assert any(k.startswith("fd_") for k in res["features"])
    # Provenance + downstream payload.
    assert res["model"] == "1dcnn_normnone"
    assert res["recommendations"]["action"] in {
        "continue", "monitor_schedule_inspection",
        "reduce_feed_plan_blade_change", "stop_replace_blade",
    }


@requires_onnx
def test_ingest_and_process_next_chunk(vib):
    perc = StreamingPerceptor(model="1dcnn_normnone", chunk_s=0.5)
    _, accel, meta = vib.generate(duration_s=1.0, wear=0.3, seed=1)
    # Not enough buffered yet -> None.
    perc.ingest(accel[: perc.chunk_samples // 2], metadata=meta)
    assert perc.process_next_chunk() is None
    # Top up past a full chunk -> one result, buffer drained by chunk_samples.
    perc.ingest(accel[perc.chunk_samples // 2 :])
    before = perc.buffered_samples()
    res = perc.process_next_chunk()
    assert res is not None
    assert perc.buffered_samples() == before - perc.chunk_samples
    assert perc.get_latest_prediction() is res


@requires_onnx
def test_stream_from_simulator_yields_predictions(vib):
    perc = StreamingPerceptor(model="1dcnn_normnone", chunk_s=1.0)
    results = list(perc.stream_from_simulator(vib, duration_s=2.0, chunk_s=1.0, wear=0.7, seed=0))
    assert len(results) >= 2
    assert all(r["predictions"]["health_state"] in
               {"critical", "healthy", "monitor", "warning"} for r in results)
    # chunk ids are monotonic from 0.
    assert [r["chunk_id"] for r in results][:2] == [0, 1]


@requires_onnx
def test_ingest_rejects_non_finite(vib):
    perc = StreamingPerceptor(model="1dcnn_normnone")
    with pytest.raises(ValueError):
        perc.ingest([0.1, np.nan, 0.2])


@requires_onnx
def test_short_chunk_rejected(vib):
    perc = StreamingPerceptor(model="1dcnn_normnone")
    with pytest.raises(ValueError):
        perc.infer_chunk(np.zeros(perc.config.min_chunk_samples - 1))


# --------------------------------------------------------------------------- #
# StreamingPerceptor - XGBoost (feature) path + model switching
# --------------------------------------------------------------------------- #
@requires_xgb
def test_xgboost_path_and_switching(vib, therm):
    _, accel, meta = vib.generate(duration_s=1.0, wear=0.9, seed=2)
    perc = StreamingPerceptor(model="xgboost", chunk_s=1.0).load()
    res = perc.infer_chunk(accel, metadata=meta, return_features=True)
    p = res["predictions"]
    assert res["model_kind"] == "xgboost"
    assert 0.0 <= p["wear_level"] <= 1.0
    assert np.isclose(sum(p["health_probs"].values()), 1.0, atol=1e-5)
    # High injected wear should push wear_level up (sanity, not exact).
    assert p["wear_level"] > 0.4


# --------------------------------------------------------------------------- #
# StreamingPerceptor - fusion path (thermal branch)
# --------------------------------------------------------------------------- #
@requires_onnx
def test_fusion_defaults_thermal_to_zeros(vib):
    if not (_MODEL_DIR / "dl_fusion_normnone.onnx").is_file():
        pytest.skip("fusion artifact missing")
    _, accel, meta = vib.generate(duration_s=1.0, wear=0.5, seed=3)
    perc = StreamingPerceptor(model="fusion_normnone", chunk_s=1.0).load()
    res = perc.infer_chunk(accel, metadata=meta)  # no thermal -> neutral zeros
    assert res["predictions"]["health_state"] in {"critical", "healthy", "monitor", "warning"}
    # Explicit prepared thermal vector also runs and stays finite.
    res2 = perc.infer_chunk(accel, metadata=meta, thermal=[0.1, -0.2, 0.0, 0.3, -0.1])
    assert np.isfinite(res2["predictions"]["wear_level"])


# --------------------------------------------------------------------------- #
# InferenceLogger - partitioned Parquet round-trip
# --------------------------------------------------------------------------- #
@requires_onnx
def test_inference_logger_partitioned_parquet(vib, tmp_path):
    from app.logging import InferenceLogger, read_logs

    logger = InferenceLogger({"log_dir": str(tmp_path), "flush_every": 2, "include_features": True})
    perc = StreamingPerceptor(model="1dcnn_normnone", chunk_s=1.0, logger=logger).load()
    for r in perc.stream_from_simulator(vib, duration_s=3.0, chunk_s=1.0, wear=0.8, seed=4,
                                        return_features=True):
        pass
    perc.close()

    # Partitioned records exist (Hive: date=.../model=...).
    records = tmp_path / "records"
    assert records.exists()
    parts = list(records.rglob("*.parquet"))
    assert parts, "no parquet record files written"
    assert any("model=1dcnn_normnone" in str(p).replace("\\", "/") for p in parts)

    df = read_logs(tmp_path)
    assert len(df) >= 3
    for col in ("timestamp", "model", "date", "pred_wear_level", "pred_health_state",
                "health_prob_healthy", "latency_ms"):
        assert col in df.columns
    assert any(c.startswith("vib_") for c in df.columns)
    # Manifest (feature-free scalar summary) is written and readable.
    import pandas as pd

    manifest = pd.read_parquet(tmp_path / "manifest.parquet")
    assert len(manifest) == len(df)
    assert not any(c.startswith("vib_") for c in manifest.columns)


def test_logger_disabled_is_noop(tmp_path):
    from app.logging import InferenceLogger

    logger = InferenceLogger({"enabled": False, "log_dir": str(tmp_path)})
    logger.log_prediction({"predictions": {}, "metadata": {}, "recommendations": {}})
    assert logger.flush() == 0
    assert not (tmp_path / "records").exists()


# --------------------------------------------------------------------------- #
# FastAPI service
# --------------------------------------------------------------------------- #
@pytest.fixture()
def client():
    pytest.importorskip("fastapi")
    if not _onnx_available():
        pytest.skip("ONNX artifact required for default model preload")
    from fastapi.testclient import TestClient

    from app.config import AppConfig
    from app.main import create_app

    cfg = AppConfig.from_env(log_enabled=False, preload_models=["1dcnn_normnone"],
                             default_model="1dcnn_normnone")
    app = create_app(cfg)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with TestClient(app) as c:
            yield c


def test_health_and_models(client):
    h = client.get("/health").json()
    assert h["status"] == "ok"
    assert h["default_model"] == "1dcnn_normnone"
    assert "1dcnn_normnone" in h["loaded_models"]  # preloaded on startup
    m = client.get("/models").json()
    assert m["default_model"] == "1dcnn_normnone"
    assert any(e["name"] == "xgboost" for e in m["models"])


def test_infer_and_batch_endpoints(client, vib):
    _, accel, meta = vib.generate(duration_s=1.0, wear=0.8, seed=5)
    body = {"model": "1dcnn_normnone", "vibration": accel.tolist(), "fs_hz": float(meta["fs_hz"])}
    r = client.post("/infer", json=body)
    assert r.status_code == 200
    j = r.json()
    assert j["model"] == "1dcnn_normnone"
    assert 0.0 <= j["predictions"]["wear_level"] <= 1.0
    assert j["latency_ms"] >= 0.0

    rb = client.post("/batch", json={"model": "1dcnn_normnone", "items": [body, body]})
    assert rb.status_code == 200
    assert rb.json()["count"] == 2


def test_infer_requires_vibration_for_dl(client):
    r = client.post("/infer", json={"model": "1dcnn_normnone"})
    assert r.status_code == 422


def test_infer_unknown_model_rejected(client, vib):
    _, accel, _ = vib.generate(duration_s=1.0, wear=0.1, seed=6)
    r = client.post("/infer", json={"model": "does_not_exist", "vibration": accel.tolist()})
    assert r.status_code == 404
