"""FastAPI inference service for Argus Panoptes (Day 4 integration layer).

Wraps :class:`models.streaming_perceptor.StreamingPerceptor` behind a small,
production-shaped HTTP API that turns raw vibration chunks (or, for XGBoost,
operating-point context) into structured JSON payloads for the downstream
cost / cycle-time / nesting optimizer. Every prediction is optionally persisted
to partitioned Parquet via :class:`app.logging.InferenceLogger`.

Endpoints
---------
* ``POST /infer``  - single-chunk inference -> :class:`InferenceResponse`.
* ``POST /batch``  - many chunks -> :class:`BatchResponse`.
* ``GET  /health`` - liveness + loaded/available model summary.
* ``GET  /models`` - selectable models / variants and artifact availability.

Model selection is by friendly name (``model`` in the body or the ``model`` query
param), mapping to XGBoost or a specific ONNX variant. Perceptors are pooled and
loaded lazily (default model[s] pre-loaded on startup for low latency).

Run
---
    uvicorn app.main:app --reload
"""

from __future__ import annotations

import logging
import threading
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field

from app.config import AppConfig
from app.logging import InferenceLogger
from models.streaming_perceptor import (
    MODEL_REGISTRY,
    StreamingPerceptor,
    available_models,
    resolve_model_name,
)

__all__ = ["app", "create_app", "PerceptorPool"]

logger = logging.getLogger("argus.app")


# --------------------------------------------------------------------------- #
# Request / response schemas
# --------------------------------------------------------------------------- #
class InferenceRequest(BaseModel):
    """One inference request: a vibration chunk (+ optional thermal / context)."""

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    model: str | None = Field(default=None, description="Friendly model name; default if omitted.")
    vibration: list[float] | None = Field(
        default=None, description="Raw 1-D vibration chunk (g). Required for DL models."
    )
    thermal: list[float] | None = Field(
        default=None,
        description="Optional prepared thermal feature vector for the fusion model.",
    )
    fs_hz: float | None = Field(default=None, gt=0, description="Sample rate (Hz).")
    params: dict[str, Any] | None = Field(
        default=None, description="Operating-point context (alloy, kinematics, thermal stats)."
    )
    metadata: dict[str, Any] | None = Field(default=None, description="Extra metadata to echo/log.")
    return_features: bool = Field(default=False, description="Include DSP features in the response.")


class BatchRequest(BaseModel):
    """A batch of inference requests sharing an optional default model."""

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    model: str | None = Field(default=None, description="Default model for items lacking one.")
    items: list[InferenceRequest] = Field(..., min_length=1)


class Predictions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    wear_level: float
    cycle_time_factor: float
    quality_score: float
    health_state: str
    health_probs: dict[str, float]
    anomaly_flag: bool
    confidence: float


class Recommendations(BaseModel):
    model_config = ConfigDict(extra="allow")

    action: str
    blade_change_suggested: bool
    cycle_time_factor: float
    quality_score: float
    note: str


class InferenceResponse(BaseModel):
    model_config = ConfigDict(extra="allow", protected_namespaces=())

    chunk_id: int
    timestamp: str
    model: str
    model_kind: str
    model_variant: str
    model_version: str
    fs_hz: float
    n_samples: int
    latency_ms: float
    predictions: Predictions
    recommendations: Recommendations
    metadata: dict[str, Any]
    features: dict[str, float] = Field(default_factory=dict)


class BatchResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    count: int
    results: list[InferenceResponse]


class HealthResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: str
    app_version: str
    perceptor_version: str
    default_model: str
    loaded_models: list[str]
    n_available_models: int
    log_enabled: bool
    n_logged: int


# --------------------------------------------------------------------------- #
# Perceptor pool (lazy, cached per model, shared logger)
# --------------------------------------------------------------------------- #
class PerceptorPool:
    """Thread-safe cache of :class:`StreamingPerceptor` instances keyed by model.

    All perceptors share one :class:`InferenceLogger` so predictions from every
    model land in the same partitioned dataset.
    """

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self._perceptors: dict[str, StreamingPerceptor] = {}
        self._lock = threading.Lock()
        self.logger: InferenceLogger | None = (
            InferenceLogger(
                {
                    "enabled": True,
                    "log_dir": config.log_dir,
                    "flush_every": config.log_flush_every,
                }
            )
            if config.log_enabled
            else None
        )

    def get(self, model: str | None) -> StreamingPerceptor:
        name = resolve_model_name(model or self.config.default_model)
        with self._lock:
            perc = self._perceptors.get(name)
            if perc is None:
                perc = StreamingPerceptor(
                    model=name,
                    fs_hz=self.config.fs_hz,
                    chunk_s=self.config.chunk_s,
                    model_dir=self.config.model_dir,
                    logger=self.logger,
                )
                perc.load()
                self._perceptors[name] = perc
            return perc

    def loaded_models(self) -> list[str]:
        with self._lock:
            return sorted(self._perceptors)

    def close(self) -> None:
        if self.logger is not None:
            self.logger.close()


# --------------------------------------------------------------------------- #
# App factory
# --------------------------------------------------------------------------- #
def create_app(config: AppConfig | None = None) -> FastAPI:
    """Build the FastAPI application (pre-loading the configured models)."""
    cfg = config or AppConfig.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        pool: PerceptorPool = app.state.pool
        for name in cfg.preload_models:
            try:
                pool.get(name)
                logger.info("Pre-loaded model %s", name)
            except Exception as exc:  # pragma: no cover - artifact-availability dependent
                logger.warning("Could not pre-load model %s: %s", name, exc)
        yield
        pool.close()

    app = FastAPI(
        title="Argus Panoptes Perception API",
        version=cfg.app_version,
        description="Streaming industrial perception: vibration/thermal -> structured predictions.",
        lifespan=lifespan,
    )
    app.state.config = cfg
    app.state.pool = PerceptorPool(cfg)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=cfg.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    def _infer_one(pool: PerceptorPool, req: InferenceRequest, default_model: str | None) -> dict:
        try:
            perc = pool.get(req.model or default_model)
        except ValueError as exc:  # unknown model name
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except FileNotFoundError as exc:  # known model, missing artifact
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        if perc.spec.kind != "xgboost" and not req.vibration:
            raise HTTPException(
                status_code=422,
                detail=f"model {perc.model_name!r} ({perc.spec.kind}) requires a 'vibration' array.",
            )
        vibration = req.vibration if req.vibration is not None else []
        try:
            return perc.infer_chunk(
                vibration,
                thermal=req.thermal,
                metadata=req.metadata,
                params=req.params,
                return_features=req.return_features,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        pool: PerceptorPool = app.state.pool
        return HealthResponse(
            status="ok",
            app_version=cfg.app_version,
            perceptor_version=StreamingPerceptor.version,
            default_model=resolve_model_name(cfg.default_model),
            loaded_models=pool.loaded_models(),
            n_available_models=sum(1 for m in available_models(cfg.model_dir) if m["available"]),
            log_enabled=pool.logger is not None,
            n_logged=pool.logger.n_logged if pool.logger is not None else 0,
        )

    @app.get("/models")
    def models() -> dict[str, Any]:
        return {
            "default_model": resolve_model_name(cfg.default_model),
            "models": available_models(cfg.model_dir),
            "aliases": sorted(set(MODEL_REGISTRY)),
        }

    @app.post("/infer", response_model=InferenceResponse)
    def infer(
        req: InferenceRequest,
        model: str | None = Query(default=None, description="Overrides body 'model'."),
    ) -> dict:
        pool: PerceptorPool = app.state.pool
        return _infer_one(pool, req, model or cfg.default_model)

    @app.post("/batch", response_model=BatchResponse)
    def batch(req: BatchRequest) -> BatchResponse:
        pool: PerceptorPool = app.state.pool
        results = [_infer_one(pool, item, req.model or cfg.default_model) for item in req.items]
        return BatchResponse(count=len(results), results=results)  # type: ignore[arg-type]

    return app


#: Module-level ASGI app for ``uvicorn app.main:app``.
app = create_app()
