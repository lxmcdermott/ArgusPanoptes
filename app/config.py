"""Environment-driven configuration for the Argus Panoptes FastAPI service.

Kept dependency-light (no ``pydantic-settings``): a small Pydantic v2 model whose
:meth:`AppConfig.from_env` reads ``ARGUS_*`` environment variables. Mirrors the
config style of :mod:`sensors.config` / :mod:`dsp.config` (typed,
``extra="forbid"``, versioned).
"""

from __future__ import annotations

import os
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

__all__ = ["AppConfig"]

_ENV_PREFIX = "ARGUS_"


class AppConfig(BaseModel):
    """Typed FastAPI-service configuration."""

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    app_version: str = "0.1.0"
    #: Default model when a request omits one (see MODEL_REGISTRY).
    default_model: str = "1dcnn_normnone"
    #: Models to pre-load on startup for low first-request latency.
    preload_models: list[str] = Field(default_factory=lambda: ["1dcnn_normnone"])
    #: Directory holding the ONNX / joblib artifacts.
    model_dir: str | None = None
    fs_hz: float = Field(default=40960.0, gt=0)
    chunk_s: float = Field(default=1.0, gt=0)
    #: Persist predictions to Parquet.
    log_enabled: bool = True
    log_dir: str = "logs/inference"
    log_flush_every: int = Field(default=50, ge=1)
    #: CORS allowed origins (``["*"]`` for any).
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])

    @classmethod
    def from_env(cls, **overrides: Any) -> "AppConfig":
        """Build config from ``ARGUS_*`` env vars, with explicit overrides on top."""
        env = os.environ

        def _get(name: str) -> str | None:
            return env.get(_ENV_PREFIX + name)

        data: dict[str, Any] = {}
        if (v := _get("DEFAULT_MODEL")) is not None:
            data["default_model"] = v
        if (v := _get("PRELOAD_MODELS")) is not None:
            data["preload_models"] = [s.strip() for s in v.split(",") if s.strip()]
        if (v := _get("MODEL_DIR")) is not None:
            data["model_dir"] = v
        if (v := _get("FS_HZ")) is not None:
            data["fs_hz"] = float(v)
        if (v := _get("CHUNK_S")) is not None:
            data["chunk_s"] = float(v)
        if (v := _get("LOG_ENABLED")) is not None:
            data["log_enabled"] = v.strip().lower() in ("1", "true", "yes", "on")
        if (v := _get("LOG_DIR")) is not None:
            data["log_dir"] = v
        if (v := _get("LOG_FLUSH_EVERY")) is not None:
            data["log_flush_every"] = int(v)
        if (v := _get("CORS_ORIGINS")) is not None:
            data["cors_origins"] = [s.strip() for s in v.split(",") if s.strip()]
        data.update(overrides)
        return cls.model_validate(data)
