"""Integration & UI layer for Argus Panoptes.

Day 4 (implemented): a **FastAPI service** (`/infer`, `/batch`, `/health`,
`/models`) emitting structured JSON payloads for the downstream cost / nesting
optimizer, a :class:`~models.streaming_perceptor.StreamingPerceptor` for
real-time chunk processing over a ring buffer, and an
:class:`~app.logging.InferenceLogger` that persists predictions to partitioned
Parquet for ops / retraining.

Planned (Day 5): mock PLC / OPC-UA tags and a Streamlit dashboard for live
simulation control and monitoring.

Public API
----------
>>> from app import create_app          # FastAPI factory
>>> from app import InferenceLogger      # Parquet logging
>>> from app import StreamingPerceptor   # re-exported for convenience

Attributes are imported **lazily** (via ``__getattr__``) so that importing
``app`` — or the torch-free ``app.logging`` — never eagerly pulls in FastAPI /
uvicorn. Import ``app.main`` (or use ``uvicorn app.main:app``) to get the ASGI
application.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

__version__ = "0.1.0"

__all__ = ["create_app", "app", "InferenceLogger", "LoggerConfig", "StreamingPerceptor", "__version__"]

if TYPE_CHECKING:  # pragma: no cover - typing only
    from app.logging import InferenceLogger, LoggerConfig
    from app.main import app, create_app
    from models.streaming_perceptor import StreamingPerceptor


def __getattr__(name: str) -> Any:
    """Lazily resolve heavy attributes to keep ``import app`` cheap."""
    if name in ("create_app", "app"):
        from app import main

        return getattr(main, name)
    if name in ("InferenceLogger", "LoggerConfig"):
        from app import logging as _logging

        return getattr(_logging, name)
    if name == "StreamingPerceptor":
        from models.streaming_perceptor import StreamingPerceptor

        return StreamingPerceptor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
