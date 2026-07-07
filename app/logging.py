"""Parquet logging of streaming inference results (Day 4 integration layer).

:class:`InferenceLogger` buffers structured prediction payloads (as produced by
:meth:`models.streaming_perceptor.StreamingPerceptor.infer_chunk`) in memory and
periodically flushes them to a **partitioned Parquet dataset**, mirroring the
``records/`` + ``manifest.parquet`` layout of ``scripts/generate_dataset.py`` so
the inference logs are queryable with the same tooling and are ready for
historical analysis / future retraining.

Layout (under ``log_dir``, default ``logs/inference/``)
-------------------------------------------------------
* ``records/`` - Hive-partitioned Parquet (default by ``date`` / ``model``) with
  the full row schema: timestamp, chunk id, model provenance, predictions +
  per-class health probabilities, latency, echoed operating-point metadata, and
  (optionally) the DSP scalar features.
* ``manifest.parquet`` - a lightweight, feature-free scalar summary of every
  logged prediction (fast to scan; rewritten on each flush).

Flushing is triggered by a count threshold (``flush_every``) or an elapsed-time
threshold (``flush_interval_s``); call :meth:`flush` / :meth:`close` to force a
write (e.g. on shutdown). The class is thread-safe so a FastAPI service can log
from concurrent requests.

Example
-------
>>> from app.logging import InferenceLogger
>>> logger = InferenceLogger({"log_dir": "logs/inference", "flush_every": 10})
>>> logger.log_prediction(result_dict)  # doctest: +SKIP
>>> logger.close()  # flush remaining buffer
"""

from __future__ import annotations

import threading
import time
import uuid
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

__version__ = "0.1.0"

__all__ = ["InferenceLogger", "LoggerConfig", "read_logs", "__version__"]

_HEALTH_CLASS_NAMES: tuple[str, ...] = ("critical", "healthy", "monitor", "warning")


class LoggerConfig(BaseModel):
    """Typed configuration for :class:`InferenceLogger` (forbids typos)."""

    model_config = ConfigDict(extra="forbid")

    logger_version: str = __version__
    enabled: bool = True
    log_dir: str = "logs/inference"
    #: Flush after this many buffered predictions.
    flush_every: int = Field(default=50, ge=1)
    #: Also flush when this many seconds have elapsed since the last flush
    #: (``0`` disables time-based flushing).
    flush_interval_s: float = Field(default=30.0, ge=0.0)
    #: Hive partition columns for ``records/`` (must be present on every row).
    partition_cols: list[str] = Field(default_factory=lambda: ["date", "model"])
    #: Persist the DSP scalar features (``vib_*``) alongside predictions.
    include_features: bool = True
    #: Rewrite the scalar ``manifest.parquet`` summary on flush.
    write_manifest: bool = True
    compression: str = "snappy"


class InferenceLogger:
    """Buffer + partitioned-Parquet writer for streaming inference results."""

    version: str = __version__

    def __init__(self, config: LoggerConfig | dict[str, Any] | str | None = None) -> None:
        self.config = self._coerce_config(config)
        self.log_dir = Path(self.config.log_dir)
        self.records_dir = self.log_dir / "records"
        self.manifest_path = self.log_dir / "manifest.parquet"

        self._buffer: list[dict[str, Any]] = []
        self._manifest_rows: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self._last_flush = time.monotonic()
        self._batch_idx = 0
        self._n_logged = 0

        if self.config.enabled:
            self.records_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _coerce_config(
        config: LoggerConfig | dict[str, Any] | str | None,
    ) -> LoggerConfig:
        if config is None:
            return LoggerConfig()
        if isinstance(config, LoggerConfig):
            return config
        if isinstance(config, dict):
            return LoggerConfig.model_validate(config)
        if isinstance(config, str):
            return LoggerConfig(log_dir=config)
        raise TypeError(  # pragma: no cover - defensive
            f"config must be LoggerConfig, dict, str, or None; got {type(config).__name__}"
        )

    # --------------------------------------------------------------- public API
    @property
    def n_logged(self) -> int:
        """Total predictions accepted by this logger (buffered + flushed)."""
        return self._n_logged

    def log_prediction(self, result: dict[str, Any]) -> None:
        """Buffer one structured prediction; flush if a threshold is crossed."""
        if not self.config.enabled:
            return
        row = self._flatten(result)
        do_flush = False
        with self._lock:
            self._buffer.append(row)
            self._manifest_rows.append({k: v for k, v in row.items() if not k.startswith("vib_")})
            self._n_logged += 1
            if len(self._buffer) >= self.config.flush_every:
                do_flush = True
            elif (
                self.config.flush_interval_s > 0
                and (time.monotonic() - self._last_flush) >= self.config.flush_interval_s
            ):
                do_flush = True
        if do_flush:
            self.flush()

    def flush(self) -> int:
        """Write buffered rows to the partitioned dataset; return rows written."""
        if not self.config.enabled:
            return 0
        with self._lock:
            rows = self._buffer
            self._buffer = []
            manifest_rows = list(self._manifest_rows)
            batch_idx = self._batch_idx
            self._batch_idx += 1
            self._last_flush = time.monotonic()
        if not rows:
            return 0
        self._write_records(rows, batch_idx)
        if self.config.write_manifest and manifest_rows:
            self._write_manifest(manifest_rows)
        return len(rows)

    def close(self) -> None:
        """Flush any remaining buffered rows (idempotent)."""
        self.flush()

    def __enter__(self) -> "InferenceLogger":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # ----------------------------------------------------------------- internals
    def _flatten(self, result: dict[str, Any]) -> dict[str, Any]:
        preds = result.get("predictions", {})
        meta = result.get("metadata", {})
        rec = result.get("recommendations", {})
        ts = str(result.get("timestamp", ""))
        health_probs = preds.get("health_probs", {}) or {}

        row: dict[str, Any] = {
            "timestamp": ts,
            "date": ts[:10] if ts else "",
            "chunk_id": result.get("chunk_id"),
            "model": result.get("model"),
            "model_kind": result.get("model_kind"),
            "model_variant": result.get("model_variant"),
            "model_version": result.get("model_version"),
            "fs_hz": result.get("fs_hz"),
            "n_samples": result.get("n_samples"),
            "latency_ms": result.get("latency_ms"),
            "pred_wear_level": preds.get("wear_level"),
            "pred_cycle_time_factor": preds.get("cycle_time_factor"),
            "pred_quality_score": preds.get("quality_score"),
            "pred_health_state": preds.get("health_state"),
            "pred_confidence": preds.get("confidence"),
            "pred_anomaly_flag": preds.get("anomaly_flag"),
            "action": rec.get("action"),
            "blade_change_suggested": rec.get("blade_change_suggested"),
            "alloy": meta.get("alloy"),
            "blade_speed_sfpm": meta.get("blade_speed_sfpm"),
            "num_teeth": meta.get("num_teeth"),
            "feed_per_tooth_mm": meta.get("feed_per_tooth_mm"),
            "depth_mm": meta.get("depth_mm"),
            "kerf_width_mm": meta.get("kerf_width_mm"),
            "tpf_hz": meta.get("tpf_hz"),
            "wear_injected": meta.get("wear_injected"),
        }
        for cls in _HEALTH_CLASS_NAMES:
            row[f"health_prob_{cls}"] = float(health_probs.get(cls, 0.0))
        if self.config.include_features:
            for k, v in (result.get("features", {}) or {}).items():
                row[f"vib_{k}"] = v
        return row

    def _write_records(self, rows: list[dict[str, Any]], batch_idx: int) -> None:
        import pandas as pd
        import pyarrow as pa
        import pyarrow.parquet as pq

        df = pd.DataFrame(rows)
        for col in self.config.partition_cols:
            if col not in df.columns:
                df[col] = "unknown"
            df[col] = df[col].astype(str)
        table = pa.Table.from_pandas(df, preserve_index=False)
        pq.write_to_dataset(
            table,
            root_path=str(self.records_dir),
            partition_cols=list(self.config.partition_cols),
            basename_template=f"part-{batch_idx:04d}-{{i}}-{uuid.uuid4().hex[:8]}.parquet",
            existing_data_behavior="overwrite_or_ignore",
            compression=self.config.compression,
        )

    def _write_manifest(self, manifest_rows: list[dict[str, Any]]) -> None:
        import pandas as pd

        df = pd.DataFrame(manifest_rows)
        tmp = self.manifest_path.with_suffix(".parquet.tmp")
        df.to_parquet(tmp, compression=self.config.compression, index=False)
        tmp.replace(self.manifest_path)


def read_logs(log_dir: str | Path = "logs/inference") -> "Any":
    """Read the partitioned inference ``records/`` back into a DataFrame.

    Convenience helper for ops / analysis; uses ``pyarrow.dataset`` so the Hive
    partition columns (``date`` / ``model``) are restored as columns.
    """
    import pyarrow.dataset as ds

    records = Path(log_dir) / "records"
    if not records.exists():
        raise FileNotFoundError(f"No inference records found under {records}")
    dataset = ds.dataset(str(records), format="parquet", partitioning="hive")
    return dataset.to_table().to_pandas()
