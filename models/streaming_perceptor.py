"""Real-time streaming perception for Argus Panoptes (Day 4 integration layer).

:class:`StreamingPerceptor` is the glue between the hardened perception layers and
the live / online world. It turns a stream of raw accelerometer samples into
structured, downstream-ready prediction payloads by reusing the *exact* same
components the offline pipeline uses:

* :class:`dsp.SignalProcessor` for preprocessing + feature / waveform / spectrogram
  preparation (so streaming inputs are identical to the training inputs), and
* either the XGBoost baseline (``experiments/models/xgb_*.joblib``) or an exported
  ONNX deep-learning variant via :class:`models.onnx_inference.ONNXPerceptor`
  (torch-free CPU inference).

Ring-buffer behaviour
---------------------
Live sensors deliver samples in irregular bursts, not in tidy analysis windows.
:meth:`ingest` appends incoming samples to an internal ``collections.deque``
(bounded to a few analysis chunks so memory is O(chunk)), and
:meth:`process_next_chunk` pops exactly one ``chunk_samples``-long window off the
*front* of the buffer whenever enough samples have accumulated, runs it through
DSP + the selected model, and returns a rich result dict. Partial tails smaller
than a chunk are simply left in the buffer until more data arrives (or dropped on
:meth:`reset`). :meth:`infer_chunk` runs the same pipeline on a caller-supplied
chunk directly (used by the FastAPI ``/infer`` endpoint), bypassing the buffer.

Model selection
---------------
Friendly names map to concrete artifacts / recipes via :data:`MODEL_REGISTRY`
(e.g. ``"xgboost"``, ``"1dcnn_noisy"``, ``"fusion_normnone"``). The DL
``*_normnone`` variants are amplitude-preserving (``normalize_for_dl="none"``)
and the ``*_noisy`` variants are noise-augmented; the perceptor configures a
matching :class:`SignalProcessor` so the streaming front-end reproduces the
training front-end. Model loading is lazy (nothing heavy is imported/loaded until
the first inference) and torch-free.

Every output is finiteness-checked and clamped to physically meaningful ranges.

Example
-------
>>> from sensors import SawVibrationSimulator
>>> from models.streaming_perceptor import StreamingPerceptor
>>> perc = StreamingPerceptor(model="1dcnn_normnone", chunk_s=1.0)
>>> sim = SawVibrationSimulator()
>>> for result in perc.stream_from_simulator(sim, duration_s=2.0, wear=0.6, seed=0):
...     print(result["predictions"]["health_state"])  # doctest: +SKIP
"""

from __future__ import annotations

import datetime as _dt
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:  # pragma: no cover - typing only
    from dsp.signal_processor import SignalProcessor

__version__ = "0.1.0"

__all__ = [
    "StreamingPerceptor",
    "StreamingConfig",
    "ModelSpec",
    "MODEL_REGISTRY",
    "available_models",
    "REGRESSION_TARGETS",
    "HEALTH_CLASS_NAMES",
    "__version__",
]

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_MODEL_DIR = _REPO_ROOT / "experiments" / "models"

#: Regression head output order (matches models.baseline / models.dl_models).
REGRESSION_TARGETS: tuple[str, ...] = ("wear_level", "cycle_time_factor", "quality_score")
#: Health-state class order (alphabetical; matches LabelEncoder & DL export).
HEALTH_CLASS_NAMES: tuple[str, ...] = ("critical", "healthy", "monitor", "warning")

#: Observable thermal statistics used as XGBoost / fusion context (see baseline).
_THERMAL_FEATURE_KEYS: tuple[str, ...] = (
    "mean_temp_c",
    "max_temp_c",
    "temp_rise_c",
    "therm_std_c",
    "therm_slope_c_per_s",
)
#: Operating-point / kinematics context features (no leakage; see baseline).
_CONTEXT_FEATURE_KEYS: tuple[str, ...] = (
    "blade_speed_sfpm",
    "num_teeth",
    "feed_per_tooth_mm",
    "depth_mm",
    "kerf_width_mm",
    "tooth_pass_freq_hz",
    "rpm",
    "cutting_velocity_m_s",
    "material_removal_rate_mm3_s",
)

#: XGBoost regressor artifact filenames keyed by regression target.
_XGB_REG_FILES: dict[str, str] = {
    "wear_level": "xgb_reg_wear_level.joblib",
    "cycle_time_factor": "xgb_reg_cycle_time_factor.joblib",
    "quality_score": "xgb_reg_quality_score.joblib",
}
_XGB_CLF_FILE = "xgb_clf_health_state.joblib"

#: Downstream action recommendation keyed by predicted health state.
_ACTION_BY_HEALTH: dict[str, str] = {
    "healthy": "continue",
    "monitor": "monitor_schedule_inspection",
    "warning": "reduce_feed_plan_blade_change",
    "critical": "stop_replace_blade",
}


# --------------------------------------------------------------------------- #
# Model registry
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ModelSpec:
    """Static description of a selectable model / variant.

    Parameters
    ----------
    name:
        Friendly identifier used in configs / the API (e.g. ``fusion_normnone``).
    kind:
        One of ``"xgboost"``, ``"waveform"``, ``"spectrogram"``, ``"fusion"``.
    onnx:
        ONNX artifact filename (relative to the model dir) for DL variants, or
        ``None`` for the ``xgboost`` ensemble.
    normalize_for_dl:
        Per-chunk DL normalization the artifact was trained with. Streaming must
        reproduce it exactly (``"none"`` for amplitude-preserving ``*_normnone``
        variants, otherwise ``"zscore"``). Ignored for XGBoost.
    description:
        Human-readable summary for ``GET /models`` / docs.
    """

    name: str
    kind: str
    onnx: str | None
    normalize_for_dl: str
    description: str


def _dl_variants() -> dict[str, ModelSpec]:
    specs: dict[str, ModelSpec] = {}
    for family, kind in (("1dcnn", "waveform"), ("fusion", "fusion")):
        for suffix, norm, note in (
            ("", "zscore", "default zscore front-end"),
            ("_normnone", "none", "amplitude-preserving (best clean accuracy)"),
            ("_noisy", "zscore", "noise-augmented sd=0.15 (robust)"),
            ("_noisy01", "zscore", "noise-augmented sd=0.10"),
        ):
            name = f"{family}{suffix}"
            specs[name] = ModelSpec(
                name=name,
                kind=kind,
                onnx=f"dl_{family}{suffix}.onnx",
                normalize_for_dl=norm,
                description=f"{family} ({kind}) - {note}",
            )
    specs["spectrogram"] = ModelSpec(
        name="spectrogram",
        kind="spectrogram",
        onnx="dl_spectrogram.onnx",
        normalize_for_dl="zscore",
        description="2-D CNN on log-power STFT spectrogram",
    )
    return specs


#: All selectable models keyed by friendly name.
MODEL_REGISTRY: dict[str, ModelSpec] = {
    "xgboost": ModelSpec(
        name="xgboost",
        kind="xgboost",
        onnx=None,
        normalize_for_dl="none",
        description="XGBoost baseline ensemble on DSP + thermal + context features",
    ),
    **_dl_variants(),
}

#: Friendly aliases -> canonical registry keys.
_MODEL_ALIASES: dict[str, str] = {
    "xgb": "xgboost",
    "cnn1d": "1dcnn",
    "vibration1dcnn": "1dcnn",
    "cnn2d": "spectrogram",
    "spectrogramcnn": "spectrogram",
    "fusionmodel": "fusion",
}


def resolve_model_name(name: str) -> str:
    """Map a friendly name / alias to a canonical :data:`MODEL_REGISTRY` key."""
    key = name.strip().lower()
    key = _MODEL_ALIASES.get(key, key)
    if key not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model {name!r}. Choose from: {sorted(MODEL_REGISTRY)} "
            f"(aliases: {sorted(_MODEL_ALIASES)})."
        )
    return key


def available_models(model_dir: str | Path | None = None) -> list[dict[str, Any]]:
    """Return registry entries annotated with on-disk artifact availability."""
    mdir = Path(model_dir) if model_dir is not None else _DEFAULT_MODEL_DIR
    out: list[dict[str, Any]] = []
    for spec in MODEL_REGISTRY.values():
        if spec.kind == "xgboost":
            files = [_XGB_CLF_FILE, *_XGB_REG_FILES.values()]
            available = all((mdir / f).is_file() for f in files)
        else:
            available = bool(spec.onnx) and (mdir / spec.onnx).is_file()
        out.append(
            {
                "name": spec.name,
                "kind": spec.kind,
                "normalize_for_dl": spec.normalize_for_dl,
                "description": spec.description,
                "available": available,
            }
        )
    return out


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
class StreamingConfig(BaseModel):
    """Typed configuration for :class:`StreamingPerceptor` (forbids typos)."""

    # ``protected_namespaces=()`` lets us use ``model`` / ``model_dir`` fields.
    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    perceptor_version: str = __version__
    fs_hz: float = Field(default=40960.0, gt=0)
    chunk_s: float = Field(default=1.0, gt=0)
    model: str = "1dcnn_normnone"
    model_dir: str | None = None
    #: Crop/pad each DL chunk to this many samples (``None`` -> use the chunk as-is;
    #: the CNNs global-average-pool so variable length is fine).
    target_len: int | None = None
    #: Ring-buffer capacity as a multiple of ``chunk_samples`` (bounds memory).
    buffer_capacity_chunks: float = Field(default=8.0, gt=0)
    #: Reject / skip chunks shorter than this (guards degenerate DSP).
    min_chunk_samples: int = Field(default=64, ge=8)
    #: ONNX Runtime execution providers (``None`` -> CPU).
    providers: list[str] | None = None


# --------------------------------------------------------------------------- #
# Loaded-model backends
# --------------------------------------------------------------------------- #
@dataclass
class _XGBBackend:
    regressors: dict[str, Any]
    clf: Any
    classes: list[str]
    feature_names: list[str]


@dataclass
class _ONNXBackend:
    perceptor: Any  # models.onnx_inference.ONNXPerceptor
    kind: str
    thermal_dim: int


# --------------------------------------------------------------------------- #
# StreamingPerceptor
# --------------------------------------------------------------------------- #
class StreamingPerceptor:
    """Chunked, real-time perception over a ring buffer of live samples.

    Parameters
    ----------
    config:
        A :class:`StreamingConfig`, a plain ``dict`` of overrides, a ``str`` path
        to a YAML file, or ``None`` for defaults. Explicit keyword arguments
        (``model`` / ``chunk_s`` / ``fs_hz`` / ...) take precedence over ``config``.
    model:
        Friendly model name (see :data:`MODEL_REGISTRY`). Overrides ``config.model``.
    chunk_s, fs_hz, target_len, model_dir:
        Convenience overrides for the matching :class:`StreamingConfig` fields.
    processor:
        An optional pre-built :class:`dsp.SignalProcessor` for the tabular / feature
        path. DL variants get a *derived* processor with the correct
        ``normalize_for_dl`` recipe. ``None`` builds a default processor lazily.
    logger:
        An ``app.logging.InferenceLogger`` (or its config ``dict`` / ``None``) to
        persist every prediction to partitioned Parquet.
    """

    version: str = __version__

    def __init__(
        self,
        config: StreamingConfig | dict[str, Any] | str | None = None,
        *,
        model: str | None = None,
        chunk_s: float | None = None,
        fs_hz: float | None = None,
        target_len: int | None = None,
        model_dir: str | Path | None = None,
        processor: "SignalProcessor | None" = None,
        logger: Any | None = None,
    ) -> None:
        self.config = self._coerce_config(config)
        # Explicit kwargs win over the config object.
        overrides: dict[str, Any] = {}
        if model is not None:
            overrides["model"] = model
        if chunk_s is not None:
            overrides["chunk_s"] = chunk_s
        if fs_hz is not None:
            overrides["fs_hz"] = fs_hz
        if target_len is not None:
            overrides["target_len"] = target_len
        if model_dir is not None:
            overrides["model_dir"] = str(model_dir)
        if overrides:
            self.config = self.config.model_copy(update=overrides)

        self.model_name = resolve_model_name(self.config.model)
        self.spec = MODEL_REGISTRY[self.model_name]
        self.fs_hz = float(self.config.fs_hz)
        self.chunk_samples = max(1, int(round(self.config.chunk_s * self.fs_hz)))
        self.model_dir = Path(self.config.model_dir) if self.config.model_dir else _DEFAULT_MODEL_DIR

        self._processor = processor  # lazy default
        self._dl_processor: "SignalProcessor | None" = None
        self._backend: _XGBBackend | _ONNXBackend | None = None

        buf_max = max(self.chunk_samples, int(self.config.buffer_capacity_chunks * self.chunk_samples))
        self._buffer: deque[float] = deque(maxlen=buf_max)
        self._buffer_metadata: dict[str, Any] = {}
        self._chunk_counter = 0
        self._latest: dict[str, Any] | None = None

        self.logger = self._coerce_logger(logger)

    # ------------------------------------------------------------------ config
    @staticmethod
    def _coerce_config(
        config: StreamingConfig | dict[str, Any] | str | None,
    ) -> StreamingConfig:
        if config is None:
            return StreamingConfig()
        if isinstance(config, StreamingConfig):
            return config
        if isinstance(config, dict):
            return StreamingConfig.model_validate(config)
        if isinstance(config, str):
            import yaml

            with open(config, "r", encoding="utf-8") as fh:
                data = yaml.safe_load(fh) or {}
            return StreamingConfig.model_validate(data)
        raise TypeError(  # pragma: no cover - defensive
            f"config must be StreamingConfig, dict, str path, or None; got {type(config).__name__}"
        )

    @staticmethod
    def _coerce_logger(logger: Any | None) -> Any | None:
        if logger is None or logger is False:
            return None
        # Already an InferenceLogger-like object (duck-typed).
        if hasattr(logger, "log_prediction"):
            return logger
        from app.logging import InferenceLogger  # lazy: keeps models import light

        return InferenceLogger(logger)

    # --------------------------------------------------------------- processors
    @property
    def processor(self) -> "SignalProcessor":
        """The DSP :class:`SignalProcessor` for the tabular / feature path."""
        if self._processor is None:
            from dsp import SignalProcessor

            self._processor = SignalProcessor()
        return self._processor

    @property
    def dl_processor(self) -> "SignalProcessor":
        """A :class:`SignalProcessor` whose ``normalize_for_dl`` matches the model."""
        if self._dl_processor is None:
            from dsp import SignalProcessor
            from dsp.config import ProcessorConfig

            base = self.processor.config.model_dump()
            base["dl"] = {**base.get("dl", {}), "normalize_for_dl": self.spec.normalize_for_dl}
            self._dl_processor = SignalProcessor(ProcessorConfig.model_validate(base))
        return self._dl_processor

    # ------------------------------------------------------------ model loading
    def load(self) -> "StreamingPerceptor":
        """Eagerly load the selected model backend (otherwise lazy on first use)."""
        self._ensure_backend()
        return self

    def _ensure_backend(self) -> _XGBBackend | _ONNXBackend:
        if self._backend is not None:
            return self._backend
        if self.spec.kind == "xgboost":
            self._backend = self._load_xgboost()
        else:
            self._backend = self._load_onnx()
        return self._backend

    def _load_xgboost(self) -> _XGBBackend:
        import joblib

        regressors: dict[str, Any] = {}
        for target, fname in _XGB_REG_FILES.items():
            path = self.model_dir / fname
            if not path.is_file():
                raise FileNotFoundError(f"XGBoost regressor artifact not found: {path}")
            regressors[target] = joblib.load(path)
        clf_path = self.model_dir / _XGB_CLF_FILE
        if not clf_path.is_file():
            raise FileNotFoundError(f"XGBoost classifier artifact not found: {clf_path}")
        bundle = joblib.load(clf_path)
        clf, le = bundle["model"], bundle["label_encoder"]
        feature_names = [str(f) for f in getattr(clf, "feature_names_in_", [])]
        return _XGBBackend(
            regressors=regressors,
            clf=clf,
            classes=[str(c) for c in le.classes_],
            feature_names=feature_names,
        )

    def _load_onnx(self) -> _ONNXBackend:
        from models.onnx_inference import ONNXPerceptor

        assert self.spec.onnx is not None
        path = self.model_dir / self.spec.onnx
        if not path.is_file():
            raise FileNotFoundError(f"ONNX artifact not found: {path}")
        perc = ONNXPerceptor(path, providers=self.config.providers)
        thermal_dim = 5
        if self.spec.kind == "fusion":
            for inp in perc.session.get_inputs():
                if inp.name == "thermal":
                    dim = inp.shape[1] if len(inp.shape) > 1 else None
                    thermal_dim = int(dim) if isinstance(dim, int) else 5
        return _ONNXBackend(perceptor=perc, kind=self.spec.kind, thermal_dim=thermal_dim)

    # ---------------------------------------------------------------- ingestion
    def ingest(
        self, samples: np.ndarray | list[float], metadata: dict[str, Any] | None = None
    ) -> int:
        """Append live vibration samples to the ring buffer.

        Parameters
        ----------
        samples:
            New 1-D vibration samples (g). Non-finite values are rejected.
        metadata:
            Optional operating-point / context metadata (``alloy``, kinematics,
            ``tooth_pass_freq_hz``, thermal stats, ...). Merged into the buffer's
            rolling context and used by the next :meth:`process_next_chunk`.

        Returns
        -------
        int
            The number of samples currently buffered.
        """
        arr = np.asarray(samples, dtype=np.float64).ravel()
        if arr.size and not np.all(np.isfinite(arr)):
            raise ValueError("ingest() received non-finite samples.")
        self._buffer.extend(arr.tolist())
        if metadata:
            self._buffer_metadata.update(metadata)
        return len(self._buffer)

    def buffered_samples(self) -> int:
        """Number of samples currently held in the ring buffer."""
        return len(self._buffer)

    def reset(self) -> None:
        """Clear the ring buffer and rolling metadata (keeps the loaded model)."""
        self._buffer.clear()
        self._buffer_metadata = {}

    def process_next_chunk(self) -> dict[str, Any] | None:
        """Pop and process one full chunk from the buffer, if available.

        Returns a structured result dict (see :meth:`infer_chunk`) when at least
        ``chunk_samples`` samples are buffered, otherwise ``None`` (the partial
        tail is kept until more samples arrive).
        """
        if len(self._buffer) < self.chunk_samples:
            return None
        chunk = np.fromiter(
            (self._buffer.popleft() for _ in range(self.chunk_samples)),
            dtype=np.float64,
            count=self.chunk_samples,
        )
        return self.infer_chunk(chunk, metadata=dict(self._buffer_metadata))

    # ----------------------------------------------------------------- inference
    def infer_chunk(
        self,
        vibration: np.ndarray | list[float],
        *,
        thermal: np.ndarray | list[float] | None = None,
        metadata: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
        chunk_id: int | None = None,
        wear_injected: float | None = None,
        return_features: bool = True,
    ) -> dict[str, Any]:
        """Run DSP + the selected model on one chunk and return a rich result.

        Parameters
        ----------
        vibration:
            1-D raw vibration chunk (g). Must be finite and >= ``min_chunk_samples``.
        thermal:
            Optional thermal feature vector for the ``fusion`` DL model. It must be
            *already prepared* the way training expected (standardized thermal
            scalars); if omitted a neutral zero vector (the standardized dataset
            mean) is used. For XGBoost, thermal scalars are read from ``metadata``
            (``mean_temp_c`` etc.) instead.
        metadata, params:
            Operating-point / context values. ``params`` is merged on top of
            ``metadata`` (convenience for API callers). Used for XGBoost context
            features and echoed into the payload.
        chunk_id:
            Explicit chunk id; defaults to a monotonic counter.
        wear_injected:
            Ground-truth wear (from a simulator) echoed into the payload/logs.
        return_features:
            Include the DSP scalar features in the payload (always logged when a
            logger is attached and configured to include features).

        Returns
        -------
        dict
            Structured payload with ``predictions`` (wear_level, cycle_time_factor,
            quality_score, health_state + probabilities, anomaly_flag, confidence),
            ``recommendations``, echoed ``metadata``, optional ``features``, plus
            ``latency_ms`` / ``timestamp`` / ``model`` provenance.
        """
        t0 = time.perf_counter()
        x = np.asarray(vibration, dtype=np.float64).ravel()
        if x.size < self.config.min_chunk_samples:
            raise ValueError(
                f"chunk has {x.size} samples < min_chunk_samples "
                f"({self.config.min_chunk_samples})."
            )
        if not np.all(np.isfinite(x)):
            raise ValueError("infer_chunk() received non-finite vibration samples.")

        meta: dict[str, Any] = dict(metadata or {})
        if params:
            meta.update(params)
        fs = float(meta.get("fs_hz", self.fs_hz))
        if wear_injected is None and "wear" in meta:
            wear_injected = float(meta["wear"])

        backend = self._ensure_backend()
        features: dict[str, float] = {}
        if isinstance(backend, _XGBBackend):
            reg_vec, probs, features = self._infer_xgboost(backend, x, fs, meta)
        else:
            reg_vec, probs, features = self._infer_onnx(backend, x, fs, meta, thermal, return_features)

        latency_ms = (time.perf_counter() - t0) * 1e3
        cid = self._chunk_counter if chunk_id is None else int(chunk_id)
        self._chunk_counter = cid + 1
        result = self._build_result(
            reg_vec=reg_vec,
            probs=probs,
            features=features if return_features else {},
            meta=meta,
            fs=fs,
            n_samples=int(x.size),
            latency_ms=latency_ms,
            chunk_id=cid,
            wear_injected=wear_injected,
        )
        self._latest = result
        if self.logger is not None:
            self.logger.log_prediction(result)
        return result

    def _infer_xgboost(
        self, backend: _XGBBackend, x: np.ndarray, fs: float, meta: dict[str, Any]
    ) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
        import pandas as pd

        proc = self.processor.process(x, fs=fs, metadata=meta)
        features = dict(proc["features"])
        available: dict[str, Any] = {f"vib_{k}": v for k, v in features.items()}
        for key in _THERMAL_FEATURE_KEYS + _CONTEXT_FEATURE_KEYS:
            available[key] = meta.get(key, np.nan)
        names = backend.feature_names or (
            [f"vib_{k}" for k in features] + list(_THERMAL_FEATURE_KEYS) + list(_CONTEXT_FEATURE_KEYS)
        )
        row = pd.DataFrame([{n: available.get(n, np.nan) for n in names}], columns=names)
        reg_vec = np.array(
            [float(backend.regressors[t].predict(row)[0]) for t in REGRESSION_TARGETS],
            dtype=np.float64,
        )
        raw_probs = np.asarray(backend.clf.predict_proba(row)[0], dtype=np.float64)
        probs = self._align_probs(raw_probs, backend.classes)
        return reg_vec, probs, features

    def _infer_onnx(
        self,
        backend: _ONNXBackend,
        x: np.ndarray,
        fs: float,
        meta: dict[str, Any],
        thermal: np.ndarray | list[float] | None,
        return_features: bool,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
        features: dict[str, float] = {}
        if return_features or self._logger_wants_features():
            features = dict(self.processor.process(x, fs=fs, metadata=meta)["features"])

        if backend.kind == "waveform":
            w = self._prep_waveform(x, fs)
            out = backend.perceptor.predict(w[None, None, :])
        elif backend.kind == "spectrogram":
            spec = self.dl_processor.compute_spectrogram(x, fs).astype(np.float32)
            out = backend.perceptor.predict(spec[None, None, :, :])
        elif backend.kind == "fusion":
            w = self._prep_waveform(x, fs)
            th = self._prep_thermal(thermal, backend.thermal_dim)
            out = backend.perceptor.predict(waveform=w[None, None, :], thermal=th[None, :])
        else:  # pragma: no cover - defensive
            raise ValueError(f"Unknown backend kind: {backend.kind!r}")

        reg_vec = np.asarray(out["regression"], dtype=np.float64).ravel()[:3]
        probs = np.asarray(out["health_probs"], dtype=np.float64).ravel()
        # ONNX health-logit order is DEFAULT_CLASS_NAMES (== HEALTH_CLASS_NAMES).
        probs = self._align_probs(probs, list(HEALTH_CLASS_NAMES))
        return reg_vec, probs, features

    def _prep_waveform(self, x: np.ndarray, fs: float) -> np.ndarray:
        w = self.dl_processor.get_normalized_waveform(x, fs)
        if self.config.target_len:
            w = _crop_or_pad(w, self.config.target_len)
        return np.ascontiguousarray(w, dtype=np.float32)

    @staticmethod
    def _prep_thermal(thermal: np.ndarray | list[float] | None, dim: int) -> np.ndarray:
        if thermal is None:
            return np.zeros(dim, dtype=np.float32)
        th = np.asarray(thermal, dtype=np.float32).ravel()
        if not np.all(np.isfinite(th)):
            raise ValueError("infer_chunk() received non-finite thermal features.")
        if th.size < dim:
            th = np.pad(th, (0, dim - th.size))
        elif th.size > dim:
            th = th[:dim]
        return th

    def _logger_wants_features(self) -> bool:
        return bool(
            self.logger is not None
            and getattr(getattr(self.logger, "config", None), "include_features", False)
        )

    @staticmethod
    def _align_probs(probs: np.ndarray, classes: list[str]) -> np.ndarray:
        """Reorder ``probs`` (in ``classes`` order) into HEALTH_CLASS_NAMES order."""
        probs = np.asarray(probs, dtype=np.float64).ravel()
        if list(classes) == list(HEALTH_CLASS_NAMES) and probs.size == len(HEALTH_CLASS_NAMES):
            return probs
        aligned = np.zeros(len(HEALTH_CLASS_NAMES), dtype=np.float64)
        idx = {c: i for i, c in enumerate(classes)}
        for j, name in enumerate(HEALTH_CLASS_NAMES):
            if name in idx and idx[name] < probs.size:
                aligned[j] = probs[idx[name]]
        total = aligned.sum()
        return aligned / total if total > 0 else aligned

    # ------------------------------------------------------------------ payload
    def _build_result(
        self,
        *,
        reg_vec: np.ndarray,
        probs: np.ndarray,
        features: dict[str, float],
        meta: dict[str, Any],
        fs: float,
        n_samples: int,
        latency_ms: float,
        chunk_id: int,
        wear_injected: float | None,
    ) -> dict[str, Any]:
        reg = np.nan_to_num(np.asarray(reg_vec, dtype=np.float64).ravel(), nan=0.0)
        wear_level = float(np.clip(reg[0] if reg.size > 0 else 0.0, 0.0, 1.0))
        cycle_time_factor = float(max(reg[1] if reg.size > 1 else 0.0, 0.0))
        quality_score = float(np.clip(reg[2] if reg.size > 2 else 0.0, 0.0, 1.0))

        probs = np.nan_to_num(np.asarray(probs, dtype=np.float64).ravel(), nan=0.0)
        if probs.size != len(HEALTH_CLASS_NAMES):
            probs = np.resize(probs, len(HEALTH_CLASS_NAMES))
        cls_idx = int(np.argmax(probs)) if probs.size else 0
        health_state = HEALTH_CLASS_NAMES[cls_idx]
        confidence = float(probs[cls_idx]) if probs.size else 0.0
        health_probs = {c: float(p) for c, p in zip(HEALTH_CLASS_NAMES, probs)}

        anomaly_flag = bool(health_state == "critical" or wear_level >= 0.85)
        blade_change = bool(health_state in ("warning", "critical") or wear_level >= 0.7)

        echoed = {
            "alloy": meta.get("alloy"),
            "blade_speed_sfpm": _opt_float(meta.get("blade_speed_sfpm")),
            "num_teeth": _opt_int(meta.get("num_teeth")),
            "feed_per_tooth_mm": _opt_float(meta.get("feed_per_tooth_mm")),
            "depth_mm": _opt_float(meta.get("depth_mm")),
            "kerf_width_mm": _opt_float(meta.get("kerf_width_mm")),
            "tpf_hz": _opt_float(meta.get("tooth_pass_freq_hz")),
            "wear_injected": _opt_float(wear_injected),
        }

        return {
            "chunk_id": chunk_id,
            "timestamp": _dt.datetime.now(_dt.timezone.utc).isoformat(),
            "model": self.model_name,
            "model_kind": self.spec.kind,
            "model_variant": self.spec.onnx or "xgboost_ensemble",
            "model_version": self.version,
            "fs_hz": float(fs),
            "n_samples": int(n_samples),
            "latency_ms": float(latency_ms),
            "predictions": {
                "wear_level": wear_level,
                "cycle_time_factor": cycle_time_factor,
                "quality_score": quality_score,
                "health_state": health_state,
                "health_probs": health_probs,
                "anomaly_flag": anomaly_flag,
                "confidence": confidence,
            },
            "recommendations": {
                "action": _ACTION_BY_HEALTH.get(health_state, "monitor_schedule_inspection"),
                "blade_change_suggested": blade_change,
                "cycle_time_factor": cycle_time_factor,
                "quality_score": quality_score,
                "note": f"health={health_state} (conf {confidence:.2f}); "
                f"est. wear {wear_level:.2f}",
            },
            "metadata": echoed,
            "features": {k: float(v) for k, v in features.items()},
        }

    # ----------------------------------------------------------------- streaming
    def get_latest_prediction(self) -> dict[str, Any] | None:
        """Return the most recent structured prediction (``None`` if none yet)."""
        return self._latest

    def stream_from_simulator(
        self,
        simulator: Any,
        *,
        duration_s: float = 5.0,
        chunk_s: float | None = None,
        params: dict[str, Any] | None = None,
        wear: float = 0.0,
        seed: int | None = None,
        thermal_simulator: Any | None = None,
        return_features: bool = False,
    ) -> Iterator[dict[str, Any]]:
        """Pull chunks from a simulator's ``stream()`` and yield predictions.

        Convenience generator wiring ``simulator.stream()`` -> DSP -> model ->
        (optional) Parquet log -> structured payload. Operating-point context
        (TPF / kinematics) is duration-independent, so it is fetched once via a
        short :meth:`generate` call and attached to every chunk. When a
        ``thermal_simulator`` is supplied, observable thermal statistics are added
        so the XGBoost / fusion context is complete.
        """
        chunk_s = float(chunk_s if chunk_s is not None else self.config.chunk_s)
        meta = self._context_metadata(
            simulator, params=params, wear=wear, seed=seed, chunk_s=chunk_s,
            thermal_simulator=thermal_simulator, duration_s=duration_s,
        )
        for i, (_t_chunk, accel_chunk) in enumerate(
            simulator.stream(duration_s=duration_s, chunk_s=chunk_s, params=params, wear=wear, seed=seed)
        ):
            if np.asarray(accel_chunk).size < self.config.min_chunk_samples:
                continue
            yield self.infer_chunk(
                accel_chunk,
                metadata=meta,
                chunk_id=i,
                wear_injected=wear,
                return_features=return_features,
            )

    def _context_metadata(
        self,
        simulator: Any,
        *,
        params: dict[str, Any] | None,
        wear: float,
        seed: int | None,
        chunk_s: float,
        thermal_simulator: Any | None,
        duration_s: float,
    ) -> dict[str, Any]:
        _, _, meta = simulator.generate(duration_s=chunk_s, params=params, wear=wear, seed=seed)
        context = {k: meta[k] for k in _CONTEXT_FEATURE_KEYS if k in meta}
        context["alloy"] = meta.get("alloy")
        context["fs_hz"] = meta.get("fs_hz", self.fs_hz)
        context["tooth_pass_freq_hz"] = meta.get("tooth_pass_freq_hz")
        context["wear"] = wear
        if thermal_simulator is not None:
            _, temp, tmeta = thermal_simulator.generate(
                duration_s=duration_s, params=params, wear=wear, seed=seed
            )
            context["mean_temp_c"] = tmeta.get("mean_temp_c")
            context["max_temp_c"] = tmeta.get("max_temp_c")
            context["temp_rise_c"] = tmeta.get("temp_rise_c")
            context.update(_thermal_stats(np.asarray(temp), float(tmeta.get("fs_hz", 200.0))))
        return context

    # -------------------------------------------------------------------- close
    def flush(self) -> None:
        """Flush the attached logger's buffered predictions to Parquet (if any)."""
        if self.logger is not None:
            self.logger.flush()

    def close(self) -> None:
        """Flush and release the logger (safe to call multiple times)."""
        if self.logger is not None:
            self.logger.close()

    def __enter__(self) -> "StreamingPerceptor":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _crop_or_pad(x: np.ndarray, target_len: int) -> np.ndarray:
    """Center-crop or zero-pad a 1-D signal to ``target_len``."""
    if x.size == target_len:
        return x
    if x.size > target_len:
        start = (x.size - target_len) // 2
        return x[start : start + target_len]
    out = np.zeros(target_len, dtype=x.dtype)
    out[: x.size] = x
    return out


def _thermal_stats(temp: np.ndarray, fs_hz: float) -> dict[str, float]:
    """Compute the observable thermal DSP stats used as context features."""
    temp = np.asarray(temp, dtype=np.float64).ravel()
    n = temp.size
    std_c = float(np.std(temp)) if n else 0.0
    if n >= 2 and fs_hz > 0:
        tt = np.arange(n, dtype=np.float64) / fs_hz
        slope = float(np.polyfit(tt, temp, 1)[0])
    else:
        slope = 0.0
    return {"therm_std_c": std_c, "therm_slope_c_per_s": slope}


def _opt_float(v: Any) -> float | None:
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if np.isfinite(f) else None


def _opt_int(v: Any) -> int | None:
    if v is None:
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None
