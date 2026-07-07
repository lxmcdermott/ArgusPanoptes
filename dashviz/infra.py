"""Session state, cached resources, and the unified inference path.

This module is the performance foundation of the dashboard:

* :func:`init_session_state` installs a clean, comprehensive ``st.session_state``
  schema exactly once.
* Cached resources (``st.cache_resource``): the in-process
  :class:`StreamingPerceptor` (per model / chunk size), an optional HTTP client
  for API mode, the physics simulators, and the DSP :class:`SignalProcessor`.
* Cached data (``st.cache_data``): experiment-metric loaders and Parquet
  inference-log reads.
* :func:`run_inference` - one call that works in both **direct** (in-process,
  lowest latency) and **API** (HTTP) modes and always returns a dict matching
  the ``InferenceResponse`` structure.
* :func:`generate_chunk` - drive the simulators to produce a raw vibration chunk
  (+ thermal context) for visualization, and :func:`compute_stft_for_display`.

Heavy third-party imports (perceptor, simulators, pyarrow) are done lazily
inside the cached factories so the app's first paint stays fast.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

# Re-export the shared downsampler so callers can do ``infra.downsample``.
from dashviz.plots import downsample  # noqa: F401

#: Operating-point context keys the perceptor consumes (mirrors the perceptor).
_CONTEXT_KEYS: tuple[str, ...] = (
    "blade_speed_sfpm",
    "num_teeth",
    "feed_per_tooth_mm",
    "depth_mm",
    "kerf_width_mm",
    "tooth_pass_freq_hz",
    "rpm",
    "cutting_velocity_m_s",
    "material_removal_rate_mm3_s",
    "alloy",
    "fs_hz",
)
_THERMAL_KEYS: tuple[str, ...] = (
    "mean_temp_c",
    "max_temp_c",
    "temp_rise_c",
    "therm_std_c",
    "therm_slope_c_per_s",
)

#: Alloy choices (from sensor_specs.yaml machining.alloys).
ALLOY_OPTIONS: tuple[str, ...] = ("6061", "7075")

#: History cap (lightweight summaries only).
HISTORY_CAP: int = 100

DEFAULT_MODEL = "1dcnn_normnone"
DEFAULT_CHUNK_S = 0.5
DEFAULT_API_URL = "http://127.0.0.1:8000"
DEFAULT_LOG_DIR = "logs/inference"


# --------------------------------------------------------------------------- #
# Session state
# --------------------------------------------------------------------------- #
def init_session_state() -> None:
    """Install the default ``st.session_state`` schema on first run (idempotent)."""
    ss = st.session_state
    if ss.get("_argus_initialized"):
        return

    ss.running = False
    ss.step = 0
    ss.max_steps = 24
    ss.delay_s = 0.10
    ss.current_result = None
    ss.history = []                 # list[dict] lightweight summaries (capped)
    ss.last_waveform = None         # (t, accel) tuple for the latest chunk
    ss.last_meta = None
    ss.last_stft = None             # dict(freqs, times, power)
    ss.last_latency_ms = None

    ss.selected_model = DEFAULT_MODEL
    ss.chunk_s = DEFAULT_CHUNK_S
    ss.use_api_mode = False
    ss.api_base_url = DEFAULT_API_URL
    ss.persist_logs = False
    ss.log_dir = DEFAULT_LOG_DIR

    ss.sim_params = {
        "alloy": "6061",
        "blade_speed_sfpm": 800.0,
        "feed_per_tooth_mm": 0.12,
        "depth_mm": 25.0,
        "num_teeth": 80,
    }
    ss.manual_wear = 0.35

    ss.thresholds = {
        "wear_alert": 0.70,
        "anomaly_confidence": 0.50,
        "min_confidence": 0.0,
    }

    ss.demo_scenario = None
    ss.active_scenario = None       # Scenario object currently driving a run
    ss.run_seed = 7
    ss.live_error = ""

    ss._argus_initialized = True


def reset_run_state() -> None:
    """Clear live-run transient state (history, current result, plots)."""
    ss = st.session_state
    ss.running = False
    ss.step = 0
    ss.current_result = None
    ss.history = []
    ss.last_waveform = None
    ss.last_meta = None
    ss.last_stft = None
    ss.last_latency_ms = None
    ss.active_scenario = None


def push_history(summary: dict[str, Any]) -> None:
    """Append a lightweight result summary to history (capped at ``HISTORY_CAP``)."""
    hist = st.session_state.history
    hist.append(summary)
    if len(hist) > HISTORY_CAP:
        del hist[: len(hist) - HISTORY_CAP]


# --------------------------------------------------------------------------- #
# Cached resources
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner=False)
def get_perceptor(model: str, chunk_s: float, persist: bool, log_dir: str):
    """Return a loaded in-process :class:`StreamingPerceptor` (cached per config).

    When ``persist`` is set, a Parquet :class:`InferenceLogger` is attached so
    live predictions land in the same dataset the Historical Explorer reads.
    """
    from models.streaming_perceptor import StreamingPerceptor

    logger_cfg = {"log_dir": log_dir, "flush_every": 8} if persist else None
    perc = StreamingPerceptor(model=model, chunk_s=chunk_s, logger=logger_cfg)
    perc.load()
    return perc


@st.cache_resource(show_spinner=False)
def get_http_client(base_url: str):
    """Return a cached ``httpx.Client`` for API mode (or ``None`` if httpx absent)."""
    try:
        import httpx
    except ImportError:
        return None
    return httpx.Client(base_url=base_url.rstrip("/"), timeout=15.0)


@st.cache_resource(show_spinner=False)
def get_simulators():
    """Return cached ``(SawVibrationSimulator, ThermalSimulator)`` instances."""
    from sensors import SawVibrationSimulator, ThermalSimulator

    return SawVibrationSimulator(), ThermalSimulator()


@st.cache_resource(show_spinner=False)
def get_signal_processor():
    """Return a cached DSP :class:`SignalProcessor` for STFT/feature display."""
    from dsp import SignalProcessor

    return SignalProcessor()


# --------------------------------------------------------------------------- #
# Cached data (metrics + logs)
# --------------------------------------------------------------------------- #
@st.cache_data(ttl=300, show_spinner=False)
def available_model_entries(model_dir: str | None = None) -> list[dict[str, Any]]:
    """Registry entries annotated with on-disk artifact availability (cached)."""
    from models.streaming_perceptor import available_models

    return available_models(model_dir)


@st.cache_data(ttl=300, show_spinner=False)
def known_model_names() -> list[str]:
    """Canonical model registry names (cached)."""
    from models.streaming_perceptor import MODEL_REGISTRY

    return sorted(MODEL_REGISTRY)


@st.cache_data(ttl=60, show_spinner=False)
def load_logs_cached(log_dir: str):
    """Read partitioned inference logs into a DataFrame (cached, ``None`` on miss)."""
    from app.logging import read_logs

    try:
        return read_logs(log_dir)
    except FileNotFoundError:
        return None
    except Exception as exc:  # pragma: no cover - defensive against corrupt logs
        raise RuntimeError(f"Failed to read logs from {log_dir!r}: {exc}") from exc


# --------------------------------------------------------------------------- #
# Simulation + inference
# --------------------------------------------------------------------------- #
def _thermal_stats(temp: np.ndarray, fs_hz: float) -> dict[str, float]:
    """Observable thermal DSP stats (std, slope) used as model context."""
    temp = np.asarray(temp, dtype=np.float64).ravel()
    n = temp.size
    std_c = float(np.std(temp)) if n else 0.0
    if n >= 2 and fs_hz > 0:
        tt = np.arange(n, dtype=np.float64) / fs_hz
        slope = float(np.polyfit(tt, temp, 1)[0])
    else:
        slope = 0.0
    return {"therm_std_c": std_c, "therm_slope_c_per_s": slope}


def generate_chunk(
    sim_params: dict[str, Any],
    wear: float,
    seed: int,
    chunk_s: float,
    *,
    with_thermal: bool = True,
) -> dict[str, Any]:
    """Generate one vibration chunk (+ thermal context) for viz and inference.

    Returns a dict with ``t``, ``accel``, ``meta`` (simulator metadata enriched
    with thermal stats so XGBoost/fusion context is complete), and ``mean_temp_c``.
    """
    saw, therm = get_simulators()
    t, accel, meta = saw.generate(duration_s=chunk_s, params=sim_params, wear=wear, seed=seed)

    mean_temp = float("nan")
    if with_thermal:
        _, temp, tmeta = therm.generate(duration_s=chunk_s, params=sim_params, wear=wear, seed=seed)
        meta["mean_temp_c"] = tmeta.get("mean_temp_c")
        meta["max_temp_c"] = tmeta.get("max_temp_c")
        meta["temp_rise_c"] = tmeta.get("temp_rise_c")
        meta.update(_thermal_stats(np.asarray(temp), float(tmeta.get("fs_hz", 200.0))))
        mean_temp = float(tmeta.get("mean_temp_c", float("nan")))

    return {"t": t, "accel": accel, "meta": meta, "mean_temp_c": mean_temp}


def _context_params(meta: dict[str, Any]) -> dict[str, Any]:
    """Extract the JSON-safe context/thermal keys the perceptor/API consume."""
    out: dict[str, Any] = {}
    for k in _CONTEXT_KEYS + _THERMAL_KEYS:
        v = meta.get(k)
        if v is not None and not (isinstance(v, float) and not np.isfinite(v)):
            out[k] = v
    if "wear" in meta:
        out["wear"] = meta["wear"]
    return out


def run_inference(
    vibration: np.ndarray,
    meta: dict[str, Any],
    *,
    model: str,
    use_api: bool,
    api_base_url: str,
    return_features: bool = False,
    wear_injected: float | None = None,
) -> dict[str, Any]:
    """Run one chunk through the perceptor (direct) or the API (HTTP).

    Always returns a dict matching ``app.main.InferenceResponse`` (predictions,
    recommendations, provenance, latency, optional features). Raises
    ``RuntimeError`` with a friendly message on API/model failures.
    """
    vib = np.asarray(vibration, dtype=np.float64).ravel()
    fs = float(meta.get("fs_hz", 40960.0))

    if use_api:
        return _run_inference_api(
            vib, meta, model=model, api_base_url=api_base_url,
            return_features=return_features, fs=fs,
        )

    ss = st.session_state
    perc = get_perceptor(model, float(ss.chunk_s), bool(ss.persist_logs), ss.log_dir)
    try:
        return perc.infer_chunk(
            vib,
            metadata=meta,
            wear_injected=wear_injected,
            return_features=return_features,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"Model artifact for {model!r} not found. Install the matching extra "
            f"(ml/dl) and ensure experiments/models/ is populated. ({exc})"
        ) from exc
    except ValueError as exc:
        raise RuntimeError(f"Inference failed: {exc}") from exc


def _run_inference_api(
    vibration: np.ndarray,
    meta: dict[str, Any],
    *,
    model: str,
    api_base_url: str,
    return_features: bool,
    fs: float,
) -> dict[str, Any]:
    client = get_http_client(api_base_url)
    if client is None:
        raise RuntimeError("API mode requires httpx (pip install -e \".[dashboard]\").")
    body = {
        "model": model,
        "vibration": vibration.tolist(),
        "fs_hz": fs,
        "params": _context_params(meta),
        "return_features": return_features,
    }
    try:
        resp = client.post("/infer", json=body)
    except Exception as exc:
        raise RuntimeError(
            f"Could not reach the API at {api_base_url}. Is `uvicorn app.main:app` "
            f"running? ({exc})"
        ) from exc
    if resp.status_code != 200:
        detail = _safe_detail(resp)
        raise RuntimeError(f"API returned {resp.status_code}: {detail}")
    return resp.json()


def _safe_detail(resp: Any) -> str:
    try:
        return str(resp.json().get("detail", resp.text))
    except Exception:
        return getattr(resp, "text", "<no detail>")


def api_health(api_base_url: str) -> dict[str, Any]:
    """GET /health from the API (raises ``RuntimeError`` on failure)."""
    client = get_http_client(api_base_url)
    if client is None:
        raise RuntimeError("httpx not installed.")
    try:
        resp = client.get("/health")
    except Exception as exc:
        raise RuntimeError(f"Could not reach API at {api_base_url}: {exc}") from exc
    if resp.status_code != 200:
        raise RuntimeError(f"/health returned {resp.status_code}")
    return resp.json()


def api_models(api_base_url: str) -> dict[str, Any]:
    """GET /models from the API (raises ``RuntimeError`` on failure)."""
    client = get_http_client(api_base_url)
    if client is None:
        raise RuntimeError("httpx not installed.")
    try:
        resp = client.get("/models")
    except Exception as exc:
        raise RuntimeError(f"Could not reach API at {api_base_url}: {exc}") from exc
    if resp.status_code != 200:
        raise RuntimeError(f"/models returned {resp.status_code}")
    return resp.json()


@st.cache_data(ttl=300, show_spinner=False)
def kinematics_preview(
    alloy: str,
    blade_speed_sfpm: float,
    feed_per_tooth_mm: float,
    depth_mm: float,
    num_teeth: int,
) -> dict[str, float]:
    """Derived kinematics (RPM, TPF, MRR) for the current operating point.

    Runs a very short simulator ``generate`` (duration-independent physics) so the
    sidebar can show derived values without a full chunk. Cached per parameter set.
    """
    saw, _ = get_simulators()
    params = {
        "alloy": alloy,
        "blade_speed_sfpm": blade_speed_sfpm,
        "feed_per_tooth_mm": feed_per_tooth_mm,
        "depth_mm": depth_mm,
        "num_teeth": int(num_teeth),
    }
    _, _, meta = saw.generate(duration_s=0.05, params=params, wear=0.0, seed=0)
    return {
        "rpm": float(meta.get("rpm", 0.0)),
        "tpf_hz": float(meta.get("tooth_pass_freq_hz", 0.0)),
        "mrr_mm3_s": float(meta.get("material_removal_rate_mm3_s", 0.0)),
        "cutting_velocity_m_s": float(meta.get("cutting_velocity_m_s", 0.0)),
    }


def compute_stft_for_display(accel: np.ndarray, fs: float) -> dict[str, np.ndarray]:
    """Compute an STFT power spectrogram (dB) of a chunk for the heatmap."""
    proc = get_signal_processor()
    try:
        return proc.compute_stft(accel, fs)
    except Exception:
        return {"freqs": np.array([]), "times": np.array([]), "power": np.zeros((0, 0))}


# --------------------------------------------------------------------------- #
# Result summarization
# --------------------------------------------------------------------------- #
def summarize_result(
    result: dict[str, Any],
    *,
    injected_wear: float | None = None,
    mean_temp_c: float | None = None,
) -> dict[str, Any]:
    """Reduce a full inference result to a lightweight history row."""
    p = result.get("predictions", {})
    rec = result.get("recommendations", {})
    return {
        "chunk_id": result.get("chunk_id"),
        "timestamp": result.get("timestamp"),
        "model": result.get("model"),
        "wear_level": float(p.get("wear_level", 0.0)),
        "cycle_time_factor": float(p.get("cycle_time_factor", 0.0)),
        "quality_score": float(p.get("quality_score", 0.0)),
        "health_state": p.get("health_state", "unknown"),
        "confidence": float(p.get("confidence", 0.0)),
        "anomaly_flag": bool(p.get("anomaly_flag", False)),
        "latency_ms": float(result.get("latency_ms", 0.0)),
        "action": rec.get("action", ""),
        "blade_change_suggested": bool(rec.get("blade_change_suggested", False)),
        "injected_wear": injected_wear,
        "mean_temp_c": mean_temp_c,
    }


def params_from_log_row(row: "Any") -> dict[str, Any]:
    """Rebuild simulator params from a Parquet inference-log row."""
    params: dict[str, Any] = {}
    if pd.notna(row.get("alloy")):
        params["alloy"] = str(row["alloy"])
    for key, col in (
        ("blade_speed_sfpm", "blade_speed_sfpm"),
        ("feed_per_tooth_mm", "feed_per_tooth_mm"),
        ("depth_mm", "depth_mm"),
        ("num_teeth", "num_teeth"),
    ):
        if col in row and pd.notna(row[col]):
            params[key] = float(row[col]) if key != "num_teeth" else int(row[col])
    if not params:
        params = {
            "alloy": "6061",
            "blade_speed_sfpm": 800.0,
            "feed_per_tooth_mm": 0.12,
            "depth_mm": 25.0,
            "num_teeth": 80,
        }
    return params


def alert_level_for(result: dict[str, Any], thresholds: dict[str, float]) -> tuple[str, str, str]:
    """Map a result + thresholds to an ``(level, title, message)`` alert tuple."""
    p = result.get("predictions", {})
    wear = float(p.get("wear_level", 0.0))
    conf = float(p.get("confidence", 0.0))
    state = str(p.get("health_state", "healthy"))
    anomaly = bool(p.get("anomaly_flag", False))

    if state == "critical" or (anomaly and conf >= thresholds.get("anomaly_confidence", 0.5)):
        return ("critical", "CRITICAL", f"Anomaly detected \u2014 wear {wear:.0%}, conf {conf:.0%}. Stop & inspect.")
    if wear >= thresholds.get("wear_alert", 0.7) or state == "warning":
        return ("warning", "WARNING", f"Elevated wear {wear:.0%} ({state}). Plan a blade change.")
    if state == "monitor":
        return ("info", "MONITOR", f"Wear {wear:.0%}; schedule an inspection soon.")
    return ("healthy", "HEALTHY", f"Nominal operation \u2014 wear {wear:.0%}, conf {conf:.0%}.")
