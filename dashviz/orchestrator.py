"""Framework-agnostic simulation + inference core for the Argus dashboards.

This module is the performance foundation of the **NiceGUI** operator dashboard
(``app/nicegui_dashboard.py``). It contains everything the UI needs but nothing
UI-specific, so it can run headless and be unit-tested without any web
framework:

* a small process-wide cache of loaded :class:`StreamingPerceptor` instances,
  simulators, DSP processor, and HTTP clients (thread-safe, keyed by config);
* pure functions to drive the simulators (:func:`generate_chunk`), run one chunk
  through the perceptor either in-process or over HTTP (:func:`run_inference`),
  and prepare an STFT for display (:func:`compute_stft_for_display`);
* :class:`SimulationOrchestrator` - a background-thread producer that owns the
  operating point and continuously turns simulated vibration + thermal into
  structured predictions, publishing an immutable :class:`Snapshot` the UI can
  read at its own refresh rate.

Contracts consumed (unchanged): ``StreamingPerceptor.infer_chunk`` /
``available_models`` / ``resolve_model_name`` / ``MODEL_REGISTRY``, the
simulators' ``generate`` methods, ``dsp.SignalProcessor.compute_stft``, the
FastAPI ``/infer`` endpoint, and ``app.logging.read_logs``. Nothing in
``sensors/``, ``dsp/``, ``models/``, or ``app/`` is modified.
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from dashviz.plots import downsample  # noqa: F401  (re-exported for callers)
from dashviz.scenarios import SCENARIOS, Scenario

# --------------------------------------------------------------------------- #
# Constants / defaults (mirrors the old dashviz.infra contract)
# --------------------------------------------------------------------------- #
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

ALLOY_OPTIONS: tuple[str, ...] = ("6061", "7075")
HISTORY_CAP: int = 240

DEFAULT_MODEL = "1dcnn_normnone"
DEFAULT_CHUNK_S = 0.5
DEFAULT_API_URL = "http://127.0.0.1:8000"
DEFAULT_LOG_DIR = "logs/inference"

#: Aggressive display caps keep multi-stream live updates smooth (see module doc).
LIVE_WAVE_POINTS = 1200
LIVE_FFT_POINTS = 1200


# --------------------------------------------------------------------------- #
# Process-wide cached resources (thread-safe)
# --------------------------------------------------------------------------- #
_RES_LOCK = threading.RLock()
_PERCEPTORS: dict[tuple[str, float, bool, str], Any] = {}
_HTTP_CLIENTS: dict[str, Any] = {}
_SIMULATORS: tuple[Any, Any] | None = None
_PROCESSOR: Any = None


def get_perceptor(model: str, chunk_s: float, persist: bool, log_dir: str):
    """Return a loaded in-process ``StreamingPerceptor`` (cached per config).

    When ``persist`` is set, a Parquet ``InferenceLogger`` is attached so live
    predictions land in the same dataset the Historical Explorer reads.
    """
    key = (model, float(chunk_s), bool(persist), str(log_dir))
    with _RES_LOCK:
        perc = _PERCEPTORS.get(key)
        if perc is None:
            from models.streaming_perceptor import StreamingPerceptor

            logger_cfg = {"log_dir": log_dir, "flush_every": 8} if persist else None
            perc = StreamingPerceptor(model=model, chunk_s=chunk_s, logger=logger_cfg)
            perc.load()
            _PERCEPTORS[key] = perc
        return perc


def get_http_client(base_url: str):
    """Return a cached ``httpx.Client`` for API mode (or ``None`` if httpx absent)."""
    key = base_url.rstrip("/")
    with _RES_LOCK:
        client = _HTTP_CLIENTS.get(key)
        if client is None:
            try:
                import httpx
            except ImportError:
                return None
            client = httpx.Client(base_url=key, timeout=15.0)
            _HTTP_CLIENTS[key] = client
        return client


def get_simulators():
    """Return cached ``(SawVibrationSimulator, ThermalSimulator)`` instances."""
    global _SIMULATORS
    with _RES_LOCK:
        if _SIMULATORS is None:
            from sensors import SawVibrationSimulator, ThermalSimulator

            _SIMULATORS = (SawVibrationSimulator(), ThermalSimulator())
        return _SIMULATORS


def get_signal_processor():
    """Return a cached DSP ``SignalProcessor`` for STFT / feature display."""
    global _PROCESSOR
    with _RES_LOCK:
        if _PROCESSOR is None:
            from dsp import SignalProcessor

            _PROCESSOR = SignalProcessor()
        return _PROCESSOR


def clear_caches() -> None:
    """Drop cached perceptors / clients (used by the UI 'reload models' button)."""
    global _SIMULATORS, _PROCESSOR
    with _RES_LOCK:
        for c in _HTTP_CLIENTS.values():
            try:
                c.close()
            except Exception:  # pragma: no cover - best effort
                pass
        _PERCEPTORS.clear()
        _HTTP_CLIENTS.clear()
        _SIMULATORS = None
        _PROCESSOR = None


# --------------------------------------------------------------------------- #
# Model registry helpers (thin, cache-free wrappers)
# --------------------------------------------------------------------------- #
def available_model_entries(model_dir: str | None = None) -> list[dict[str, Any]]:
    """Registry entries annotated with on-disk artifact availability."""
    from models.streaming_perceptor import available_models

    return available_models(model_dir)


def known_model_names() -> list[str]:
    """Canonical model registry names (sorted)."""
    from models.streaming_perceptor import MODEL_REGISTRY

    return sorted(MODEL_REGISTRY)


# --------------------------------------------------------------------------- #
# Simulation + inference (pure; no UI dependency)
# --------------------------------------------------------------------------- #
def _thermal_stats(temp: np.ndarray, fs_hz: float) -> dict[str, float]:
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
    """Generate one vibration chunk (+ thermal context) for viz and inference."""
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
    chunk_s: float,
    persist: bool,
    log_dir: str,
    return_features: bool = False,
    wear_injected: float | None = None,
) -> dict[str, Any]:
    """Run one chunk through the perceptor (direct) or the API (HTTP).

    Always returns a dict matching ``app.main.InferenceResponse``. Raises
    ``RuntimeError`` with a friendly message on API / model failures.
    """
    vib = np.asarray(vibration, dtype=np.float64).ravel()
    fs = float(meta.get("fs_hz", 40960.0))

    if use_api:
        return _run_inference_api(
            vib, meta, model=model, api_base_url=api_base_url,
            return_features=return_features, fs=fs,
        )

    perc = get_perceptor(model, float(chunk_s), bool(persist), log_dir)
    try:
        return perc.infer_chunk(
            vib, metadata=meta, wear_injected=wear_injected, return_features=return_features,
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
        raise RuntimeError("API mode requires httpx (pip install -e \".[dashboard-nicegui]\").")
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
        raise RuntimeError(f"API returned {resp.status_code}: {_safe_detail(resp)}")
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


def kinematics_preview(
    alloy: str,
    blade_speed_sfpm: float,
    feed_per_tooth_mm: float,
    depth_mm: float,
    num_teeth: int,
) -> dict[str, float]:
    """Derived kinematics (RPM, TPF, MRR) for the current operating point."""
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


def load_logs(log_dir: str):
    """Read partitioned inference logs into a DataFrame (``None`` on miss)."""
    from app.logging import read_logs

    try:
        return read_logs(log_dir)
    except FileNotFoundError:
        return None
    except Exception as exc:  # pragma: no cover - defensive against corrupt logs
        raise RuntimeError(f"Failed to read logs from {log_dir!r}: {exc}") from exc


# --------------------------------------------------------------------------- #
# Result summarization / alerts (ported from the old infra, UI-free)
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


def params_from_log_row(row: Any) -> dict[str, Any]:
    """Rebuild simulator params from a Parquet inference-log row."""
    import pandas as pd

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
            "alloy": "6061", "blade_speed_sfpm": 800.0, "feed_per_tooth_mm": 0.12,
            "depth_mm": 25.0, "num_teeth": 80,
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
        return ("critical", "CRITICAL",
                f"Anomaly detected \u2014 wear {wear:.0%}, conf {conf:.0%}. Stop & inspect.")
    if wear >= thresholds.get("wear_alert", 0.7) or state == "warning":
        return ("warning", "WARNING", f"Elevated wear {wear:.0%} ({state}). Plan a blade change.")
    if state == "monitor":
        return ("info", "MONITOR", f"Wear {wear:.0%}; schedule an inspection soon.")
    return ("healthy", "HEALTHY", f"Nominal operation \u2014 wear {wear:.0%}, conf {conf:.0%}.")


# --------------------------------------------------------------------------- #
# Session config + immutable snapshot
# --------------------------------------------------------------------------- #
@dataclass
class SessionConfig:
    """Mutable operating configuration shared with :class:`SimulationOrchestrator`."""

    model: str = DEFAULT_MODEL
    chunk_s: float = DEFAULT_CHUNK_S
    use_api: bool = False
    api_base_url: str = DEFAULT_API_URL
    persist_logs: bool = False
    log_dir: str = DEFAULT_LOG_DIR
    delay_s: float = 0.12
    sim_params: dict[str, Any] = field(default_factory=lambda: {
        "alloy": "6061", "blade_speed_sfpm": 800.0, "feed_per_tooth_mm": 0.12,
        "depth_mm": 25.0, "num_teeth": 80,
    })
    thresholds: dict[str, float] = field(default_factory=lambda: {
        "wear_alert": 0.70, "anomaly_confidence": 0.50, "min_confidence": 0.0,
    })


@dataclass(frozen=True)
class Snapshot:
    """An immutable, self-consistent view of the latest orchestrator state.

    The background thread swaps a *new* ``Snapshot`` in atomically, so a UI
    reader always sees a coherent frame (never a half-updated one) without
    holding the lock while it renders.
    """

    running: bool = False
    paused: bool = False
    step: int = 0
    max_steps: int = 24
    scenario_name: str = "Manual"
    model: str = DEFAULT_MODEL
    mode: str = "Standalone"
    latency_ms: float | None = None
    chunk_s: float = DEFAULT_CHUNK_S
    error: str = ""
    current_result: dict[str, Any] | None = None
    wave_t: np.ndarray | None = None
    wave_accel: np.ndarray | None = None
    meta: dict[str, Any] | None = None
    stft: dict[str, np.ndarray] | None = None
    mean_temp_c: float = float("nan")
    history: list[dict[str, Any]] = field(default_factory=list)
    generation: int = 0  # increments every produced chunk (change detection)


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #
class SimulationOrchestrator:
    """Background producer: simulate -> DSP -> infer, publishing a ``Snapshot``.

    The orchestrator runs a single daemon thread for its entire lifetime. While
    *running* it produces one chunk per ``config.delay_s`` tick; while idle it
    parks cheaply on an event. All state mutations are guarded by a lock and the
    heavy DSP / inference work is done *outside* the lock so param updates and
    ``snapshot()`` reads never block on inference.
    """

    def __init__(self, config: SessionConfig | None = None) -> None:
        self.config = config or SessionConfig()
        self._lock = threading.RLock()
        self._wake = threading.Event()
        self._stop = threading.Event()

        self._running = False
        self._paused = False
        self._step = 0
        self._max_steps = 24
        self._run_seed = 7
        self._active_scenario: Scenario | None = None
        self._manual_wear = 0.35
        self._manual_noise = 0.0
        self._generation = 0
        self._history: deque[dict[str, Any]] = deque(maxlen=HISTORY_CAP)
        self._snapshot = Snapshot(model=self.config.model, chunk_s=self.config.chunk_s)

        self._thread = threading.Thread(target=self._loop, name="argus-sim", daemon=True)
        self._thread.start()

    # ------------------------------------------------------------------ control
    def start(self, scenario_key: str | None = None) -> None:
        """Reset run state and begin producing chunks (optionally a scenario)."""
        with self._lock:
            self._reset_locked()
            if scenario_key and scenario_key in SCENARIOS:
                sc = SCENARIOS[scenario_key]
                self._active_scenario = sc
                self.config.sim_params = dict(sc.params)
                self.config.delay_s = float(sc.delay_s)
                self._max_steps = int(sc.n_chunks)
                self._run_seed = int(sc.seed)
            else:
                self._active_scenario = None
                self._max_steps = 24
            self._running = True
            self._paused = False
        self._wake.set()

    def start_manual(self) -> None:
        """Start a free-running manual session (no scenario)."""
        self.start(scenario_key=None)

    def stop(self) -> None:
        with self._lock:
            self._running = False
            self._paused = False
        self._publish()

    def toggle_pause(self) -> bool:
        with self._lock:
            if self._running:
                self._paused = not self._paused
            paused = self._paused
        if not paused:
            self._wake.set()
        self._publish()
        return paused

    def step_once(self) -> None:
        """Advance exactly one chunk synchronously (used by the Step button)."""
        with self._lock:
            self._running = False
            self._paused = False
        self._advance(single=True)

    def reset(self) -> None:
        with self._lock:
            self._reset_locked()
        self._publish()

    def _reset_locked(self) -> None:
        self._running = False
        self._paused = False
        self._step = 0
        self._active_scenario = None
        self._history.clear()
        self._generation += 1
        self._snapshot = Snapshot(model=self.config.model, chunk_s=self.config.chunk_s,
                                  generation=self._generation)

    def shutdown(self) -> None:
        """Stop the thread and release resources (idempotent)."""
        self._stop.set()
        self._wake.set()
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)

    # --------------------------------------------------------------- live params
    def update_params(self, **params: Any) -> None:
        """Merge operating-point params; take effect on the next manual chunk."""
        with self._lock:
            self.config.sim_params.update(params)

    def set_manual_wear(self, wear: float) -> None:
        with self._lock:
            self._manual_wear = float(np.clip(wear, 0.0, 1.0))

    def set_manual_noise(self, noise_sd: float) -> None:
        with self._lock:
            self._manual_noise = float(max(0.0, noise_sd))

    def bump_wear(self, delta: float = 0.1) -> float:
        with self._lock:
            self._manual_wear = float(np.clip(self._manual_wear + delta, 0.0, 1.0))
            return self._manual_wear

    def set_model(self, model: str) -> None:
        with self._lock:
            self.config.model = model

    def set_delay(self, delay_s: float) -> None:
        with self._lock:
            self.config.delay_s = float(max(0.02, delay_s))

    def set_mode(self, use_api: bool, api_base_url: str | None = None) -> None:
        with self._lock:
            self.config.use_api = bool(use_api)
            if api_base_url is not None:
                self.config.api_base_url = api_base_url

    def set_persist(self, persist: bool, log_dir: str | None = None) -> None:
        with self._lock:
            self.config.persist_logs = bool(persist)
            if log_dir is not None:
                self.config.log_dir = log_dir

    def set_chunk_s(self, chunk_s: float) -> None:
        with self._lock:
            self.config.chunk_s = float(chunk_s)

    def set_thresholds(self, **th: float) -> None:
        with self._lock:
            self.config.thresholds.update({k: float(v) for k, v in th.items()})

    # ------------------------------------------------------------------ snapshot
    def snapshot(self) -> Snapshot:
        """Return the latest immutable snapshot (safe to read from any thread)."""
        with self._lock:
            return self._snapshot

    @property
    def manual_wear(self) -> float:
        with self._lock:
            return self._manual_wear

    @property
    def manual_noise(self) -> float:
        with self._lock:
            return self._manual_noise

    # --------------------------------------------------------------------- loop
    def _loop(self) -> None:
        while not self._stop.is_set():
            with self._lock:
                active = self._running and not self._paused
                delay = float(self.config.delay_s)
            if active:
                try:
                    self._advance()
                except Exception as exc:  # keep the thread alive; surface via snapshot
                    with self._lock:
                        self._running = False
                    self._publish(error=str(exc))
                # Pace the loop; wake early on stop/pause changes.
                self._wake.wait(timeout=delay)
                self._wake.clear()
            else:
                # Idle: park until woken (zero busy-work when not running).
                self._wake.wait(timeout=0.25)
                self._wake.clear()

    def _advance(self, *, single: bool = False) -> None:
        """Produce and publish exactly one chunk."""
        with self._lock:
            sc = self._active_scenario
            step = self._step
            if sc is not None:
                if step >= sc.n_chunks:
                    self._running = False
                    self._publish_locked()
                    return
                wear = sc.wear_for_step(step)
                params = dict(sc.params)
                seed = sc.seed + step
                noise_sd = float(getattr(sc, "noise_sd", 0.0))
            else:
                if not single and step >= self._max_steps:
                    self._running = False
                    self._publish_locked()
                    return
                wear = float(self._manual_wear)
                params = dict(self.config.sim_params)
                seed = int(self._run_seed) + step
                noise_sd = float(self._manual_noise)
            cfg = replace(self.config)  # shallow copy of the current config

        # --- Heavy work OUTSIDE the lock ---
        chunk = generate_chunk(params, wear, seed, float(cfg.chunk_s), with_thermal=True)
        accel = chunk["accel"]
        if noise_sd > 0:
            rng = np.random.default_rng(seed + 777)
            rms = float(np.sqrt(np.mean(accel**2))) or 1.0
            accel = accel + rng.normal(0.0, noise_sd * rms, size=accel.shape)
        result = run_inference(
            accel, chunk["meta"], model=cfg.model, use_api=cfg.use_api,
            api_base_url=cfg.api_base_url, chunk_s=cfg.chunk_s, persist=cfg.persist_logs,
            log_dir=cfg.log_dir, return_features=False, wear_injected=wear,
        )
        stft = compute_stft_for_display(accel, float(chunk["meta"].get("fs_hz", 40960.0)))
        summary = summarize_result(result, injected_wear=wear, mean_temp_c=chunk["mean_temp_c"])

        # Downsample the waveform once, here, so every consumer stays cheap.
        t_ds = downsample(chunk["t"], LIVE_WAVE_POINTS)
        a_ds = downsample(accel, LIVE_WAVE_POINTS)

        with self._lock:
            self._history.append(summary)
            self._step += 1
            self._generation += 1
            self._snapshot = Snapshot(
                running=self._running,
                paused=self._paused,
                step=self._step,
                max_steps=(sc.n_chunks if sc is not None else self._max_steps),
                scenario_name=(sc.name if sc is not None else "Manual"),
                model=result.get("model", cfg.model),
                mode=("API" if cfg.use_api else "Standalone"),
                latency_ms=float(result.get("latency_ms", 0.0)),
                chunk_s=float(cfg.chunk_s),
                error="",
                current_result=result,
                wave_t=t_ds,
                wave_accel=a_ds,
                meta=chunk["meta"],
                stft=stft,
                mean_temp_c=chunk["mean_temp_c"],
                history=list(self._history),
                generation=self._generation,
            )

    # ---------------------------------------------------------------- publishing
    def _publish(self, *, error: str | None = None) -> None:
        with self._lock:
            self._publish_locked(error=error)

    def _publish_locked(self, *, error: str | None = None) -> None:
        prev = self._snapshot
        sc = self._active_scenario
        self._generation += 1
        self._snapshot = replace(
            prev,
            running=self._running,
            paused=self._paused,
            step=self._step,
            max_steps=(sc.n_chunks if sc is not None else self._max_steps),
            scenario_name=(sc.name if sc is not None else "Manual"),
            model=self.config.model,
            mode=("API" if self.config.use_api else "Standalone"),
            chunk_s=float(self.config.chunk_s),
            history=list(self._history),
            error=error if error is not None else prev.error,
            generation=self._generation,
        )
