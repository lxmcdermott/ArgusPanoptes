"""Loaders for the experiment metric JSONs surfaced in the dashboard.

Pure helpers (no Streamlit) that read the committed ``experiments/*.json``
artifacts (ONNX latency benchmark, XGBoost baseline metrics, per-model DL
metrics, robustness ablation) and reshape them for display. Caching is applied
by the callers in :mod:`dashviz.infra` via ``st.cache_data``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
_EXPERIMENTS = _REPO_ROOT / "experiments"


def repo_root() -> Path:
    """Absolute path to the repository root."""
    return _REPO_ROOT


def _read_json(path: Path) -> dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return {}


def load_benchmark() -> dict[str, Any]:
    """Return the ONNX/XGBoost latency benchmark (``experiments/onnx_benchmark.json``)."""
    return _read_json(_EXPERIMENTS / "onnx_benchmark.json")


def load_baseline_metrics() -> dict[str, Any]:
    """Return the XGBoost baseline metrics (``experiments/baseline_metrics.json``)."""
    return _read_json(_EXPERIMENTS / "baseline_metrics.json")


def load_dl_metrics(artifact_name: str) -> dict[str, Any]:
    """Return per-model DL metrics by artifact name (e.g. ``dl_1dcnn_normnone``)."""
    return _read_json(_EXPERIMENTS / f"{artifact_name}_metrics.json")


def load_robustness() -> dict[str, Any]:
    """Return the noise-robustness ablation (``experiments/robustness_results.json``)."""
    return _read_json(_EXPERIMENTS / "robustness_results.json")


def robustness_notes() -> str:
    """Human-readable summary of the noise-robustness ablation for the UI."""
    rob = load_robustness()
    if not rob:
        return "No robustness notes available."
    baseline = rob.get("models", {}).get("1dcnn_zscore_baseline", {})
    noisy = rob.get("models", {}).get("1dcnn_zscore_noisy015", {})
    b_cfg = {c["label"]: c for c in baseline.get("configs", [])}
    n_cfg = {c["label"]: c for c in noisy.get("configs", [])}
    clean_b = b_cfg.get("clean", {}).get("dl_wear_mae")
    gauss_b = b_cfg.get("gaussian sd=0.5*rms", {}).get("dl_wear_mae")
    gauss_n = n_cfg.get("gaussian sd=0.5*rms", {}).get("dl_wear_mae")
    parts = [
        "Noise-robustness ablation (test n=300, seed 42):",
        f"- zscore 1D-CNN clean wear MAE: **{clean_b:.3f}**" if clean_b else "",
        f"- zscore baseline at sd=0.5·rms: wear MAE **{gauss_b:.3f}** → "
        f"**{gauss_n:.3f}** with `--train-noise-sd 0.15` (noisy variant)."
        if gauss_b and gauss_n
        else "",
        "- `*_normnone` variants preserve amplitude (best clean accuracy); "
        "`*_noisy` variants trade clean accuracy for sensor corruption tolerance.",
    ]
    return "\n".join(p for p in parts if p)


#: Friendly model name -> (benchmark family key, DL metrics artifact stem).
_BENCH_FAMILY = {
    "xgboost": ("xgboost", None),
    "1dcnn": ("1dcnn", "dl_1dcnn"),
    "1dcnn_normnone": ("1dcnn", "dl_1dcnn_normnone"),
    "1dcnn_noisy": ("1dcnn", "dl_1dcnn_noisy"),
    "1dcnn_noisy01": ("1dcnn", "dl_1dcnn_noisy01"),
    "fusion": ("fusion", "dl_fusion"),
    "fusion_normnone": ("fusion", "dl_fusion_normnone"),
    "fusion_noisy": ("fusion", "dl_fusion_noisy"),
    "fusion_noisy01": ("fusion", "dl_fusion_noisy01"),
    "spectrogram": ("spectrogram", "dl_spectrogram"),
}


def typical_latency_ms(model_name: str) -> float | None:
    """Return a representative p50 single-chunk latency (ms) for a model name.

    DL variants share their family's ONNX benchmark; XGBoost uses its measured
    end-to-end (DSP extract + predict) p50. Returns ``None`` when unavailable.
    """
    bench = load_benchmark()
    family, _ = _BENCH_FAMILY.get(model_name, (None, None))
    if family is None:
        return None
    if family == "xgboost":
        xgb = bench.get("xgboost", {})
        return xgb.get("end_to_end_p50_ms")
    fam = bench.get("models", {}).get(family, {})
    single = fam.get("single", {})
    return single.get("p50_ms")


def model_metric_summary(model_name: str) -> dict[str, Any]:
    """Compact accuracy/latency summary for a model, for the inventory cards.

    Returns keys: ``wear_mae``, ``health_f1``, ``latency_ms``, ``n_params``,
    ``normalize_for_dl``, ``notes`` (any subset may be ``None``).
    """
    out: dict[str, Any] = {
        "wear_mae": None,
        "health_f1": None,
        "latency_ms": typical_latency_ms(model_name),
        "n_params": None,
        "notes": None,
    }
    if model_name == "xgboost":
        b = load_baseline_metrics()
        out["wear_mae"] = b.get("regression", {}).get("wear_level", {}).get("mae")
        out["health_f1"] = b.get("classification", {}).get("health_state", {}).get("macro_f1")
        out["notes"] = "DSP + thermal + context features"
        return out

    _, stem = _BENCH_FAMILY.get(model_name, (None, None))
    if stem:
        m = load_dl_metrics(stem)
        dl = m.get("dl_metrics", {})
        out["wear_mae"] = dl.get("regression", {}).get("wear_level", {}).get("mae")
        out["health_f1"] = dl.get("classification", {}).get("macro_f1")
        out["n_params"] = m.get("n_params")
        out["notes"] = f"normalize_for_dl={m.get('normalize_for_dl', '?')}"
    return out
