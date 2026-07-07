"""Noise-robustness ablation: XGBoost (tabular DSP) vs 1D-CNN (raw waveform).

Corrupts the held-out **test** vibration waveforms with three realistic sensor
degradations and measures how blade-wear MAE and health-state macro-F1 degrade
for each model, on the *same* test rows used everywhere else:

* **Gaussian noise** - additive white noise at a fraction of the signal RMS
  (electrical / thermal sensor noise, low SNR).
* **Sensor drift** - a slow low-frequency ramp+sine baseline wander (mounting /
  thermal drift, DC bias creep).
* **Quantization** - reduced ADC bit depth (cheap / saturated data acquisition).

For each corrupted waveform the XGBoost path *re-extracts* its DSP features (the
band-pass detrend should reject drift, and integrated band energies average out
white noise), while the 1D-CNN path re-normalizes and runs ONNX inference. Both
are compared against their own clean-signal scores.

Usage
-----
    python experiments/robustness_ablation.py --data-dir data/dl_v1
    python experiments/robustness_ablation.py --data-dir data/dl_v1 --max-samples 300
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


# --------------------------------------------------------------------------- #
# Corruptions (operate on the raw float64/float32 waveform)
# --------------------------------------------------------------------------- #
def add_gaussian(w: np.ndarray, level: float, rng: np.random.Generator) -> np.ndarray:
    rms = float(np.sqrt(np.mean(w**2))) + 1e-12
    return (w + rng.normal(0.0, level * rms, size=w.shape)).astype(np.float32)


def add_drift(w: np.ndarray, level: float, rng: np.random.Generator) -> np.ndarray:
    rms = float(np.sqrt(np.mean(w**2))) + 1e-12
    n = w.size
    t = np.linspace(0.0, 1.0, n)
    # Slow ramp + a sub-Hz sine wander (both well below the 100 Hz band-pass corner).
    drift = level * rms * (t + 0.5 * np.sin(2 * np.pi * 0.5 * t))
    return (w + drift).astype(np.float32)


def quantize(w: np.ndarray, bits: int, rng: np.random.Generator) -> np.ndarray:
    lo, hi = float(np.min(w)), float(np.max(w))
    if hi <= lo:
        return w.astype(np.float32)
    levels = 2**bits - 1
    q = np.round((w - lo) / (hi - lo) * levels)
    return (lo + q / levels * (hi - lo)).astype(np.float32)


def corrupt(w: np.ndarray, kind: str, level: float, rng: np.random.Generator) -> np.ndarray:
    if kind == "clean":
        return w.astype(np.float32)
    if kind == "gaussian":
        return add_gaussian(w, level, rng)
    if kind == "drift":
        return add_drift(w, level, rng)
    if kind == "quantization":
        return quantize(w, int(level), rng)
    raise ValueError(kind)  # pragma: no cover


CONFIGS: list[tuple[str, float, str]] = [
    ("clean", 0.0, "clean"),
    ("gaussian", 0.1, "gaussian sd=0.1*rms"),
    ("gaussian", 0.25, "gaussian sd=0.25*rms"),
    ("gaussian", 0.5, "gaussian sd=0.5*rms"),
    ("drift", 0.5, "drift 0.5*rms"),
    ("drift", 1.0, "drift 1.0*rms"),
    ("quantization", 8, "quantize 8-bit"),
    ("quantization", 4, "quantize 4-bit"),
]


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    f1s = []
    for c in range(n_classes):
        tp = int(np.sum((y_pred == c) & (y_true == c)))
        fp = int(np.sum((y_pred == c) & (y_true != c)))
        fn = int(np.sum((y_pred != c) & (y_true == c)))
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) else 0.0)
    return float(np.mean(f1s))


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def run(data_dir: Path, out_json: Path, plots_dir: Path, seed: int, max_samples: int | None) -> dict:
    import joblib
    import pandas as pd  # noqa: F401

    from dsp import SignalProcessor
    from models.baseline import feature_groups
    from models.dl_data import _crop_or_pad, _stratified_split, load_waveforms
    from models.onnx_inference import ONNXPerceptor

    root = Path(__file__).resolve().parents[1]
    manifest = __import__("pandas").read_parquet(data_dir / "manifest.parquet")
    manifest = manifest.sort_values("sample_id").reset_index(drop=True)
    _, waveforms = load_waveforms(data_dir / "records")

    strat = manifest["wear_bin"].to_numpy() if "wear_bin" in manifest.columns else None
    _, _, test_idx = _stratified_split(len(manifest), strat, seed)
    if max_samples is not None and max_samples < len(test_idx):
        test_idx = test_idx[:max_samples]

    sp = SignalProcessor()
    fs = float(manifest["vib_fs_hz"].iloc[0])
    groups = feature_groups(manifest)
    all_cols = groups["all"]
    vib_cols = [c for c in all_cols if c.startswith("vib_td_") or c.startswith("vib_fd_")]

    # Trained artifacts (from models/baseline.py on this dataset).
    reg = joblib.load(root / "experiments" / "models" / "xgb_reg_wear_level.joblib")
    clf_bundle = joblib.load(root / "experiments" / "models" / "xgb_clf_health_state.joblib")
    clf, le = clf_bundle["model"], clf_bundle["label_encoder"]
    class_names = list(le.classes_)
    n_classes = len(class_names)
    cls_to_idx = {c: i for i, c in enumerate(class_names)}

    perc = ONNXPerceptor(root / "experiments" / "models" / "dl_1dcnn.onnx")
    target_len = 16384

    # Ground truth for the test rows.
    y_wear = manifest["label_wear_level"].to_numpy()[test_idx]
    y_health = manifest["label_health_state"].astype(str).map(cls_to_idx).to_numpy()[test_idx]

    # Base feature frame for the test rows (thermal + context stay fixed under
    # vibration noise; only vib_* columns are recomputed per corruption).
    X_base = manifest.iloc[test_idx][all_cols].reset_index(drop=True).copy()

    results: dict[str, Any] = {"configs": [], "class_names": class_names,
                               "n_test": int(len(test_idx))}
    print("=" * 84)
    print("ARGUS PANOPTES - noise-robustness ablation (XGBoost DSP vs 1D-CNN)")
    print(f"test rows: {len(test_idx)}   fs: {fs:.0f} Hz")
    print("=" * 84)
    print(f"{'corruption':22s} {'xgb wear MAE':>13s} {'dl wear MAE':>13s} "
          f"{'xgb F1':>9s} {'dl F1':>9s}")
    print("-" * 84)

    t0 = time.perf_counter()
    for kind, level, label in CONFIGS:
        rng = np.random.default_rng(seed)
        X_noisy = X_base.copy()
        dl_wear = np.empty(len(test_idx), dtype=np.float64)
        dl_health = np.empty(len(test_idx), dtype=np.int64)

        for j, idx in enumerate(test_idx):
            w = waveforms[idx].astype(np.float64)
            wc = corrupt(w, kind, level, rng)
            tpf = float(manifest["tooth_pass_freq_hz"].iloc[idx])
            feats = sp.process(wc, fs=fs, metadata={"tooth_pass_freq_hz": tpf, "fs_hz": fs})["features"]
            for col in vib_cols:
                X_noisy.at[j, col] = feats[col[len("vib_"):]]

            wn = _crop_or_pad(sp.get_normalized_waveform(wc, fs), target_len)
            out = perc.infer(wn[None, None, :].astype(np.float32))
            dl_wear[j] = float(out["regression"][0, 0])
            dl_health[j] = int(np.argmax(out["health_logits"][0]))

        xgb_wear_pred = reg.predict(X_noisy)
        xgb_health_pred = clf.predict(X_noisy)
        xgb_mae = float(np.mean(np.abs(xgb_wear_pred - y_wear)))
        dl_mae = float(np.mean(np.abs(dl_wear - y_wear)))
        xgb_f1 = _macro_f1(y_health, np.asarray(xgb_health_pred), n_classes)
        dl_f1 = _macro_f1(y_health, dl_health, n_classes)

        results["configs"].append({
            "kind": kind, "level": level, "label": label,
            "xgb_wear_mae": xgb_mae, "dl_wear_mae": dl_mae,
            "xgb_health_f1": xgb_f1, "dl_health_f1": dl_f1,
        })
        print(f"{label:22s} {xgb_mae:13.4f} {dl_mae:13.4f} {xgb_f1:9.4f} {dl_f1:9.4f}")

    print("-" * 84)
    print(f"done in {time.perf_counter() - t0:.1f}s")

    _plot(results, plots_dir)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    print(f"Saved robustness JSON -> {out_json}")
    return results


def _plot(results: dict, plots_dir: Path) -> None:
    labels = [c["label"] for c in results["configs"]]
    x = np.arange(len(labels))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    ax1.bar(x - 0.2, [c["xgb_wear_mae"] for c in results["configs"]], 0.4, label="XGBoost", color="#2b7bba")
    ax1.bar(x + 0.2, [c["dl_wear_mae"] for c in results["configs"]], 0.4, label="1D-CNN", color="#e07b39")
    ax1.set(ylabel="wear MAE (lower=better)", title="Wear MAE vs corruption")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=40, ha="right", fontsize=8)
    ax1.legend()
    ax1.grid(alpha=0.3, axis="y")

    ax2.bar(x - 0.2, [c["xgb_health_f1"] for c in results["configs"]], 0.4, label="XGBoost", color="#2b7bba")
    ax2.bar(x + 0.2, [c["dl_health_f1"] for c in results["configs"]], 0.4, label="1D-CNN", color="#e07b39")
    ax2.set(ylabel="health macro-F1 (higher=better)", title="Health macro-F1 vs corruption")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=40, ha="right", fontsize=8)
    ax2.legend()
    ax2.grid(alpha=0.3, axis="y")

    fig.suptitle("Argus Panoptes - noise-robustness ablation", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    plots_dir.mkdir(parents=True, exist_ok=True)
    out = plots_dir / "robustness_ablation.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"Saved robustness plot -> {out}")


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Noise-robustness ablation.")
    parser.add_argument("--data-dir", type=Path, default=root / "data" / "dl_v1")
    parser.add_argument("--out", type=Path, default=root / "experiments" / "robustness_results.json")
    parser.add_argument("--plots-dir", type=Path, default=root / "experiments" / "plots")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()
    run(args.data_dir, args.out, args.plots_dir, args.seed, args.max_samples)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
