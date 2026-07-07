"""Train / evaluate / export the Argus Panoptes DL models (Day 3).

Trains one of the three PyTorch models (:mod:`models.dl_models`) on the Parquet
dataset, evaluates it on a held-out test set, **compares against the XGBoost
baseline on the exact same test rows**, and saves a checkpoint, an ONNX
artifact, a metrics JSON, and a training-curve plot.

Usage
-----
    python models/train_dl.py --model 1dcnn      --data-dir data/dl_v1 --epochs 40
    python models/train_dl.py --model spectrogram --data-dir data/dl_v1 --epochs 40
    python models/train_dl.py --model fusion      --data-dir data/dl_v1 --epochs 40

Requires the ``dl`` extra (torch, onnx, onnxruntime) and, for the baseline
comparison, the ``ml`` extra (scikit-learn, xgboost).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import torch  # noqa: E402

from models.dl_data import prepare_dl_data  # noqa: E402
from models.dl_models import (  # noqa: E402
    DLModelConfig,
    evaluate_dl_model,
    example_inputs_for,
    export_to_onnx,
    get_model,
    set_seed,
    train_dl_model,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("argus.train_dl")

_MODEL_ALIASES = {"1dcnn": "1dcnn", "spectrogram": "spectrogram", "fusion": "fusion"}


# --------------------------------------------------------------------------- #
# XGBoost baseline on the SAME split (for an apples-to-apples comparison)
# --------------------------------------------------------------------------- #
def _baseline_on_split(
    manifest, train_idx: np.ndarray, test_idx: np.ndarray, seed: int
) -> dict[str, Any] | None:
    """Train XGBoost on the same non-test rows and evaluate on the same test rows."""
    try:
        from models.baseline import (
            CLASSIFICATION_TARGET,
            evaluate_regression,
            feature_groups,
            train_classifier,
            train_regressor,
        )
        from sklearn.metrics import accuracy_score, f1_score
        from sklearn.preprocessing import LabelEncoder
    except Exception as exc:  # pragma: no cover - ml extra missing
        logger.warning("Skipping XGBoost comparison (ml extra unavailable): %s", exc)
        return None

    groups = feature_groups(manifest)
    X_all = manifest[groups["all"]]
    out: dict[str, Any] = {"regression": {}, "classification": {}}
    reg_map = {
        "wear_level": "label_wear_level",
        "cycle_time_factor": "label_cycle_time_factor",
        "quality_score": "label_quality_score",
    }
    for short, col in reg_map.items():
        if col not in manifest.columns:
            continue
        y = manifest[col]
        model, _ = train_regressor(X_all.iloc[train_idx], y.iloc[train_idx], seed)
        out["regression"][short] = evaluate_regression(model, X_all.iloc[test_idx], y.iloc[test_idx])

    if CLASSIFICATION_TARGET in manifest.columns:
        le = LabelEncoder()
        y_enc = le.fit_transform(manifest[CLASSIFICATION_TARGET].astype(str))
        n_classes = len(le.classes_)
        clf = train_classifier(X_all.iloc[train_idx], y_enc[train_idx], seed, n_classes)
        y_pred = clf.predict(X_all.iloc[test_idx])
        y_true = y_enc[test_idx]
        labels = np.arange(n_classes)
        out["classification"] = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)),
        }
    return out


# --------------------------------------------------------------------------- #
# Plot
# --------------------------------------------------------------------------- #
def _plot_curves(history: dict[str, Any], title: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(history["train_loss"], label="train loss", color="#2b7bba")
    ax.plot(history["val_loss"], label="val loss", color="#e07b39")
    ax.set(xlabel="epoch", ylabel="multi-task loss", title=title)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    logger.info("Saved training curve -> %s", out_path)


# --------------------------------------------------------------------------- #
# Main run
# --------------------------------------------------------------------------- #
def run(
    model_name: str,
    data_dir: Path,
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
    target_len: int,
    device: str,
    models_dir: Path,
    plots_dir: Path,
    metrics_dir: Path,
    compare_baseline: bool,
) -> dict[str, Any]:
    set_seed(seed)
    t0 = time.perf_counter()
    logger.info("Preparing DL data (%s) from %s ...", model_name, data_dir)
    data = prepare_dl_data(
        data_dir, model_name, target_len=target_len, batch_size=batch_size, seed=seed
    )
    logger.info(
        "Split: %d fit / %d val / %d test (of %d)  fs=%.0f Hz",
        len(data["fit_idx"]), len(data["val_idx"]), len(data["test_idx"]),
        data["n_samples"], data["fs_hz"],
    )

    cfg = DLModelConfig(n_classes=len(data["class_names"]), thermal_dim=data["thermal_dim"])
    model = get_model(model_name, cfg)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("Model %s: %d parameters (input_kind=%s)", model_name, n_params, model.input_kind)

    t_train = time.perf_counter()
    history = train_dl_model(
        model,
        data["train_loader"],
        data["val_loader"],
        epochs=epochs,
        lr=lr,
        device=device,
        seed=seed,
    )
    train_time = time.perf_counter() - t_train
    logger.info("Trained in %.1fs (%d epochs, best val=%.4f)",
                train_time, history["epochs_run"], history["best_val_loss"])

    metrics = evaluate_dl_model(
        model, data["test_loader"], device=device, class_names=data["class_names"]
    )
    logger.info("=" * 68)
    logger.info("DL TEST METRICS (%s)", model_name)
    for tgt, m in metrics["regression"].items():
        logger.info("  reg %-18s MAE=%.4f RMSE=%.4f R2=%.4f", tgt, m["mae"], m["rmse"], m["r2"])
    c = metrics["classification"]
    logger.info("  clf health_state    Accuracy=%.4f macro-F1=%.4f", c["accuracy"], c["macro_f1"])

    # --- XGBoost comparison on identical test rows ---
    baseline_metrics = None
    if compare_baseline:
        train_all_idx = np.concatenate([data["fit_idx"], data["val_idx"]])
        baseline_metrics = _baseline_on_split(
            data["manifest"], train_all_idx, data["test_idx"], seed
        )
        if baseline_metrics:
            logger.info("-" * 68)
            logger.info("XGBoost baseline (same test rows):")
            for tgt, m in baseline_metrics["regression"].items():
                logger.info("  reg %-18s MAE=%.4f RMSE=%.4f R2=%.4f",
                            tgt, m["mae"], m["rmse"], m["r2"])
            bc = baseline_metrics.get("classification", {})
            if bc:
                logger.info("  clf health_state    Accuracy=%.4f macro-F1=%.4f",
                            bc["accuracy"], bc["macro_f1"])

    # --- Artifacts ---
    models_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = models_dir / f"dl_{model_name}.pt"
    torch.save({"state_dict": model.state_dict(), "config": vars(cfg),
                "class_names": data["class_names"], "model_name": model_name}, ckpt_path)
    logger.info("Saved checkpoint -> %s", ckpt_path)

    onnx_path = models_dir / f"dl_{model_name}.onnx"
    seq_len = data["input_len"] or target_len
    n_freq, n_time = (data["input_shape"] or (513, 161))
    example = example_inputs_for(model, batch=1, seq_len=seq_len, n_freq=n_freq, n_time=n_time)
    export_to_onnx(model, onnx_path, example_input=example)
    logger.info("Exported ONNX -> %s", onnx_path)

    _plot_curves(history, f"DL training - {model_name}", plots_dir / f"dl_{model_name}_training_curve.png")

    report: dict[str, Any] = {
        "model": model_name,
        "n_params": int(n_params),
        "seed": seed,
        "epochs_run": history["epochs_run"],
        "train_time_s": train_time,
        "n_train": int(len(data["fit_idx"])),
        "n_val": int(len(data["val_idx"])),
        "n_test": int(len(data["test_idx"])),
        "input_len": data["input_len"],
        "input_shape": data["input_shape"],
        "class_names": data["class_names"],
        "dl_metrics": metrics,
        "baseline_metrics": baseline_metrics,
        "history": {"train_loss": history["train_loss"], "val_loss": history["val_loss"]},
        "artifacts": {"checkpoint": str(ckpt_path), "onnx": str(onnx_path)},
    }
    metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = metrics_dir / f"dl_{model_name}_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=float)
    logger.info("Saved metrics JSON -> %s", metrics_path)
    logger.info("Total wall time: %.1fs", time.perf_counter() - t0)
    return report


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Train Argus Panoptes DL models.")
    parser.add_argument("--model", choices=list(_MODEL_ALIASES), required=True)
    parser.add_argument("--data-dir", type=Path, default=root / "data" / "dl_v1")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-len", type=int, default=16384,
                        help="Fixed waveform length (center-crop/pad). ~0.4 s @ 40.96 kHz.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--models-dir", type=Path, default=root / "experiments" / "models")
    parser.add_argument("--plots-dir", type=Path, default=root / "experiments" / "plots")
    parser.add_argument("--metrics-dir", type=Path, default=root / "experiments")
    parser.add_argument("--no-compare-baseline", action="store_true",
                        help="Skip the XGBoost comparison on the same split.")
    args = parser.parse_args()

    run(
        args.model,
        args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        target_len=args.target_len,
        device=args.device,
        models_dir=args.models_dir,
        plots_dir=args.plots_dir,
        metrics_dir=args.metrics_dir,
        compare_baseline=not args.no_compare_baseline,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
