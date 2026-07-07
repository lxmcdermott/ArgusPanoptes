"""XGBoost baselines + feature-group ablations for Argus Panoptes.

Trains interpretable gradient-boosted-tree baselines on the physics-informed,
DSP-derived scalar features in ``manifest.parquet`` (produced by
``scripts/generate_dataset.py --extract-features``):

* **Regression** - blade ``wear_level`` (primary), ``cycle_time_factor``,
  ``quality_score``  (MAE / RMSE / R2).
* **Classification** - ``health_state`` (healthy / monitor / warning / critical)
  (accuracy, macro-F1, per-class report).
* **Ablation** - time-domain-only vs frequency-domain-only vs all feature groups,
  reporting the metric deltas that quantify each group's contribution.

Why XGBoost for the v1 baseline? It is fast to train, handles the moderate-size
tabular feature matrix and missing values natively, needs little tuning, and -
crucially for a condition-monitoring artifact - exposes **feature importances**
that we can sanity-check against the underlying physics (tooth-pass band energy,
RMS, envelope statistics should dominate a wear model).

Feature policy (leakage-aware)
------------------------------
Only *observable* quantities are used as features: DSP time/frequency features
(``vib_td_*`` / ``vib_fd_*``), observable thermal statistics, and machine
setpoints / kinematics that are known independently of wear (blade speed, teeth,
feed, depth, kerf, TPF, RPM, cutting velocity, MRR). Quantities the force model
computes *from* wear (specific energy, cutting force/power, force multiplier,
steady-state temperature, heat input) and every ``label_*`` column are excluded
to avoid target leakage.

Usage
-----
    python models/baseline.py --data-dir data/synthetic_v1
    python models/baseline.py --data-dir data/test_dsp_v1 --seed 42
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
import pandas as pd

# Allow running as a plain script from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")  # headless / no display needed
import matplotlib.pyplot as plt  # noqa: E402

from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    classification_report,
    f1_score,
    mean_absolute_error,
    r2_score,
    root_mean_squared_error,
)
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.preprocessing import LabelEncoder  # noqa: E402
from xgboost import XGBClassifier, XGBRegressor  # noqa: E402
import joblib  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("argus.baseline")

# --------------------------------------------------------------------------- #
# Feature policy
# --------------------------------------------------------------------------- #
#: Operating-point / kinematics context known independently of wear (no leakage).
CONTEXT_FEATURES: tuple[str, ...] = (
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
#: Observable thermal statistics (measured by the IR pyrometer path).
THERMAL_FEATURES: tuple[str, ...] = (
    "mean_temp_c",
    "max_temp_c",
    "temp_rise_c",
    "therm_std_c",
    "therm_slope_c_per_s",
)

REGRESSION_TARGETS: tuple[str, ...] = (
    "label_wear_level",
    "label_cycle_time_factor",
    "label_quality_score",
)
CLASSIFICATION_TARGET = "label_health_state"

XGB_REG_PARAMS: dict[str, Any] = {
    "n_estimators": 500,
    "max_depth": 5,
    "learning_rate": 0.05,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "reg_lambda": 1.0,
    "n_jobs": -1,
    "early_stopping_rounds": 40,
    "eval_metric": "rmse",
}
XGB_CLF_PARAMS: dict[str, Any] = {
    "n_estimators": 500,
    "max_depth": 5,
    "learning_rate": 0.05,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "reg_lambda": 1.0,
    "n_jobs": -1,
    "early_stopping_rounds": 40,
    "eval_metric": "mlogloss",
}


# --------------------------------------------------------------------------- #
# Data loading & feature grouping
# --------------------------------------------------------------------------- #
def load_manifest(data_dir: Path) -> pd.DataFrame:
    manifest = data_dir / "manifest.parquet"
    if not manifest.is_file():
        raise FileNotFoundError(f"manifest.parquet not found in {data_dir}")
    df = pd.read_parquet(manifest)
    logger.info("Loaded manifest: %d rows x %d cols from %s", len(df), df.shape[1], manifest)
    return df


def feature_groups(df: pd.DataFrame) -> dict[str, list[str]]:
    """Return ordered feature-name lists per group, restricted to present columns."""
    td = [c for c in df.columns if c.startswith("vib_td_")]
    fd = [c for c in df.columns if c.startswith("vib_fd_")]
    if not td or not fd:
        raise SystemExit(
            "No DSP feature columns (vib_td_* / vib_fd_*) found in the manifest.\n"
            "Regenerate the dataset with:  "
            "python scripts/generate_dataset.py --extract-features"
        )
    thermal = [c for c in THERMAL_FEATURES if c in df.columns]
    context = [c for c in CONTEXT_FEATURES if c in df.columns]
    return {
        "time": td,
        "freq": fd,
        "thermal": thermal,
        "context": context,
        "all": td + fd + thermal + context,
    }


# --------------------------------------------------------------------------- #
# Training helpers
# --------------------------------------------------------------------------- #
def _split_train_val(X: pd.DataFrame, y: pd.Series, seed: int, stratify: Any = None):
    """Carve a validation set out of the training data for early stopping."""
    return train_test_split(X, y, test_size=0.2, random_state=seed, stratify=stratify)


def train_regressor(
    X_tr: pd.DataFrame, y_tr: pd.Series, seed: int
) -> tuple[XGBRegressor, dict[str, float]]:
    X_fit, X_val, y_fit, y_val = _split_train_val(X_tr, y_tr, seed)
    model = XGBRegressor(random_state=seed, **XGB_REG_PARAMS)
    model.fit(X_fit, y_fit, eval_set=[(X_val, y_val)], verbose=False)
    return model, {"best_iteration": int(getattr(model, "best_iteration", 0) or 0)}


def evaluate_regression(model: XGBRegressor, X_te: pd.DataFrame, y_te: pd.Series) -> dict[str, float]:
    pred = model.predict(X_te)
    return {
        "mae": float(mean_absolute_error(y_te, pred)),
        "rmse": float(root_mean_squared_error(y_te, pred)),
        "r2": float(r2_score(y_te, pred)),
    }


def train_classifier(
    X_tr: pd.DataFrame, y_tr: np.ndarray, seed: int, n_classes: int
) -> XGBClassifier:
    strat = y_tr if n_classes > 1 else None
    try:
        X_fit, X_val, y_fit, y_val = _split_train_val(X_tr, pd.Series(y_tr), seed, stratify=strat)
    except ValueError:  # too few members in some class to stratify
        X_fit, X_val, y_fit, y_val = _split_train_val(X_tr, pd.Series(y_tr), seed)
    model = XGBClassifier(
        random_state=seed, num_class=n_classes, objective="multi:softprob", **XGB_CLF_PARAMS
    )
    model.fit(X_fit, y_fit, eval_set=[(X_val, y_val)], verbose=False)
    return model


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def plot_feature_importance(
    model: XGBRegressor | XGBClassifier,
    feature_names: list[str],
    title: str,
    out_path: Path,
    top_n: int = 15,
) -> list[tuple[str, float]]:
    importances = np.asarray(model.feature_importances_, dtype=float)
    order = np.argsort(importances)[::-1][:top_n]
    names = [feature_names[i] for i in order]
    vals = importances[order]

    fig, ax = plt.subplots(figsize=(9, 6))
    y_pos = np.arange(len(names))[::-1]
    ax.barh(y_pos, vals, color="#2b7bba")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("XGBoost gain importance (normalized)")
    ax.set_title(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    logger.info("Saved importance plot -> %s", out_path)
    return list(zip(names, (float(v) for v in vals)))


# --------------------------------------------------------------------------- #
# Ablation
# --------------------------------------------------------------------------- #
def run_ablation(
    df: pd.DataFrame,
    groups: dict[str, list[str]],
    target: str,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    seed: int,
) -> dict[str, dict[str, float]]:
    """Retrain on time-only / freq-only / all feature groups; report metrics."""
    configs = {
        "time_only": groups["time"],
        "freq_only": groups["freq"],
        "all": groups["all"],
    }
    y = df[target]
    results: dict[str, dict[str, float]] = {}
    for name, cols in configs.items():
        X = df[cols]
        model, _ = train_regressor(X.iloc[train_idx], y.iloc[train_idx], seed)
        results[name] = evaluate_regression(model, X.iloc[test_idx], y.iloc[test_idx])
        results[name]["n_features"] = len(cols)
    return results


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def run(
    data_dir: Path,
    models_dir: Path,
    plots_dir: Path,
    results_path: Path,
    metrics_path: Path,
    seed: int,
) -> dict[str, Any]:
    df = load_manifest(data_dir)
    groups = feature_groups(df)
    logger.info(
        "Feature groups: time=%d freq=%d thermal=%d context=%d (all=%d)",
        len(groups["time"]), len(groups["freq"]), len(groups["thermal"]),
        len(groups["context"]), len(groups["all"]),
    )

    models_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    # One shared stratified split (on wear_bin) reused across targets/ablations.
    idx = np.arange(len(df))
    strat = df["wear_bin"] if "wear_bin" in df.columns else None
    try:
        train_idx, test_idx = train_test_split(idx, test_size=0.2, random_state=seed, stratify=strat)
    except ValueError:
        train_idx, test_idx = train_test_split(idx, test_size=0.2, random_state=seed)
    logger.info("Split: %d train / %d test (stratified on wear_bin)", len(train_idx), len(test_idx))

    X_all = df[groups["all"]]
    report: dict[str, Any] = {
        "seed": seed,
        "n_samples": int(len(df)),
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "feature_groups": {k: v for k, v in groups.items()},
        "regression": {},
        "classification": {},
        "ablation": {},
    }

    # ---- Regression targets ----
    for target in REGRESSION_TARGETS:
        if target not in df.columns:
            continue
        short = target.replace("label_", "")
        logger.info("=" * 68)
        logger.info("REGRESSION target: %s", short)
        y = df[target]
        model, info = train_regressor(X_all.iloc[train_idx], y.iloc[train_idx], seed)
        metrics = evaluate_regression(model, X_all.iloc[test_idx], y.iloc[test_idx])
        logger.info(
            "  MAE=%.4f  RMSE=%.4f  R2=%.4f  (best_iter=%d)",
            metrics["mae"], metrics["rmse"], metrics["r2"], info["best_iteration"],
        )
        top = plot_feature_importance(
            model, groups["all"],
            title=f"XGBoost feature importance - {short}",
            out_path=plots_dir / f"baseline_xgboost_feature_importance_{short}.png",
        )
        logger.info("  Top features: %s", ", ".join(f"{n}={v:.3f}" for n, v in top[:8]))

        joblib.dump(model, models_dir / f"xgb_reg_{short}.joblib")
        report["regression"][short] = {
            **metrics,
            "best_iteration": info["best_iteration"],
            "top_features": top[:10],
        }

    # ---- Ablation (on the primary wear target) ----
    logger.info("=" * 68)
    logger.info("ABLATION (target: wear_level)")
    ablation = run_ablation(df, groups, "label_wear_level", train_idx, test_idx, seed)
    all_mae = ablation["all"]["mae"]
    for name, m in ablation.items():
        delta = m["mae"] - all_mae
        logger.info(
            "  %-10s: MAE=%.4f RMSE=%.4f R2=%.4f  (nfeat=%d, dMAE vs all=%+.4f)",
            name, m["mae"], m["rmse"], m["r2"], int(m["n_features"]), delta,
        )
        m["delta_mae_vs_all"] = delta
    report["ablation"]["wear_level"] = ablation

    # ---- Classification ----
    if CLASSIFICATION_TARGET in df.columns:
        logger.info("=" * 68)
        logger.info("CLASSIFICATION target: health_state")
        le = LabelEncoder()
        y_enc = le.fit_transform(df[CLASSIFICATION_TARGET].astype(str))
        n_classes = len(le.classes_)
        clf = train_classifier(X_all.iloc[train_idx], y_enc[train_idx], seed, n_classes)
        y_pred = clf.predict(X_all.iloc[test_idx])
        y_true = y_enc[test_idx]
        all_labels = np.arange(n_classes)
        acc = float(accuracy_score(y_true, y_pred))
        f1m = float(
            f1_score(y_true, y_pred, average="macro", labels=all_labels, zero_division=0)
        )
        cls_report = classification_report(
            y_true,
            y_pred,
            labels=all_labels,
            target_names=list(le.classes_),
            zero_division=0,
            output_dict=True,
        )
        logger.info("  Accuracy=%.4f  macro-F1=%.4f", acc, f1m)
        logger.info("  Classes: %s", list(le.classes_))
        top_clf = plot_feature_importance(
            clf, groups["all"],
            title="XGBoost feature importance - health_state",
            out_path=plots_dir / "baseline_xgboost_feature_importance_health_state.png",
        )
        joblib.dump({"model": clf, "label_encoder": le}, models_dir / "xgb_clf_health_state.joblib")
        report["classification"]["health_state"] = {
            "accuracy": acc,
            "macro_f1": f1m,
            "classes": list(le.classes_),
            "report": cls_report,
            "top_features": top_clf[:10],
        }

    # ---- Persist metrics JSON + markdown summary ----
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=float)
    logger.info("Saved metrics JSON -> %s", metrics_path)

    _write_results_md(report, results_path, data_dir)
    logger.info("Saved results summary -> %s", results_path)
    return report


def _write_results_md(report: dict[str, Any], path: Path, data_dir: Path) -> None:
    lines: list[str] = []
    lines.append("# Argus Panoptes - XGBoost Baseline Results\n")
    lines.append(
        f"- Dataset: `{data_dir}`  |  samples: {report['n_samples']} "
        f"(train {report['n_train']} / test {report['n_test']})  |  seed: {report['seed']}\n"
    )
    g = report["feature_groups"]
    lines.append(
        f"- Feature groups: time={len(g['time'])}, freq={len(g['freq'])}, "
        f"thermal={len(g['thermal'])}, context={len(g['context'])} "
        f"(**all={len(g['all'])}**)\n"
    )

    lines.append("\n## Regression\n")
    lines.append("| Target | MAE | RMSE | R2 | best_iter |")
    lines.append("| --- | --- | --- | --- | --- |")
    for tgt, m in report["regression"].items():
        lines.append(
            f"| {tgt} | {m['mae']:.4f} | {m['rmse']:.4f} | {m['r2']:.4f} | {m['best_iteration']} |"
        )

    if "wear_level" in report["regression"]:
        lines.append("\n### Top features - wear_level\n")
        for n, v in report["regression"]["wear_level"]["top_features"][:10]:
            lines.append(f"- `{n}`: {v:.4f}")

    lines.append("\n## Ablation (wear_level, MAE)\n")
    lines.append("| Feature set | n_features | MAE | RMSE | R2 | dMAE vs all |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for name, m in report["ablation"].get("wear_level", {}).items():
        lines.append(
            f"| {name} | {int(m['n_features'])} | {m['mae']:.4f} | {m['rmse']:.4f} | "
            f"{m['r2']:.4f} | {m['delta_mae_vs_all']:+.4f} |"
        )

    if "health_state" in report["classification"]:
        c = report["classification"]["health_state"]
        lines.append("\n## Classification - health_state\n")
        lines.append(f"- Accuracy: **{c['accuracy']:.4f}**  |  macro-F1: **{c['macro_f1']:.4f}**")
        lines.append(f"- Classes: {c['classes']}\n")
        lines.append("### Top features - health_state\n")
        for n, v in c["top_features"][:10]:
            lines.append(f"- `{n}`: {v:.4f}")

    lines.append("\n---\n_Generated by `models/baseline.py`._\n")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Train XGBoost baselines + ablations.")
    parser.add_argument("--data-dir", type=Path, default=root / "data" / "synthetic_v1")
    parser.add_argument("--models-dir", type=Path, default=root / "experiments" / "models")
    parser.add_argument("--plots-dir", type=Path, default=root / "experiments" / "plots")
    parser.add_argument("--results", type=Path, default=root / "experiments" / "baseline_results.md")
    parser.add_argument("--metrics", type=Path, default=root / "experiments" / "baseline_metrics.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    t0 = time.perf_counter()
    run(
        data_dir=args.data_dir,
        models_dir=args.models_dir,
        plots_dir=args.plots_dir,
        results_path=args.results,
        metrics_path=args.metrics,
        seed=args.seed,
    )
    logger.info("Baseline complete in %.1fs", time.perf_counter() - t0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
