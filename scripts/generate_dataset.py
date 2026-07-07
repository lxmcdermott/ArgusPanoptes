"""Generate a labeled, physics-informed synthetic dataset for Argus Panoptes.

Samples a realistic machining parameter space, runs the vibration + thermal
simulators for each operating point, attaches labels and full metadata, and
writes a **partitioned Parquet dataset** (partitioned by ``alloy`` / ``wear_bin``)
plus a lightweight ``manifest.parquet`` for fast querying.

Raw waveforms are stored as ``float32`` list columns alongside the tabular
metadata so a single Parquet read yields both features and signals.

Optionally (``--extract-features``) the DSP :class:`~dsp.signal_processor.SignalProcessor`
is run on every vibration waveform and a handful of thermal statistics are
computed, and the resulting scalar features are merged into ``manifest.parquet``
(the raw waveform record columns are left untouched). This is **opt-in** and off
by default, so the default schema/behavior is unchanged.

Usage
-----
    python scripts/generate_dataset.py --num-samples 500
    python scripts/generate_dataset.py --num-samples 2000 --output-dir data/synthetic_v1 \
        --seed 7 --flush-every 250
    python scripts/generate_dataset.py --num-samples 100 --duration-s 3.0
    python scripts/generate_dataset.py --num-samples 300 --output-dir data/test_dsp_v1 \
        --extract-features
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

# Allow running as a plain script from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sensors import SawVibrationSimulator, ThermalSimulator, load_config  # noqa: E402
from sensors.config import SensorConfig  # noqa: E402

try:  # tqdm is a listed dependency, but degrade gracefully if missing.
    from tqdm import tqdm
except Exception:  # pragma: no cover
    def tqdm(iterable, **_kwargs):  # type: ignore[misc]
        return iterable

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("argus.datagen")

WAVEFORM_COLS = ("vibration_waveform", "thermal_waveform")
_N_WEAR_BINS = 5


# --------------------------------------------------------------------------- #
# Parameter sampling
# --------------------------------------------------------------------------- #
def _wear_bin(wear: float) -> str:
    idx = min(int(wear * _N_WEAR_BINS), _N_WEAR_BINS - 1)
    lo = idx / _N_WEAR_BINS
    hi = (idx + 1) / _N_WEAR_BINS
    return f"{lo:.1f}-{hi:.1f}"


def sample_operating_point(
    cfg: SensorConfig, rng: np.random.Generator, fixed_duration: float | None
) -> dict[str, Any]:
    """Draw one operating point from the config's sampling ranges."""
    s = cfg.machining.sampling
    duration = fixed_duration if fixed_duration is not None else float(
        rng.uniform(*s.duration_s)
    )
    return {
        "alloy": str(rng.choice(s.alloys)),
        "blade_speed_sfpm": float(rng.uniform(*s.blade_speed_sfpm)),
        "feed_per_tooth_mm": float(rng.uniform(*s.feed_per_tooth_mm)),
        "depth_mm": float(rng.uniform(*s.depth_mm)),
        "kerf_width_mm": float(rng.uniform(*s.kerf_width_mm)),
        "num_teeth": int(rng.integers(s.num_teeth[0], s.num_teeth[1] + 1)),
        "wear": float(rng.uniform(*s.wear)),
        "duration_s": duration,
    }


# --------------------------------------------------------------------------- #
# Row assembly
# --------------------------------------------------------------------------- #
def _thermal_feature_stats(temp: np.ndarray, fs_hz: float) -> dict[str, float]:
    """A few cheap DSP-derived thermal statistics (heating dynamics rise w/ wear).

    ``therm_std_c`` captures thermal roughness / transient content and
    ``therm_slope_c_per_s`` the mean cut-zone heating rate (both grow as friction
    heat increases with wear). These complement the closed-form thermal metadata
    already carried in the manifest (mean/max/steady-state/rise).
    """
    n = temp.size
    std_c = float(np.std(temp))
    if n >= 2 and fs_hz > 0:
        tt = np.arange(n, dtype=np.float64) / fs_hz
        slope = float(np.polyfit(tt, temp.astype(np.float64), 1)[0])
    else:  # pragma: no cover - guarded by simulator (n>=2)
        slope = 0.0
    return {"therm_std_c": std_c, "therm_slope_c_per_s": slope}


def build_row(
    vib: SawVibrationSimulator,
    therm: ThermalSimulator,
    op: dict[str, Any],
    sample_id: int,
    seed: int,
    processor: "Any | None" = None,
) -> dict[str, Any]:
    """Run both simulators for one operating point and assemble a dataset row.

    When ``processor`` (a ``dsp.SignalProcessor``) is supplied, its scalar
    features are extracted from the vibration waveform (``vib_td_*`` / ``vib_fd_*``
    columns) and a few thermal statistics are added (``therm_*``). This is
    additive: the base columns are identical whether or not ``processor`` is set.
    """
    params = {k: op[k] for k in ("alloy", "blade_speed_sfpm", "feed_per_tooth_mm",
                                 "depth_mm", "kerf_width_mm", "num_teeth")}
    wear = op["wear"]
    duration = op["duration_s"]

    _, accel, vm = vib.generate(duration_s=duration, params=params, wear=wear, seed=seed)
    _, temp, tm = therm.generate(duration_s=duration, params=params, wear=wear, seed=seed)

    # Combined anomaly = mechanical (wear) OR thermal.
    anomaly = bool(vm["label_anomaly_flag"] or tm.get("label_thermal_anomaly_flag", False))

    row: dict[str, Any] = {
        "sample_id": sample_id,
        "seed": seed,
        "timestamp_utc": vm["timestamp_utc"],
        "simulator_version": vm["simulator_version"],
        # --- operating point ---
        "saw_type": vm["saw_type"],
        "alloy": op["alloy"],
        "blade_speed_sfpm": op["blade_speed_sfpm"],
        "num_teeth": op["num_teeth"],
        "feed_per_tooth_mm": op["feed_per_tooth_mm"],
        "depth_mm": op["depth_mm"],
        "kerf_width_mm": op["kerf_width_mm"],
        "wear": wear,
        "wear_bin": _wear_bin(wear),
        "duration_s": duration,
        # --- derived physics ---
        "tooth_pass_freq_hz": vm["tooth_pass_freq_hz"],
        "rpm": vm["rpm"],
        "cutting_velocity_m_s": vm["cutting_velocity_m_s"],
        "material_removal_rate_mm3_s": vm["material_removal_rate_mm3_s"],
        "specific_energy_j_per_mm3": vm["specific_energy_j_per_mm3"],
        "avg_cutting_power_w": vm["avg_cutting_power_w"],
        "avg_cutting_force_n": vm["avg_cutting_force_n"],
        "per_tooth_force_n": vm["per_tooth_force_n"],
        "force_wear_multiplier": vm["force_wear_multiplier"],
        # --- vibration signal stats ---
        "vib_fs_hz": vm["fs_hz"],
        "vib_n_samples": vm["n_samples"],
        "vib_rms_g": vm["signal_rms_g"],
        "vib_kurtosis": vm["signal_kurtosis"],
        "vib_crest_factor": vm["signal_crest_factor"],
        "vib_peak_g": vm["signal_peak_g"],
        "vib_clipped": vm["clipped"],
        # --- thermal signal stats ---
        "thermal_fs_hz": tm["fs_hz"],
        "thermal_n_samples": tm["n_samples"],
        "mean_temp_c": tm["mean_temp_c"],
        "steady_state_temp_c": tm["steady_state_temp_c"],
        "max_temp_c": tm["max_temp_c"],
        "temp_rise_c": tm["temp_rise_c"],
        "q_in_w": tm["q_in_w"],
        # --- labels ---
        "label_wear_level": vm["label_wear_level"],
        "label_rul_fraction": vm["label_rul_fraction"],
        "label_rul_cycles": vm["label_rul_cycles"],
        "label_cycle_time_factor": vm["label_cycle_time_factor"],
        "label_quality_score": vm["label_quality_score"],
        "label_health_state": vm["label_health_state"],
        "label_anomaly_flag": anomaly,
        "label_thermal_anomaly_flag": bool(tm.get("label_thermal_anomaly_flag", False)),
        # --- raw waveforms (float32 list columns) ---
        "vibration_waveform": accel.astype(np.float32),
        "thermal_waveform": temp.astype(np.float32),
    }

    # --- optional DSP-extracted scalar features (opt-in, additive) ---
    if processor is not None:
        vib_features = processor.process(accel, fs=vm["fs_hz"], metadata=vm)["features"]
        row.update({f"vib_{k}": v for k, v in vib_features.items()})
        row.update(_thermal_feature_stats(temp, tm["fs_hz"]))

    return row


# --------------------------------------------------------------------------- #
# Parquet writing
# --------------------------------------------------------------------------- #
def _rows_to_table(rows: list[dict[str, Any]]) -> pa.Table:
    """Build a pyarrow Table, encoding waveform columns as list<float32>."""
    columns: dict[str, Any] = {}
    keys = [k for k in rows[0] if k not in WAVEFORM_COLS]
    for key in keys:
        columns[key] = pa.array([r[key] for r in rows])
    for wcol in WAVEFORM_COLS:
        columns[wcol] = pa.array(
            [r[wcol].tolist() for r in rows], type=pa.list_(pa.float32())
        )
    return pa.table(columns)


def flush_batch(rows: list[dict[str, Any]], records_dir: Path, batch_idx: int) -> None:
    """Write one batch to the partitioned Parquet dataset (under records_dir)."""
    if not rows:
        return
    table = _rows_to_table(rows)
    pq.write_to_dataset(
        table,
        root_path=str(records_dir),
        partition_cols=["alloy", "wear_bin"],
        basename_template=f"part-{batch_idx:04d}-{{i}}-{uuid.uuid4().hex[:8]}.parquet",
        existing_data_behavior="overwrite_or_ignore",
        compression="snappy",
    )


# --------------------------------------------------------------------------- #
# Summary
# --------------------------------------------------------------------------- #
def _feature_columns(df: "Any") -> list[str]:
    """DSP-extracted feature columns (present only with --extract-features)."""
    prefixes = ("vib_td_", "vib_fd_", "therm_std_c", "therm_slope_c_per_s")
    return [c for c in df.columns if any(c.startswith(p) for p in prefixes)]


def print_summary(manifest_rows: list[dict[str, Any]], out_dir: Path, records_dir: Path) -> None:
    import pandas as pd

    df = pd.DataFrame(manifest_rows)
    logger.info("=" * 68)
    logger.info("DATASET SUMMARY")
    logger.info("=" * 68)
    logger.info("Samples: %d", len(df))
    logger.info("Alloys : %s", df["alloy"].value_counts().to_dict())
    logger.info("Health : %s", df["label_health_state"].value_counts().to_dict())
    logger.info(
        "Anomalies: %d (%.1f%%)",
        int(df["label_anomaly_flag"].sum()),
        100.0 * df["label_anomaly_flag"].mean(),
    )

    logger.info("Wear distribution by bin:")
    for b, c in df["wear_bin"].value_counts().sort_index().items():
        logger.info("    %s : %d", b, c)

    logger.info("Label correlations with wear:")
    for col in ("vib_rms_g", "mean_temp_c", "steady_state_temp_c",
                "label_quality_score", "label_cycle_time_factor", "avg_cutting_force_n"):
        corr = df["wear"].corr(df[col])
        logger.info("    corr(wear, %-24s) = %+.3f", col, corr)

    # --- DSP feature report (only when --extract-features was used) ---
    feature_cols = _feature_columns(df)
    if feature_cols:
        logger.info("-" * 68)
        logger.info("DSP FEATURES: %d extracted per sample (columns: %s ... )",
                    len(feature_cols), ", ".join(feature_cols[:4]))
        targets = [
            ("label_wear_level", "wear_level"),
            ("label_quality_score", "quality_score"),
            ("label_cycle_time_factor", "cycle_time_factor"),
        ]
        for tcol, tname in targets:
            if tcol not in df.columns:
                continue
            corrs = (
                df[feature_cols]
                .apply(lambda s: s.corr(df[tcol]))
                .dropna()
                .abs()
                .sort_values(ascending=False)
            )
            logger.info("Top feature correlations with %s:", tname)
            for name, val in corrs.head(6).items():
                signed = df[name].corr(df[tcol])
                logger.info("    corr(%-28s, %-16s) = %+.3f", name, tname, signed)

    parquet_files = list(records_dir.rglob("*.parquet"))
    total_bytes = sum(p.stat().st_size for p in parquet_files)
    total_bytes += (out_dir / "manifest.parquet").stat().st_size
    logger.info("On-disk size: %.2f MB across %d record files + manifest",
                total_bytes / 1e6, len(parquet_files))
    logger.info("Output dir  : %s", out_dir)
    logger.info("  records/  : partitioned Parquet (alloy / wear_bin) with waveforms")
    logger.info("  manifest.parquet : tabular metadata + labels (no waveforms)")
    logger.info("=" * 68)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def _write_dl_config(
    dl_output_dir: Path, processor: "Any", normalize_for_dl: str, target_len: int | None
) -> None:
    """Persist the DL input recipe so the DataLoader reproduces identical inputs.

    The spectrograms / normalized waveforms are computed **on-the-fly** by
    ``models/train_dl.py`` (leveraging the raw ``float32`` waveform columns in
    ``records/``) to avoid multi-hundred-MB spectrogram storage. This JSON simply
    records the exact STFT / normalization config used so training is reproducible.
    """
    import json

    dl_output_dir.mkdir(parents=True, exist_ok=True)
    recipe = {
        "processor_version": processor.version,
        "normalize_for_dl": normalize_for_dl,
        "target_len": target_len,
        "stft": {
            "nperseg": processor.stft_cfg.nperseg,
            "overlap": processor.stft_cfg.overlap,
            "window": processor.stft_cfg.window,
            "log_scale": processor.stft_cfg.log_scale,
        },
        "note": "Spectrograms/normalized waveforms are computed on-the-fly in the "
        "DataLoader from records/ waveforms; nothing extra is stored.",
    }
    path = dl_output_dir / "dl_config.json"
    with path.open("w", encoding="utf-8") as fh:
        json.dump(recipe, fh, indent=2)
    logger.info("Wrote DL input recipe -> %s", path)


def generate(
    num_samples: int,
    output_dir: Path,
    seed: int,
    duration_s: float | None,
    flush_every: int,
    config_path: Path | None,
    extract_features: bool = False,
    processor_config_path: Path | None = None,
    compute_spectrogram: bool = False,
    normalize_for_dl: str = "zscore",
    dl_output_dir: Path | None = None,
) -> None:
    cfg = load_config(config_path)
    vib = SawVibrationSimulator(cfg)
    therm = ThermalSimulator(cfg)

    processor = None
    if extract_features:
        from dsp import SignalProcessor  # local import: keeps default path dsp-free
        from dsp.config import load_processor_config

        # Override the DL normalization (used only by the DL convenience methods /
        # DataLoader) without touching the tabular feature path or its schema.
        proc_cfg = load_processor_config(
            str(processor_config_path) if processor_config_path else None,
            overrides={"dl": {"normalize_for_dl": normalize_for_dl}},
        )
        processor = SignalProcessor(proc_cfg)
        logger.info("DSP feature extraction ENABLED (processor v%s)", processor.version)

    output_dir.mkdir(parents=True, exist_ok=True)
    records_dir = output_dir / "records"
    records_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    logger.info("Generating %d samples -> %s (base seed=%d)", num_samples, output_dir, seed)
    t0 = time.perf_counter()

    batch: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    batch_idx = 0

    for i in tqdm(range(num_samples), desc="samples", unit="smp"):
        op = sample_operating_point(cfg, rng, duration_s)
        row = build_row(vib, therm, op, sample_id=i, seed=seed + i, processor=processor)
        batch.append(row)
        manifest_rows.append({k: v for k, v in row.items() if k not in WAVEFORM_COLS})

        if len(batch) >= flush_every:
            flush_batch(batch, records_dir, batch_idx)
            batch_idx += 1
            batch = []

    flush_batch(batch, records_dir, batch_idx)

    # Lightweight manifest (no waveforms) for fast queries / stats.
    manifest_table = pa.Table.from_pylist(manifest_rows)
    pq.write_table(manifest_table, output_dir / "manifest.parquet", compression="snappy")

    elapsed = time.perf_counter() - t0
    logger.info("Done in %.1fs (%.1f ms/sample)", elapsed, 1e3 * elapsed / max(1, num_samples))
    print_summary(manifest_rows, output_dir, records_dir)

    # --- DL-readiness recipe (opt-in; nothing large is stored) ---
    if compute_spectrogram:
        if processor is None:
            from dsp import SignalProcessor
            from dsp.config import load_processor_config

            processor = SignalProcessor(
                load_processor_config(overrides={"dl": {"normalize_for_dl": normalize_for_dl}})
            )
        _write_dl_config(dl_output_dir or output_dir, processor, normalize_for_dl, target_len=None)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate the Argus Panoptes synthetic dataset.")
    parser.add_argument("--num-samples", type=int, default=500, help="Number of samples to generate.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "synthetic_v1",
        help="Output directory for the partitioned Parquet dataset.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Base RNG seed (reproducible).")
    parser.add_argument(
        "--duration-s",
        type=float,
        default=None,
        help="Fixed recording duration (s). If omitted, sampled from config range.",
    )
    parser.add_argument("--flush-every", type=int, default=200, help="Rows per Parquet flush.")
    parser.add_argument(
        "--config", type=Path, default=None, help="Optional path to an alternative sensor_specs.yaml."
    )
    parser.add_argument(
        "--extract-features",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run the DSP SignalProcessor on each vibration waveform and merge "
        "scalar features (vib_td_*/vib_fd_*/therm_*) into manifest.parquet. "
        "Opt-in; OFF by default so the default schema is unchanged.",
    )
    parser.add_argument(
        "--processor-config",
        type=Path,
        default=None,
        help="Optional path to an alternative dsp/processor_config.yaml (with --extract-features).",
    )
    parser.add_argument(
        "--compute-spectrogram",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Mark the dataset DL-ready and write dl_config.json (STFT + normalize "
        "recipe). Spectrograms are computed on-the-fly by models/train_dl.py, so "
        "nothing large is stored.",
    )
    parser.add_argument(
        "--normalize-for-dl",
        type=str,
        default="zscore",
        choices=["none", "zscore", "peak", "rms"],
        help="Per-chunk normalization recipe recorded for the DL paths (default zscore).",
    )
    parser.add_argument(
        "--dl-output-dir",
        type=Path,
        default=None,
        help="Where to write dl_config.json (defaults to --output-dir).",
    )
    args = parser.parse_args()

    if args.num_samples <= 0:
        parser.error("--num-samples must be positive")

    generate(
        num_samples=args.num_samples,
        output_dir=args.output_dir,
        seed=args.seed,
        duration_s=args.duration_s,
        flush_every=args.flush_every,
        config_path=args.config,
        extract_features=args.extract_features,
        processor_config_path=args.processor_config,
        compute_spectrogram=args.compute_spectrogram,
        normalize_for_dl=args.normalize_for_dl,
        dl_output_dir=args.dl_output_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
