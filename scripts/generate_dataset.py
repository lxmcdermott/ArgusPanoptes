"""Generate a labeled, physics-informed synthetic dataset for Argus Panoptes.

Samples a realistic machining parameter space, runs the vibration + thermal
simulators for each operating point, attaches labels and full metadata, and
writes a **partitioned Parquet dataset** (partitioned by ``alloy`` / ``wear_bin``)
plus a lightweight ``manifest.parquet`` for fast querying.

Raw waveforms are stored as ``float32`` list columns alongside the tabular
metadata so a single Parquet read yields both features and signals.

Usage
-----
    python scripts/generate_dataset.py --num-samples 500
    python scripts/generate_dataset.py --num-samples 2000 --output-dir data/synthetic_v1 \
        --seed 7 --flush-every 250
    python scripts/generate_dataset.py --num-samples 100 --duration-s 3.0
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
def build_row(
    vib: SawVibrationSimulator,
    therm: ThermalSimulator,
    op: dict[str, Any],
    sample_id: int,
    seed: int,
) -> dict[str, Any]:
    """Run both simulators for one operating point and assemble a dataset row."""
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
def generate(
    num_samples: int,
    output_dir: Path,
    seed: int,
    duration_s: float | None,
    flush_every: int,
    config_path: Path | None,
) -> None:
    cfg = load_config(config_path)
    vib = SawVibrationSimulator(cfg)
    therm = ThermalSimulator(cfg)

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
        row = build_row(vib, therm, op, sample_id=i, seed=seed + i)
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
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
