# Argus Panoptes — Session Change Log

**Date:** 2026-07-06
**Scope:** Functional code only (synthetic-signals `sensors/` module v1, scripts,
tests, and dataset pipeline). Documentation-only edits are intentionally excluded.
**Environment:** Python 3.13 (Windows / PowerShell). Deps: NumPy, SciPy, Pandas,
PyArrow, Pydantic, PyYAML, Matplotlib, tqdm, pytest.

---

## 1. Configuration layer

### `sensors/sensor_specs.yaml` (new)
Authoritative, human-editable spec with named-unit keys:
- **Vibration:** IEPE accelerometer, 100 mV/g, bandwidth 20 kHz, resonance 35 kHz,
  noise floor 0.001 g rms, `fs_hz=40960`, single-axis, `clip_g=50`. Added
  force→acceleration coupling and stochastic fields: `accel_g_per_kN=8.0`,
  `broadband_base_g=0.05`, `broadband_wear_gain=3.0`, `broadband_exponent=1.0`.
  Structural modes (1800 Hz Q6 gain2, 4200 Hz Q8 gain1.4), tooth-pulse spec
  (`exp_decay_sine`, ring 2000 Hz), harmonic weights `[1.0, 0.5, 0.3, 0.15]`.
- **Thermal:** IR pyrometer @ cut zone, `fs_hz=200`, `noise_std_c=0.5`. Lumped
  model: `ambient_c=25`, `heat_partition=0.10`, `loss_coefficient_w_per_c=2.2`,
  `thermal_mass_j_per_c=9.0`, `friction_power_gain_w=260`, `max_temp_c=600`.
- **Machining:** circular saw default (Ø350 mm, 80 teeth, 800 SFPM, feed 0.12 mm,
  depth 25 mm, kerf 3 mm), alloys 6061/7075 with base specific energy +
  wear-energy gain, force coefficients, and parameter-sampling ranges.
- **Labels:** RUL max cycles, cycle-time/quality gains, anomaly thresholds.

### `sensors/config.py` (new)
Typed **pydantic v2** models (`extra="forbid"` to catch typos): `SensorConfig`,
`VibrationConfig`, `ThermalConfig`, `MachiningConfig`, `LabelConfig`, plus
`StructuralMode`, `PulseSpec`, `ThermalModel`, `AlloySpec`, `MachiningSampling`.
- Field-level validation (ranges, allowed `pulse.shape`, non-negative harmonics,
  `saw_type ∈ {circular, band}`).
- `load_config(path=None, *, overrides=None)`: resolves path from arg →
  `ARGUS_SENSOR_SPECS` env var → packaged default; deep-merges `overrides`;
  validates. `SensorConfig.alloy(name)` convenience accessor.

---

## 2. Shared physics & label helpers

### `sensors/utils.py` (new)
Pure, side-effect-free functions (RNG passed explicitly):
- **Kinematics:** `rpm_from_sfpm`, `cutting_velocity_m_s`, `calculate_tpf`.
  - Circular: `RPM = SFPM / (π · D_ft)`, `TPF = RPM · num_teeth / 60`.
  - Band: `TPF = V_s[ft/min] · 12 · TPI / 60`.
- **Machining physics:** `specific_energy(base_u, gain, wear) = base_u·(1+gain·wear)`;
  `compute_force_model(...) -> ForceModel` implementing the chain
  `feed_velocity = feed·TPF → MRR = kerf·depth·feed_velocity → P = u·MRR →
  F_avg = P/V_c → F_tooth = F_avg·(1 + force_wear_gain·wear + friction_wear_gain·wear)`.
  (Unit note: `1 J/mm³ = 1000 N/mm²`.) Returns a frozen dataclass with TPF, RPM,
  velocities, MRR, specific energy, power, avg/per-tooth force, multiplier, extras.
- **Signal helpers:** `generate_colored_noise` (FFT-shaped 1/f^α, scaled to target
  RMS), `apply_sensor_noise`, `rms`, `kurtosis` (excess/Fisher), `crest_factor`.
- **Labels:** `generate_labels(...)` producing `wear_level`, `rul_fraction`
  (`1-wear`), `rul_cycles`, `cycle_time_factor` (`1 + gain·wear + force term`),
  `quality_score` (clipped `1 - wear_gain·wear - kurt_gain·norm_kurt`),
  `health_state` (healthy/monitor/warning/critical), `anomaly_flag`
  (wear ≥ threshold OR temp anomaly), plus `thermal_anomaly_flag` when temp given.

---

## 3. Simulators

### `sensors/vibration_simulator.py` (new) — `SawVibrationSimulator`, `__version__="0.1.0"`
- Ctor accepts `SensorConfig | dict | None` (validated/loaded accordingly).
- `generate(duration_s=5.0, params=None, wear=0.0, seed=None) -> (t, accel_g, metadata)`:
  1. Builds `ForceModel`; peak budget `peak_g = accel_g_per_kN · F_tooth[kN]`.
  2. Slow ~5% force envelope (3–9 Hz).
  3. **Tooth-impact train**: one unit-peak shaped pulse per engagement at TPF
     (`_pulse_kernel`: `half_sine` or damped `exp_decay_sine`), amplitude-modulated
     by force envelope + 3% per-strike jitter.
  4. **Harmonic comb** (secondary, 0.25·peak) at `k·TPF` below Nyquist.
  5. **Wear-modulated broadband**: pink noise `rms = base·(1+gain·wear)`, boosted
     near structural modes via narrow Butterworth band-pass mixing.
  6. **Sensor noise** (white @ noise floor). Clip-check against `clip_g`.
  - Returns rich metadata: sensor summary, resolved operating point, derived
    physics, signal stats (RMS/kurtosis/crest/peak), and `label_*` fields.
- `stream(duration_s, chunk_s, ...)`: chunk generator (stub for `StreamingPerceptor`).

### `sensors/thermal_simulator.py` (new) — `ThermalSimulator`, `__version__="0.1.0"`
- `generate(...) -> (t, temp_C, metadata)`.
- Lumped first-order model `C·dT/dt = Q_in − k·(T−T_amb)` with
  `Q_in = heat_partition·P_cut + friction_power_gain·wear`; solved in closed form
  `T(t) = T_ss + (T_amb − T_ss)·e^(−t/τ)`, `T_ss = T_amb + Q_in/k`, `τ = C/k`.
- Adds white IR sensor noise; clamps to `max_temp_c`. Metadata includes Q terms,
  time constant, steady-state/mean/max/final temps, and labels (temp-aware).

### `sensors/__init__.py` (new)
Public API export: `SawVibrationSimulator`, `ThermalSimulator`, `load_config`,
config models, `__version__="0.1.0"`.

---

## 4. Scripts

### `scripts/validate_simulators.py` (new)
Numerical sanity checks + 4-panel diagnostic plot (waveform, Welch PSD with TPF
harmonic markers, thermal transients, wear-sensitivity). CLI: `--outdir`,
`--no-show`, `--no-plots`. Returns non-zero exit on failure.
Checks: TPF detection < 1% error, RMS↑ with wear, temperature↑ with wear (in
range), finiteness/dtype/no-clip, seed reproducibility.

### `scripts/generate_dataset.py` (new)
Parameter-space sampler → runs both simulators → combined labels + metadata →
**partitioned Parquet**.
- Writes partitioned records under `<out>/records/` (partition by
  `alloy`/`wear_bin`, waveforms stored as `float32` list columns) + a
  waveform-free `<out>/manifest.parquet` for fast queries.
- CLI: `--num-samples`, `--output-dir`, `--seed`, `--duration-s`, `--flush-every`,
  `--config`. tqdm progress, logging, reproducible per-sample seeding.
- Summary report: alloy/health/anomaly counts, wear-bin distribution,
  correlations of wear vs RMS/temp/quality/cycle-time/force, on-disk size.

---

## 5. Tests (`tests/`, all passing — 32 cases)

- `conftest.py`: path setup + `config`/`vib`/`therm` fixtures.
- `test_vibration_simulator.py`: TPF exactness (circular + band), spectral TPF
  detection, RMS/spectral/broadband monotonicity in wear, reproducibility,
  length/dtype, finiteness, no-clip, metadata completeness, label ranges,
  high-wear anomaly, param override, stream coverage, dict construction.
- `test_thermal_simulator.py`: steady-state monotonicity, physical range/clamp,
  transient rise from ambient, approach to steady state, positive time constant,
  friction-heat scaling, reproducibility, length/dtype, finiteness, metadata
  completeness, thermal-anomaly label, input validation.

---

## 6. Functional bug fixes / tuning during the session

1. **Thermal calibration:** `heat_partition` retuned `0.75 → 0.10` so aluminum
   cut-zone temperatures land in the realistic ~100–400 °C band (most heat leaves
   with the chip/coolant on highly-conductive Al). Verified monotonic
   127.8 → 338.5 °C across wear.
2. **Dataset double-count fix:** moved partitioned Parquet under a `records/`
   subdirectory so `manifest.parquet` (at the dataset root) is no longer scanned
   as part of the partitioned dataset (previously inflated `alloy` filter counts
   ~2×). Updated summary accounting accordingly.
3. **Partition-read guidance:** confirmed `alloy` values (`6061`/`7075`) are
   numeric-looking; reads require an explicit string partition schema (verified
   round-trip: manifest 6061=23 == records 6061=23, total 50).

---

## 7. Verification snapshot

- `pytest`: 32 passed.
- `validate_simulators.py`: TPF error 0.23–0.38%; RMS 2.30→9.16 g and
  T_ss 127.8→338.5 °C monotonic; finiteness/dtype/clip/reproducibility PASS.
- Dataset (50 samples): ~122 ms/sample; corr(wear, quality)=−1.00,
  corr(wear, cycle_time)=+1.00; Parquet queryable via manifest + partitioned records.
