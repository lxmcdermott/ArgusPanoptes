# `sensors/` — Physics-Informed Synthetic Signal Generation

The `sensors/` package is the foundation of Argus Panoptes: it produces
**labeled, physics-informed vibration and thermal signals** for aluminum sawing
and CNC machining. These signals stand in for a real DAQ during development and
feed the DSP → ML → inference → cost/nesting pipeline.

> **Version:** `0.1.0` · **Status:** v1 complete, validated, tested.

---

## 1. Quickstart

```python
from sensors import SawVibrationSimulator, ThermalSimulator, load_config

cfg = load_config()                      # loads sensor_specs.yaml (validated)

vib = SawVibrationSimulator(cfg)
t, accel_g, vmeta = vib.generate(duration_s=3.0, wear=0.4, seed=7)

therm = ThermalSimulator(cfg)
t2, temp_c, tmeta = therm.generate(duration_s=3.0, wear=0.4, seed=7)

print(vmeta["tooth_pass_freq_hz"], vmeta["signal_rms_g"])
print(tmeta["mean_temp_c"], tmeta["label_wear_level"])
```

Override any operating point per call:

```python
t, accel_g, meta = vib.generate(
    duration_s=2.0,
    params={"alloy": "7075", "blade_speed_sfpm": 1000, "feed_per_tooth_mm": 0.25,
            "depth_mm": 40, "kerf_width_mm": 4, "num_teeth": 100},
    wear=0.8,
    seed=0,
)
```

Override config globally (deep-merged onto the YAML):

```python
cfg = load_config(overrides={"vibration": {"fs_hz": 20480}})
```

---

## 2. Sensor realism & mounting

| Modality  | Sensor                              | Mounting / focus                                                        | Sample rate |
| --------- | ----------------------------------- | ---------------------------------------------------------------------- | ----------- |
| Vibration | IEPE (ICP) piezo accelerometer, 100 mV/g | **Stud or magnet on the blade guide / spindle housing**, cutting-direction axis | 40.96 kHz   |
| Thermal   | Fast IR pyrometer                   | **Tool-chip interface / rake face** (the cut zone)                      | 200 Hz      |

These placements mirror standard industrial condition-monitoring practice: the
accelerometer sits on stiff structure near the cut so tooth-pass forcing couples
in cleanly (bandwidth to 20 kHz, mounted resonance ~35 kHz well above the band of
interest), and the pyrometer watches the hottest, most wear-sensitive spot.

All specifications live in [`sensor_specs.yaml`](./sensor_specs.yaml) and are
validated into typed [`SensorConfig`](./config.py) objects with `pydantic`.

---

## 3. Physics model

### 3.1 Kinematics — tooth-pass frequency (TPF)

TPF is computed **exactly** from saw kinematics (`sensors.utils.calculate_tpf`):

- **Circular saw:** `RPM = SFPM / (π · D_ft)`, then `TPF = RPM · num_teeth / 60`.
- **Band saw:** `TPF = V_s[ft/min] · 12[in/ft] · TPI / 60`.

Example (defaults: 800 SFPM, Ø350 mm, 80 teeth) → **RPM ≈ 221.8**, **TPF ≈ 295.7 Hz**
— i.e. the "hundreds of Hz" range typical of real saws.

### 3.2 Cutting mechanics — force ~ specific energy × chip area

Chain implemented in `sensors.utils.compute_force_model`
(units: `1 J/mm³ = 1000 N/mm²`):

1. `feed_velocity = feed_per_tooth · TPF`  (mm/s)
2. `MRR = kerf_width · depth · feed_velocity`  (mm³/s)
3. `u = base_u · (1 + wear_energy_gain · wear)`  (J/mm³) — dull edges rub / form
   built-up-edge on gummy aluminum, so specific energy **rises with wear**
4. `P = u · MRR`  (W)  →  `F_avg = P / V_c`  (N)
5. direct force/friction wear multiplier
   `m = 1 + force_wear_gain·wear + friction_wear_gain·wear`
6. `F_tooth = F_avg · m` drives the impact amplitudes

Defaults give ≈ 2.3 kW cutting power and ≈ 560 N average cutting force at a sharp
edge — realistic for an aluminum saw.

### 3.3 Vibration synthesis

`accel(t) = tooth-impact train  +  harmonic comb  +  wear-modulated broadband  +  sensor noise`

- **Tooth-impact train:** one shaped pulse per tooth engagement at TPF. Default
  pulse is a damped ring (`exp_decay_sine`), a good impact model; `half_sine` is
  also available. Amplitude = `accel_g_per_kN · F_tooth`, modulated by a slow
  force envelope plus small per-strike jitter. A periodic impact train is
  naturally rich in TPF **harmonics**.
- **Harmonic comb:** a secondary tonal comb at `k·TPF` (weights from
  `vibration.harmonics`) keeps spectral lines crisp for the DSP layer.
- **Wear-modulated broadband:** pink (`1/f`) Gaussian noise with
  `rms = broadband_base_g · (1 + broadband_wear_gain · wear)` — captures
  stochastic chip formation and rising friction. Shaped by configurable
  **structural modes** (narrow band-pass boosts) to emulate machine dynamics.
- **Sensor noise:** white Gaussian at `noise_floor_g_rms`.

Output is in **g**; the signal is clip-checked against the accelerometer
full-scale range (`clip_g`).

### 3.4 Thermal synthesis — lumped cut-zone model

First-order lumped model (`sensors.thermal_simulator`):

```
C · dT/dt = Q_in − k · (T − T_amb)
Q_in = heat_partition · P_cut + friction_power_gain · wear
```

solved in closed form `T(t) = T_ss + (T_amb − T_ss)·e^(−t/τ)` with
`T_ss = T_amb + Q_in/k` and `τ = C/k` (≈ 4 s). **Wear scales the friction-heat
term**, so cut-zone temperature is the primary cut-condition observable. Defaults
yield ~130 °C steady state at a sharp edge rising toward ~340 °C near end-of-life
— within the realistic 100–400 °C band for aluminum. Small IR sensor noise is
added and the trace is clamped below the aluminum melt point.

---

## 4. Labels & metadata

Deterministic, pure functions (`sensors.utils.generate_labels`) attach:

| Label                 | Meaning                                                    |
| --------------------- | --------------------------------------------------------- |
| `wear_level`          | 0 (sharp) … 1 (end-of-life), pass-through                 |
| `rul_fraction`        | remaining-useful-life fraction `1 − wear`                 |
| `rul_cycles`          | RUL in remaining cuts (`rul_max_cycles · (1 − wear)`)     |
| `cycle_time_factor`   | effective cycle-time multiplier ≥ 1 (higher forces slow)  |
| `quality_score`       | part-quality proxy in [0, 1] (↓ with wear & impulsiveness)|
| `health_state`        | `healthy` / `monitor` / `warning` / `critical`           |
| `anomaly_flag`        | high wear or thermal anomaly                              |

Every `generate()` call returns a **metadata dict** containing all input
parameters, resolved operating point, derived physics (TPF, RPM, MRR, specific
energy, cutting power/force, time constant, steady-state temp…), signal
statistics (RMS, kurtosis, crest factor, peak), a UTC timestamp, and the
simulator version — everything needed to log a queryable Parquet row.

---

## 5. How outputs feed the rest of the stack

```
sensors/  ──►  dsp/ (features)  ──►  models/ (XGBoost / CNN / fusion)  ──►  app/ (FastAPI + Streamlit)  ──►  cost/nesting
   │                                                                              
   └──►  scripts/generate_dataset.py  ──►  data/synthetic_v1/*.parquet (labeled dataset)
```

- **DSP:** raw `accel_g` / `temp_c` arrays are the direct inputs to the
  `SignalProcessor` (Welch PSD, band energies, STFT, etc.).
- **ML:** the Parquet dataset (features + labels + metadata) trains wear/RUL
  regression, health classification, and anomaly detection.
- **Integration:** predictions become `cycle_time_factor` / `quality_score`
  payloads for downstream cost/nesting models.

---

## 6. Limitations (v1)

- Fully **synthetic**: designed to be swapped for a real DAQ
  (Pi + MPU6050 + MLX90640) behind the same interface.
- Single-axis vibration (multi-axis planned; `axes` field reserved).
- Lumped (0-D) thermal model — no spatial gradients.
- Force/thermal magnitudes are physically **structured and monotonic**, tuned to
  realistic ranges rather than calibrated to a specific machine.
- `stream()` is a chunking stub for the forthcoming `StreamingPerceptor`.

---

## 7. Files

| File                     | Purpose                                                        |
| ------------------------ | ------------------------------------------------------------- |
| `sensor_specs.yaml`      | Authoritative sensor + machining + label configuration        |
| `config.py`              | Pydantic models + `load_config()` (validation, overrides, env)|
| `utils.py`               | Kinematics, force model, signal helpers, label functions      |
| `vibration_simulator.py` | `SawVibrationSimulator`                                        |
| `thermal_simulator.py`   | `ThermalSimulator`                                             |
| `__init__.py`            | Public API                                                    |
