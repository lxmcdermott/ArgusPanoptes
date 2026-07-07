# Argus Panoptes — Industrial Perception Stack

> A runnable, multi-modal **industrial perception prototype** for aluminum
> sawing and CNC machining cells. It owns the perception layer — **vibration,
> thermal (and vision hooks)** — that feeds accurate data into job costing,
> cycle-time prediction, and nesting optimization (making the downstream
> **cost & nesting optimizer** accurate).
>
> Built with heavy emphasis on **signal processing for blade-wear and
> cut-condition monitoring** using **physics-informed synthetic data**.

Named after the hundred-eyed, ever-watchful giant of Greek myth — always
watching the factory floor.

---

## Status

| Layer                                   | Module          | Status                              |
| --------------------------------------- | --------------- | ----------------------------------- |
| **Synthetic data generator**            | `sensors/`      | ✅ **v1 complete**                  |
| **DSP & feature extraction**            | `dsp/`          | ✅ **v1 implemented (Day 2)**       |
| ML pipeline & experiments               | `models/`       | 🚧 XGBoost baseline + ablations (Day 2) |
| Inference, FastAPI, Streamlit           | `app/`          | 🚧 scaffold (Day 4–5)               |
| Docker / edge                           | `deployment/`   | 🚧 scaffold (Day 6)                 |

This repository currently delivers a **production-quality v1 of the `sensors/`
and `dsp/` modules**: physics-informed vibration + thermal simulators, labels,
validation, a Parquet dataset-generation pipeline, a modular `SignalProcessor`
that extracts time/frequency features (including tooth-pass-relative band
energies), and interpretable **XGBoost baselines + ablations** on those features.
The remaining layers are scaffolded so the one-week plan can proceed immediately.

---

## Architecture

```mermaid
flowchart LR
    subgraph physical [Physical layer - simulated]
        SAW["Saw / CNC cell"]
        ACC["IEPE accelerometer<br/>blade guide / spindle"]
        IR["IR pyrometer<br/>cut zone"]
        SAW --> ACC
        SAW --> IR
    end

    subgraph edge [Edge layer]
        SIM["sensors/<br/>SawVibrationSimulator<br/>ThermalSimulator"]
        DSP["dsp/<br/>SignalProcessor<br/>RMS, PSD, STFT"]
        ML["models/<br/>XGBoost, 1D-CNN, fusion<br/>ONNX"]
        LOG[("data/<br/>Parquet logs")]
    end

    subgraph integration [Integration layer]
        API["app/<br/>FastAPI /infer /batch<br/>StreamingPerceptor"]
        PLC["Mock PLC / OPC-UA"]
    end

    subgraph ops [User / ops layer]
        DASH["Streamlit dashboard<br/>live viz, KPIs, alerts"]
        COST["Cost and nesting optimizer<br/>cost / cycle-time / nesting"]
    end

    ACC --> SIM
    IR --> SIM
    SIM --> DSP --> ML --> API
    SIM --> LOG
    ML --> LOG
    API --> COST
    API --> DASH
    PLC --> API
```

---

## Quickstart

```bash
# 1. Install (Python 3.11+)
pip install -r requirements.txt

# 2. Validate the physics simulators (prints sanity metrics + saves plots)
python scripts/validate_simulators.py

# 3. Run the test suite
pytest

# 4. Generate a labeled synthetic dataset (Parquet)
python scripts/generate_dataset.py --num-samples 500 --output-dir data/synthetic_v1

# 5. (Optional) Generate with DSP features + train the XGBoost baseline
pip install -e ".[ml]"   # scikit-learn, xgboost, joblib
python scripts/generate_dataset.py --num-samples 300 --output-dir data/synthetic_v1 --extract-features
python models/baseline.py --data-dir data/synthetic_v1
```

Outputs:

- Validation plots → `experiments/plots/`
- Dataset → `data/synthetic_v1/`:
  - `records/` — Parquet partitioned by `alloy` / `wear_bin`, with raw
    `vibration_waveform` and `thermal_waveform` list columns.
  - `manifest.parquet` — tabular metadata + labels only (fast to query).

Reading it back:

```python
import pandas as pd, pyarrow as pa, pyarrow.dataset as ds

# Fast metadata/label queries + stats:
meta = pd.read_parquet("data/synthetic_v1/manifest.parquet")

# Waveforms with predicate pushdown (alloy values are numeric-looking,
# so pass an explicit string partition schema):
part = ds.partitioning(
    schema=pa.schema([("alloy", pa.string()), ("wear_bin", pa.string())]),
    flavor="hive",
)
d = ds.dataset("data/synthetic_v1/records", partitioning=part, format="parquet")
table = d.to_table(filter=ds.field("wear_bin") == "0.8-1.0")
```

---

## The `sensors/` module (v1 deliverable)

Physics-informed generators for **blade-wear and cut-condition monitoring**:

- **`SawVibrationSimulator`** — 40.96 kHz acceleration (g) with tooth-pass
  frequency + harmonics, wear-modulated impact amplitude and broadband noise,
  configurable structural modes and sensor noise. TPF is derived *exactly* from
  saw kinematics; impact amplitude follows a **force ≈ specific-energy × chip-area**
  model that rises with wear.
- **`ThermalSimulator`** — lumped first-order cut-zone temperature model where
  wear scales the friction-heat term (100–400 °C band for aluminum).
- **Auto labels** — wear, RUL, cycle-time factor, quality score, health state,
  anomaly flag — plus rich metadata for every recording.

See [`sensors/README.md`](sensors/README.md) for the full physics write-up,
mounting realism, and usage.

---

## Repository layout

```
ArgusPanoptes/
├── sensors/            # ✅ physics-informed synthetic signal generation (v1)
│   ├── sensor_specs.yaml
│   ├── config.py       # pydantic config + loader
│   ├── utils.py        # kinematics, force model, signal & label helpers
│   ├── vibration_simulator.py
│   ├── thermal_simulator.py
│   └── README.md
├── dsp/                # ✅ SignalProcessor: features + STFT (v1, Day 2)
│   ├── processor_config.yaml
│   ├── config.py       # pydantic config + loader
│   └── signal_processor.py
├── models/             # 🚧 XGBoost baseline + ablations (Day 2)
│   └── baseline.py
├── app/                # 🚧 FastAPI + Streamlit (scaffold)
├── deployment/         # 🚧 Dockerfile + compose (scaffold)
├── experiments/        # notebooks + generated plots
├── scripts/
│   ├── validate_simulators.py
│   └── generate_dataset.py
├── tests/              # pytest suites for the simulators
├── data/               # generated Parquet (git-ignored)
├── requirements.txt / pyproject.toml
└── README.md
```

---

## Tech stack

Python 3.11+ · NumPy · SciPy (signal, fft, welch) · Pandas · PyArrow (Parquet) ·
Pydantic · PyYAML · Matplotlib · pytest.
ML/app dependencies (PyTorch, ONNX, FastAPI, Streamlit, Plotly) are deferred to
later days of the plan and intentionally kept out of the core install.

---

## Capabilities

This prototype demonstrates **ownership of the full perception stack**
(sensors on saws/CNC), **vibration for blade wear** and **thermal for cut
conditions**, an **end-to-end pipeline** (sensor → DSP → labeling → ML →
edge/cloud inference → monitoring), **Parquet data capture** for ML/ops, **clean
API interfaces** feeding downstream cost/nesting models, **experiments/ablations**,
and a **builder mindset** shipping a quality v1 fast.

---

## Roadmap

Day 1 ✅ sensors + validation → Day 2 ✅ DSP features + dataset integration +
XGBoost baseline + ablations → Day 3 DL (1D-CNN + spectrogram) + ONNX → Day 4
streaming + FastAPI → Day 5 Streamlit + cost/nesting integration mock → Day 6
experiments + Docker/edge → Day 7 polish + demo. Future: swap simulators for a
real DAQ (Pi + MPU6050 + MLX90640) and add vision depth.
