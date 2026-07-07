"""Benchmark ONNX Runtime inference latency for the Argus Panoptes DL models.

Times single-chunk (batch=1) and batched inference on **CPU** for the exported
1D-CNN / spectrogram-CNN / fusion ONNX artifacts, reporting p50 / p95 latency
and throughput (chunks/sec). For context it also times the tabular XGBoost path
end-to-end (DSP feature extraction + tree ``predict``), since that is the real
per-chunk cost of the baseline on the edge.

Edge targets: the lighter models should comfortably hit < 50 ms/chunk on CPU.

Usage
-----
    python scripts/benchmark_onnx.py --models-dir experiments/models
    python scripts/benchmark_onnx.py --chunk-len 40960 --batch 32 --iters 200

Notes on further edge optimization (not implemented here):
* **NVIDIA Jetson / TensorRT** - convert the .onnx with ``trtexec`` (FP16/INT8)
  or the ONNX Runtime TensorRT EP for 2-5x speedups on Orin-class devices.
* **Intel / OpenVINO** - use the ONNX Runtime OpenVINO EP or ``mo`` for CPU/iGPU
  acceleration; INT8 PTQ via NNCF for further gains.
* **ARM CPUs** - onnxruntime with the XNNPACK EP; quantize to INT8.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _time_fn(fn: Callable[[], Any], iters: int, warmup: int = 10) -> dict[str, float]:
    """Return latency stats (ms) for ``fn`` over ``iters`` timed calls."""
    for _ in range(warmup):
        fn()
    samples = np.empty(iters, dtype=np.float64)
    for i in range(iters):
        t0 = time.perf_counter()
        fn()
        samples[i] = (time.perf_counter() - t0) * 1e3  # ms
    return {
        "p50_ms": float(np.percentile(samples, 50)),
        "p95_ms": float(np.percentile(samples, 95)),
        "mean_ms": float(np.mean(samples)),
        "min_ms": float(np.min(samples)),
    }


def _rand(*shape: int) -> np.ndarray:
    return np.random.randn(*shape).astype(np.float32)


def _bench_onnx_model(
    onnx_path: Path, kind: str, chunk_len: int, spec_shape: tuple[int, int],
    thermal_dim: int, batch: int, iters: int,
) -> dict[str, Any] | None:
    if not onnx_path.is_file():
        return None
    from models.onnx_inference import ONNXPerceptor

    perc = ONNXPerceptor(onnx_path)

    def make_inputs(b: int) -> tuple[np.ndarray, ...]:
        if kind == "waveform":
            return (_rand(b, 1, chunk_len),)
        if kind == "spectrogram":
            return (_rand(b, 1, spec_shape[0], spec_shape[1]),)
        return (_rand(b, 1, chunk_len), _rand(b, thermal_dim))  # fusion

    single_inputs = make_inputs(1)
    batch_inputs = make_inputs(batch)
    single = _time_fn(lambda: perc.infer(*single_inputs), iters)
    batched = _time_fn(lambda: perc.infer(*batch_inputs), iters)
    return {
        "single": {**single, "throughput_cps": 1000.0 / single["p50_ms"]},
        "batched": {
            **batched,
            "batch": batch,
            "throughput_cps": batch * 1000.0 / batched["p50_ms"],
        },
    }


def _bench_xgboost(models_dir: Path, iters: int) -> dict[str, Any] | None:
    """Time DSP feature extraction + XGBoost predict for the tabular path."""
    joblib_path = models_dir / "xgb_reg_wear_level.joblib"
    try:
        import joblib
        from sensors import SawVibrationSimulator
        from dsp import SignalProcessor
    except Exception:  # pragma: no cover
        return None
    if not joblib_path.is_file():
        return None

    from models.baseline import CONTEXT_FEATURES, THERMAL_FEATURES

    model = joblib.load(joblib_path)
    sp = SignalProcessor()
    sim = SawVibrationSimulator()
    _, accel, meta = sim.generate(duration_s=1.0, wear=0.5, seed=0)

    dsp_stats = _time_fn(lambda: sp.process(accel, fs=meta["fs_hz"], metadata=meta), iters)

    # Build a single feature row matching the trained model's feature order.
    feats = sp.process(accel, fs=meta["fs_hz"], metadata=meta)["features"]
    n_feat = int(getattr(model, "n_features_in_", 0)) or (len(feats) + len(THERMAL_FEATURES) + len(CONTEXT_FEATURES))
    row = np.zeros((1, n_feat), dtype=np.float32)
    for i, v in enumerate(list(feats.values())[:n_feat]):
        row[0, i] = v
    predict = _time_fn(lambda: model.predict(row), iters)
    return {
        "dsp_extract": dsp_stats,
        "xgb_predict": predict,
        "end_to_end_p50_ms": dsp_stats["p50_ms"] + predict["p50_ms"],
    }


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Benchmark ONNX DL inference latency (CPU).")
    parser.add_argument("--models-dir", type=Path, default=root / "experiments" / "models")
    parser.add_argument(
        "--model",
        choices=["1dcnn", "spectrogram", "fusion", "all"],
        default="all",
        help="Which model artifact(s) to benchmark (default: all).",
    )
    parser.add_argument(
        "--artifact",
        type=str,
        default=None,
        help="Explicit ONNX stem under --models-dir (e.g. dl_fusion_noisy). "
        "Overrides --model when set.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="ONNX Runtime execution provider hint (cpu only in this script).",
    )
    parser.add_argument("--chunk-len", type=int, default=16384,
                        help="Waveform chunk length (samples). ~0.4 s @ 40.96 kHz.")
    parser.add_argument("--spec-freq", type=int, default=513)
    parser.add_argument("--spec-time", type=int, default=65)
    parser.add_argument("--thermal-dim", type=int, default=5)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--out", type=Path, default=root / "experiments" / "onnx_benchmark.json")
    args = parser.parse_args()

    kinds = {"1dcnn": "waveform", "spectrogram": "spectrogram", "fusion": "fusion"}
    if args.artifact:
        stem = args.artifact.replace(".onnx", "")
        name = stem.replace("dl_", "", 1) if stem.startswith("dl_") else stem
        bench_items = [(name, kinds.get(name.split("_")[0], "waveform"))]
    elif args.model == "all":
        bench_items = list(kinds.items())
    else:
        bench_items = [(args.model, kinds[args.model])]

    results: dict[str, Any] = {
        "config": {
            "chunk_len": args.chunk_len, "spec_shape": [args.spec_freq, args.spec_time],
            "batch": args.batch, "iters": args.iters, "device": args.device,
        },
        "models": {},
    }

    print("=" * 78)
    print("ARGUS PANOPTES - ONNX Runtime CPU latency benchmark")
    print("=" * 78)
    print(f"{'model':18s} {'p50 (ms)':>10s} {'p95 (ms)':>10s} {'batch p50':>12s} "
          f"{'throughput':>14s}")
    print("-" * 78)
    for name, kind in bench_items:
        if args.artifact:
            stem = args.artifact if args.artifact.endswith(".onnx") else f"{args.artifact}.onnx"
            onnx_path = args.models_dir / stem
        else:
            onnx_path = args.models_dir / f"dl_{name}.onnx"
        r = _bench_onnx_model(
            onnx_path, kind, args.chunk_len, (args.spec_freq, args.spec_time),
            args.thermal_dim, args.batch, args.iters,
        )
        if r is None:
            print(f"{name:18s}  (missing {onnx_path.name} - skipped)")
            continue
        results["models"][name] = r
        print(f"{name:18s} {r['single']['p50_ms']:10.3f} {r['single']['p95_ms']:10.3f} "
              f"{r['batched']['p50_ms']:12.3f} "
              f"{r['batched']['throughput_cps']:10.0f} c/s")

    print("-" * 78)
    xgb = _bench_xgboost(args.models_dir, args.iters)
    if xgb:
        results["xgboost"] = xgb
        print(f"{'xgboost':18s} predict p50={xgb['xgb_predict']['p50_ms']:.3f} ms  "
              f"+ DSP extract p50={xgb['dsp_extract']['p50_ms']:.3f} ms  "
              f"-> end-to-end {xgb['end_to_end_p50_ms']:.3f} ms")
    else:
        print(f"{'xgboost':18s} (no xgb_reg_wear_level.joblib - run models/baseline.py first)")
    print("=" * 78)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    print(f"Saved benchmark JSON -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
