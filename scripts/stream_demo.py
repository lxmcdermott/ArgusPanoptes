"""Live streaming-inference demo for Argus Panoptes (Day 4 integration layer).

Wires the whole online path end-to-end without any HTTP: start a simulator,
stream vibration chunks through :class:`models.streaming_perceptor.StreamingPerceptor`,
run DSP + the selected model, persist every prediction to partitioned Parquet via
:class:`app.logging.InferenceLogger`, and print the structured payloads.

Usage
-----
    python scripts/stream_demo.py                       # 1D-CNN (normnone), 5 s
    python scripts/stream_demo.py --model xgboost --wear 0.9
    python scripts/stream_demo.py --model fusion_normnone --with-thermal --wear 0.7
    python scripts/stream_demo.py --duration-s 3 --chunk-s 0.5 --no-log
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sensors import SawVibrationSimulator, ThermalSimulator  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Argus Panoptes live streaming demo.")
    parser.add_argument("--model", type=str, default="1dcnn_normnone", help="Friendly model name.")
    parser.add_argument("--duration-s", type=float, default=5.0)
    parser.add_argument("--chunk-s", type=float, default=1.0)
    parser.add_argument("--wear", type=float, default=0.6, help="Ground-truth wear [0,1].")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--with-thermal", action="store_true", help="Add thermal context features.")
    parser.add_argument("--log-dir", type=str, default="logs/inference")
    parser.add_argument("--no-log", action="store_true", help="Disable Parquet logging.")
    parser.add_argument("--show-features", action="store_true", help="Include DSP features in output.")
    args = parser.parse_args()

    # Lazy import so `--help` stays fast and torch is never required.
    from models.streaming_perceptor import StreamingPerceptor

    logger_cfg = None if args.no_log else {"log_dir": args.log_dir, "flush_every": 4}
    perc = StreamingPerceptor(
        model=args.model,
        chunk_s=args.chunk_s,
        logger=logger_cfg,
    ).load()

    sim = SawVibrationSimulator()
    therm = ThermalSimulator() if args.with_thermal else None

    print(f"Streaming model={perc.model_name} kind={perc.spec.kind} "
          f"chunk_s={args.chunk_s} wear={args.wear}")
    print("=" * 72)

    n = 0
    with perc:
        for result in perc.stream_from_simulator(
            sim,
            duration_s=args.duration_s,
            chunk_s=args.chunk_s,
            wear=args.wear,
            seed=args.seed,
            thermal_simulator=therm,
            return_features=args.show_features,
        ):
            n += 1
            p = result["predictions"]
            print(
                f"chunk {result['chunk_id']:>3} | "
                f"health={p['health_state']:<8} conf={p['confidence']:.2f} | "
                f"wear={p['wear_level']:.3f} cycle={p['cycle_time_factor']:.3f} "
                f"quality={p['quality_score']:.3f} | "
                f"anomaly={p['anomaly_flag']} | {result['latency_ms']:.1f} ms | "
                f"-> {result['recommendations']['action']}"
            )
            if args.show_features and result["features"]:
                sample = {k: result["features"][k] for k in list(result["features"])[:4]}
                print(f"        features[0:4]={json.dumps(sample)}")

    print("=" * 72)
    print(f"Processed {n} chunks.")
    if not args.no_log:
        print(f"Parquet logs written under: {Path(args.log_dir).resolve()}")
        print("Read them back with:  python -c \"from app.logging import read_logs; "
              f"print(read_logs('{args.log_dir}').shape)\"")
    print("StreamingPerceptor smoke test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
