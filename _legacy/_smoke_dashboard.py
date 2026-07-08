"""Headless smoke test for the dashboard's heavy paths (no Streamlit server)."""
from __future__ import annotations

import numpy as np

from dashviz import metrics, optimization as opt, plots
from dashviz.scenarios import SCENARIOS
from sensors import SawVibrationSimulator, ThermalSimulator
from models.streaming_perceptor import StreamingPerceptor, available_models
from dsp import SignalProcessor


def main() -> int:
    # --- scenarios ---
    for k, sc in SCENARIOS.items():
        ws = [sc.wear_for_step(i) for i in range(sc.n_chunks)]
        assert all(0.0 <= w <= 1.0 for w in ws), k
    print("scenarios OK:", list(SCENARIOS))

    # --- generate a chunk (mirror infra.generate_chunk) ---
    saw, therm = SawVibrationSimulator(), ThermalSimulator()
    params = {"alloy": "7075", "blade_speed_sfpm": 900.0, "feed_per_tooth_mm": 0.18,
              "depth_mm": 32.0, "num_teeth": 72}
    t, accel, meta = saw.generate(duration_s=0.5, params=params, wear=0.7, seed=3)
    _, temp, tmeta = therm.generate(duration_s=0.5, params=params, wear=0.7, seed=3)
    meta["mean_temp_c"] = tmeta["mean_temp_c"]
    meta["max_temp_c"] = tmeta["max_temp_c"]
    meta["temp_rise_c"] = tmeta["temp_rise_c"]
    fs = float(meta["fs_hz"])
    print(f"chunk: n={accel.size} fs={fs} tpf={meta['tooth_pass_freq_hz']:.1f} "
          f"temp={tmeta['mean_temp_c']:.1f}C")

    # --- inference (direct) ---
    perc = StreamingPerceptor(model="1dcnn_normnone", chunk_s=0.5).load()
    res = perc.infer_chunk(accel, metadata=meta, wear_injected=0.7, return_features=True)
    p = res["predictions"]
    assert 0 <= p["wear_level"] <= 1
    assert abs(sum(p["health_probs"].values()) - 1.0) < 1e-4
    print(f"infer OK: wear={p['wear_level']:.3f} health={p['health_state']} "
          f"conf={p['confidence']:.2f} lat={res['latency_ms']:.2f}ms "
          f"action={res['recommendations']['action']}")

    # --- STFT + all figures ---
    stft = SignalProcessor().compute_stft(accel, fs)
    figs = {
        "waveform": plots.build_waveform_figure(accel, t=t),
        "fft": plots.build_fft_figure(accel, fs, tpf_hz=meta["tooth_pass_freq_hz"]),
        "stft": plots.build_stft_heatmap(stft["power"], stft["freqs"], stft["times"]),
        "gauge": plots.build_gauge_figure(p["wear_level"], "wear"),
        "probs": plots.build_health_prob_bar(p["health_probs"]),
        "hist": plots.build_histogram(np.random.default_rng(0).random(50)),
        "pie": plots.build_pie(["healthy", "warning"], [30, 10], color_map=None),
        "bar": plots.build_grouped_bar(["a", "b"], {"s1": [1, 2], "s2": [3, 4]}),
    }
    for name, fig in figs.items():
        assert fig is not None and hasattr(fig, "to_dict")
        fig.to_dict()  # force full serialization
    print("figures OK:", list(figs))

    # --- empty-figure guards ---
    plots.build_waveform_figure(np.array([])).to_dict()
    plots.build_fft_figure(np.array([]), 40960.0).to_dict()
    plots.build_stft_heatmap(np.zeros((0, 0)), np.array([]), np.array([])).to_dict()
    print("empty-figure guards OK")

    # --- optimization ---
    state = opt.PerceptionState(wear_level=p["wear_level"], cycle_time_factor=p["cycle_time_factor"],
                                quality_score=p["quality_score"], health_state=p["health_state"],
                                confidence=p["confidence"], anomaly_flag=p["anomaly_flag"])
    impact = opt.compute_production_impact(state, opt.ProductionInputs())
    payload = opt.downstream_payload(state, opt.ProductionInputs(), impact, model="1dcnn_normnone")
    assert payload["schema"] == "argus.production_plan.v1"
    assert impact["current"]["cost_per_good_part"] > 0
    print(f"optimization OK: eff={impact['current']['efficiency_score']:.1f} "
          f"cost=${impact['current']['cost_per_good_part']:.2f} "
          f"maint={impact['maintenance']['level']}")

    # --- metrics loaders ---
    assert metrics.load_benchmark()
    for name in ("xgboost", "1dcnn_normnone", "fusion_normnone", "spectrogram"):
        s = metrics.model_metric_summary(name)
        print(f"metrics {name}: wear_mae={s['wear_mae']} f1={s['health_f1']} "
              f"lat={s['latency_ms']}")
    n_avail = sum(1 for m in available_models() if m["available"])
    print(f"available models: {n_avail}/{len(available_models())}")

    # --- xgboost path (features + context) ---
    px = StreamingPerceptor(model="xgboost", chunk_s=0.5).load()
    rx = px.infer_chunk(accel, metadata=meta, wear_injected=0.7)
    print(f"xgboost OK: wear={rx['predictions']['wear_level']:.3f} "
          f"kind={rx['model_kind']}")

    print("\nALL SMOKE TESTS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
