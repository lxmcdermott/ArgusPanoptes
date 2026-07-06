"""Validate the Argus Panoptes physics simulators.

Runs several representative operating points (nominal aluminum sawing, high wear,
edge cases), checks physical sanity, and renders side-by-side diagnostic plots.

Sanity checks
-------------
* TPF detected in the Welch PSD within 1% of the analytic value.
* Vibration signal energy (RMS) increases monotonically with wear.
* Cut-zone temperature increases monotonically with wear.
* No NaNs/Infs; values within physical ranges; no hard clipping at nominal.

Usage
-----
    python scripts/validate_simulators.py                 # save plots, print report
    python scripts/validate_simulators.py --no-show       # don't open windows
    python scripts/validate_simulators.py --outdir experiments/plots
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy import signal as sp_signal

# Allow running as a plain script from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sensors import SawVibrationSimulator, ThermalSimulator, load_config  # noqa: E402
from sensors.utils import FloatArray  # noqa: E402

GREEN = "\033[92m"
RED = "\033[91m"
RESET = "\033[0m"


def _ok(passed: bool) -> str:
    return f"{GREEN}PASS{RESET}" if passed else f"{RED}FAIL{RESET}"


def dominant_tpf_error_pct(
    accel: FloatArray, fs: float, tpf_expected: float
) -> tuple[float, float]:
    """Return (detected_peak_hz, pct_error) for the TPF region of the PSD."""
    freqs, psd = sp_signal.welch(accel, fs=fs, nperseg=min(len(accel), 8192))
    # Search a window around the expected TPF for the dominant line.
    lo, hi = tpf_expected * 0.5, tpf_expected * 1.5
    mask = (freqs >= lo) & (freqs <= hi)
    if not np.any(mask):
        return float("nan"), float("inf")
    band_freqs, band_psd = freqs[mask], psd[mask]
    peak_hz = float(band_freqs[int(np.argmax(band_psd))])
    return peak_hz, abs(peak_hz - tpf_expected) / tpf_expected * 100.0


def run_checks() -> tuple[bool, list[dict]]:
    """Execute the numerical sanity checks; return (all_passed, records)."""
    cfg = load_config()
    vib = SawVibrationSimulator(cfg)
    therm = ThermalSimulator(cfg)
    fs = cfg.vibration.fs_hz

    print("=" * 74)
    print("ARGUS PANOPTES — SIMULATOR VALIDATION")
    print("=" * 74)

    all_passed = True

    # --- Check 1: TPF detection within 1% across configs ---------------------
    print("\n[1] Tooth-pass frequency detection (target < 1% error)")
    configs = [
        ("nominal 6061", {"alloy": "6061"}, 0.1),
        ("high-speed 7075", {"alloy": "7075", "blade_speed_sfpm": 1100}, 0.5),
        ("heavy cut", {"depth_mm": 45, "feed_per_tooth_mm": 0.3}, 0.7),
    ]
    for name, params, wear in configs:
        _, accel, meta = vib.generate(duration_s=4.0, params=params, wear=wear, seed=3)
        peak_hz, err = dominant_tpf_error_pct(accel, fs, meta["tooth_pass_freq_hz"])
        passed = err < 1.0
        all_passed &= passed
        print(
            f"    {name:18s} TPF={meta['tooth_pass_freq_hz']:7.2f} Hz  "
            f"detected={peak_hz:7.2f} Hz  err={err:5.2f}%  {_ok(passed)}"
        )

    # --- Check 2: vibration RMS monotonic in wear ----------------------------
    print("\n[2] Vibration RMS increases with wear")
    wears = [0.0, 0.25, 0.5, 0.75, 1.0]
    rms_vals = []
    for w in wears:
        _, _, meta = vib.generate(duration_s=3.0, wear=w, seed=11)
        rms_vals.append(meta["signal_rms_g"])
    mono = all(b > a for a, b in zip(rms_vals, rms_vals[1:]))
    all_passed &= mono
    print("    wear:  " + "  ".join(f"{w:.2f}" for w in wears))
    print("    rms_g: " + "  ".join(f"{r:.3f}" for r in rms_vals) + f"   {_ok(mono)}")

    # --- Check 3: temperature monotonic in wear ------------------------------
    print("\n[3] Cut-zone temperature increases with wear")
    temp_vals = []
    for w in wears:
        _, _, meta = therm.generate(duration_s=6.0, wear=w, seed=11)
        temp_vals.append(meta["steady_state_temp_c"])
    mono_t = all(b > a for a, b in zip(temp_vals, temp_vals[1:]))
    in_range = all(20.0 < v < 500.0 for v in temp_vals)
    all_passed &= mono_t and in_range
    print("    wear:   " + "  ".join(f"{w:.2f}" for w in wears))
    print(
        "    T_ss_C: " + "  ".join(f"{v:5.1f}" for v in temp_vals)
        + f"   mono={_ok(mono_t)} range={_ok(in_range)}"
    )

    # --- Check 4: finiteness, dtype, no clipping at nominal ------------------
    print("\n[4] Finiteness / dtype / clipping")
    t, accel, vmeta = vib.generate(duration_s=3.0, wear=0.5, seed=1)
    _, temp, tmeta = therm.generate(duration_s=3.0, wear=0.5, seed=1)
    finite = bool(np.all(np.isfinite(accel)) and np.all(np.isfinite(temp)))
    dtype_ok = accel.dtype == np.float64 and temp.dtype == np.float64
    no_clip = not vmeta["clipped"]
    all_passed &= finite and dtype_ok and no_clip
    print(f"    finite={_ok(finite)}  float64={_ok(dtype_ok)}  no_clip={_ok(no_clip)}")

    # --- Check 5: reproducibility -------------------------------------------
    print("\n[5] Reproducibility (identical seed -> identical signal)")
    _, a1, _ = vib.generate(duration_s=1.0, wear=0.3, seed=99)
    _, a2, _ = vib.generate(duration_s=1.0, wear=0.3, seed=99)
    repro = bool(np.array_equal(a1, a2))
    all_passed &= repro
    print(f"    identical arrays: {_ok(repro)}")

    print("\n" + "=" * 74)
    print(f"OVERALL: {_ok(all_passed)}")
    print("=" * 74)

    records = [
        {"wears": wears, "rms": rms_vals, "temp": temp_vals},
    ]
    return all_passed, records


def make_plots(outdir: Path, show: bool) -> Path:
    """Render side-by-side diagnostic plots for new vs worn blades."""
    import matplotlib

    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cfg = load_config()
    vib = SawVibrationSimulator(cfg)
    therm = ThermalSimulator(cfg)
    fs = cfg.vibration.fs_hz

    t_new, a_new, m_new = vib.generate(duration_s=2.0, wear=0.05, seed=7)
    t_worn, a_worn, m_worn = vib.generate(duration_s=2.0, wear=0.95, seed=7)
    tpf = m_new["tooth_pass_freq_hz"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle(
        "Argus Panoptes — Vibration & Thermal Validation (sharp vs worn blade)",
        fontsize=13,
        fontweight="bold",
    )

    # (a) Waveform (first ~40 ms)
    win = int(0.04 * fs)
    ax = axes[0, 0]
    ax.plot(t_new[:win] * 1e3, a_new[:win], lw=0.8, label="wear=0.05", color="tab:blue")
    ax.plot(t_worn[:win] * 1e3, a_worn[:win], lw=0.8, label="wear=0.95", color="tab:red", alpha=0.8)
    ax.set(xlabel="time (ms)", ylabel="accel (g)", title="(a) Vibration waveform")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)

    # (b) Welch PSD with TPF harmonic markers
    ax = axes[0, 1]
    for accel, lab, col in ((a_new, "wear=0.05", "tab:blue"), (a_worn, "wear=0.95", "tab:red")):
        f, psd = sp_signal.welch(accel, fs=fs, nperseg=8192)
        ax.semilogy(f, psd, lw=0.8, label=lab, color=col, alpha=0.85)
    for k in range(1, 6):
        if k * tpf < fs / 2:
            ax.axvline(k * tpf, color="green", ls="--", lw=0.7, alpha=0.5)
    ax.set(
        xlabel="frequency (Hz)",
        ylabel="PSD (g²/Hz)",
        title=f"(b) Welch PSD — TPF={tpf:.1f} Hz (dashed = harmonics)",
        xlim=(0, min(6000, fs / 2)),
    )
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3, which="both")

    # (c) Thermal transient across wear levels
    ax = axes[1, 0]
    for w, col in ((0.0, "tab:blue"), (0.5, "tab:orange"), (1.0, "tab:red")):
        tt, temp, _ = therm.generate(duration_s=8.0, wear=w, seed=7)
        ax.plot(tt, temp, lw=1.2, label=f"wear={w:.1f}", color=col)
    ax.set(xlabel="time (s)", ylabel="cut-zone temp (°C)", title="(c) Thermal transient")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3)

    # (d) RMS & steady-state temp vs wear
    ax = axes[1, 1]
    wears = np.linspace(0, 1, 11)
    rms_vals, temp_vals = [], []
    for w in wears:
        _, _, vm = vib.generate(duration_s=2.0, wear=float(w), seed=5)
        _, _, tm = therm.generate(duration_s=6.0, wear=float(w), seed=5)
        rms_vals.append(vm["signal_rms_g"])
        temp_vals.append(tm["steady_state_temp_c"])
    ax.plot(wears, rms_vals, "o-", color="tab:blue", label="vibration RMS (g)")
    ax.set_xlabel("wear level")
    ax.set_ylabel("vibration RMS (g)", color="tab:blue")
    ax.tick_params(axis="y", labelcolor="tab:blue")
    ax2 = ax.twinx()
    ax2.plot(wears, temp_vals, "s-", color="tab:red", label="steady-state T (°C)")
    ax2.set_ylabel("steady-state temp (°C)", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")
    ax.set_title("(d) Wear sensitivity")
    ax.grid(alpha=0.3)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    outdir.mkdir(parents=True, exist_ok=True)
    out_path = outdir / "simulator_validation.png"
    fig.savefig(out_path, dpi=130)
    print(f"\nSaved diagnostic plot -> {out_path}")
    if show:
        plt.show()
    plt.close(fig)
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate Argus Panoptes simulators.")
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "experiments" / "plots",
        help="Directory to save diagnostic plots.",
    )
    parser.add_argument("--no-show", action="store_true", help="Do not open plot windows.")
    parser.add_argument("--no-plots", action="store_true", help="Skip plot generation.")
    args = parser.parse_args()

    all_passed, _ = run_checks()
    if not args.no_plots:
        make_plots(args.outdir, show=not args.no_show)

    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
