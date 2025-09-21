#!/usr/bin/env python3
"""
plot_actuator_torques_lp.py

Load actuator_torques.csv, apply smoothing/low-pass, and SAVE one plot per leg.
Methods:
  - moving  : centered moving average (no deps)
  - savgol  : Savitzky–Golay (SciPy optional)
  - butter  : zero-phase Butterworth low-pass (SciPy)
  - ema     : first-order exponential low-pass (no deps)

Examples:
  python plot_actuator_torques_lp.py --method butter --lp-cutoff-hz 5 --lp-order 4
  python plot_actuator_torques_lp.py --method ema --lp-cutoff-hz 5
  python plot_actuator_torques_lp.py --method moving --window-sec 0.2
"""

import argparse
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from scipy.signal import savgol_filter, butter, filtfilt, lfilter
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False

LEGS = {
    "fl": ["hip_abduction_fl", "thigh_rotation_fl", "calf_fl"],
    "fr": ["hip_abduction_fr", "thigh_rotation_fr", "calf_fr"],
    "rl": ["hip_abduction_rl", "thigh_rotation_rl", "calf_rl"],
    "rr": ["hip_abduction_rr", "thigh_rotation_rr", "calf_rr"],
}
LEG_TITLES = {"fl":"Front-Left (FL)","fr":"Front-Right (FR)","rl":"Rear-Left (RL)","rr":"Rear-Right (RR)"}
LEGEND_NICE = {c:lab for leg in LEGS.values() for c,lab in zip(leg,["hip","thigh","calf"])}

def ensure_columns(df: pd.DataFrame, cols):
    miss = [c for c in cols if c not in df.columns]
    if miss:
        raise ValueError(f"CSV missing columns: {miss}")

def infer_fs(df: pd.DataFrame, sim_col="sim_time", sample_rate=None):
    """Return sampling rate (Hz). Prefer --sample-rate, else infer from sim_time."""
    if sample_rate is not None:
        return float(sample_rate)
    if sim_col in df.columns:
        t = df[sim_col].to_numpy()
        if len(t) >= 2:
            dt = np.nanmedian(np.diff(t))
            if dt > 0:
                return 1.0 / dt
    raise ValueError("Could not infer sample rate. Provide --sample-rate or include sim_time column.")

def infer_window_samples(df, xcol, window_samples, window_sec):
    if window_samples is not None:
        return max(3, int(window_samples))
    if window_sec is not None:
        if xcol != "sim_time":
            raise ValueError("--window-sec requires --x sim_time")
        fs = infer_fs(df)
        return max(3, int(round(window_sec * fs)))
    return 21

def make_odd(w): return w if w % 2 else w + 1

def smooth_series(series: pd.Series, method: str, *, win_samples=21,
                  polyorder=3, fs=None, fc=None, butter_order=4):
    x = series.to_numpy()
    n = len(x)
    if n == 0:
        return series

    if method == "moving":
        w = max(1, min(win_samples, n))
        return series.rolling(window=w, center=True, min_periods=1).mean()

    if method == "savgol":
        if not SCIPY_AVAILABLE:
            warnings.warn("SciPy not available; falling back to moving average.")
            return smooth_series(series, "moving", win_samples=win_samples)
        w = make_odd(max(polyorder + 2, 3, min(win_samples, n if n % 2 else n - 1)))
        if w <= polyorder or w < 3:
            return smooth_series(series, "moving", win_samples=win_samples)
        y = savgol_filter(x, window_length=w, polyorder=polyorder, mode="interp")
        return pd.Series(y, index=series.index)

    if method == "butter":
        if not SCIPY_AVAILABLE:
            warnings.warn("SciPy not available; using EMA low-pass instead.")
            return smooth_series(series, "ema", fs=fs, fc=fc)
        if fs is None or fc is None:
            raise ValueError("butter requires fs and fc")
        nyq = 0.5 * fs
        wn = np.clip(fc / nyq, 1e-6, 0.999999)  # normalized cutoff
        b, a = butter(butter_order, wn, btype="low", analog=False)
        # filtfilt (zero phase); if too short, fall back to lfilter
        pad_needed = 3 * max(len(a), len(b))
        if n > pad_needed:
            y = filtfilt(b, a, x, method="gust")
        else:
            y = lfilter(b, a, x)
        return pd.Series(y, index=series.index)

    if method == "ema":
        # first-order RC low-pass: alpha = dt/(tau+dt) with tau=1/(2*pi*fc)
        if fs is None or fc is None:
            raise ValueError("ema requires fs and fc")
        dt = 1.0 / fs
        tau = 1.0 / (2.0 * np.pi * fc)
        alpha = dt / (tau + dt)
        alpha = float(np.clip(alpha, 1e-6, 1.0))
        return series.ewm(alpha=alpha, adjust=False).mean()

    raise ValueError(f"Unknown method: {method}")

def plot_leg(df_sm: pd.DataFrame, xcol: str, leg_key: str, out_dir: Path, fmt="png"):
    cols = LEGS[leg_key]
    fig = plt.figure()
    for c in cols:
        plt.plot(df_sm[xcol], df_sm[c], label=LEGEND_NICE.get(c, c))
    plt.title(f"{LEG_TITLES[leg_key]} actuator efforts (low-pass/smoothed)")
    plt.xlabel("time (s)" if xcol == "sim_time" else xcol)
    plt.ylabel("actuator effort (N·m or N)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    out_path = out_dir / f"torques_{leg_key}.{fmt}"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved {out_path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, default=Path("actuator_torques.csv"))
    ap.add_argument("--x", dest="xcol", choices=["sim_time", "step"], default="sim_time")
    ap.add_argument("--out", type=Path, default=Path("torque_plots_lp"))
    ap.add_argument("--format", dest="fmt", choices=["png", "pdf", "svg"], default="png")

    ap.add_argument("--method", choices=["moving", "savgol", "butter", "ema"], default="butter")

    # windowed smoothers
    ap.add_argument("--window-samples", type=int, default=None)
    ap.add_argument("--window-sec", type=float, default=None)
    ap.add_argument("--polyorder", type=int, default=3)

    # low-pass params
    ap.add_argument("--lp-cutoff-hz", type=float, default=5.0, help="Low-pass cutoff (Hz)")
    ap.add_argument("--lp-order", type=int, default=4, help="Butterworth order")
    ap.add_argument("--sample-rate", type=float, default=None,
                    help="Override inferred sample rate (Hz)")

    args = ap.parse_args()

    if not args.csv.exists():
        raise FileNotFoundError(f"CSV not found: {args.csv}")

    df = pd.read_csv(args.csv)
    ensure_columns(df, ["step"])  # sim_time optional if you pass --sample-rate
    all_cols = sum(LEGS.values(), [])
    ensure_columns(df, all_cols)

    # figure out parameters for chosen method
    if args.method in ("moving", "savgol"):
        win_samples = infer_window_samples(df, args.xcol, args.window_samples, args.window_sec)
        fs = fc = None
    else:  # butter / ema
        fs = infer_fs(df, sample_rate=args.sample_rate)
        fc = float(args.lp_cutoff_hz)
        win_samples = None

    # smooth a copy
    df_sm = df.copy()
    for col in all_cols:
        df_sm[col] = smooth_series(
            df[col], args.method,
            win_samples=win_samples or 21,
            polyorder=args.polyorder,
            fs=fs, fc=fc, butter_order=args.lp_order
        )

    args.out.mkdir(parents=True, exist_ok=True)
    for leg in ["fl", "fr", "rl", "rr"]:
        plot_leg(df_sm, args.xcol, leg, args.out, fmt=args.fmt)

if __name__ == "__main__":
    main()
