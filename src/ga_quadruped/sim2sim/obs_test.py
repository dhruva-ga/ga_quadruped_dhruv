#!/usr/bin/env python3
"""
Plot qpos and qvel for 12 joints from a CSV where each observation is 45 values long.

Defaults:
- period = 45 values per observation
- qpos indices = [9:21]  (12 values)
- qvel indices = [21:33] (12 values)

Handles CSVs that are:
1) One observation per row (45 columns),
2) A single row with N*45 values (auto-reshaped),
3) Any shape where total entries are a multiple of 45.

Usage:
    python plot_qpos_qvel.py data.csv
    # Optional:
    python plot_qpos_qvel.py data.csv --period 45 --offset 9 --nqpos 12 --nqvel 12 --save-prefix out
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_observations(csv_path: Path, period: int) -> np.ndarray:
    # Read raw CSV with no header; allow flexible whitespace/commas
    try:
        df = pd.read_csv(
            csv_path,
            header=None,
            sep=None,               # auto-detects delimiter (comma/whitespace)
            engine="python",
            skip_blank_lines=True
        )
    except Exception as e:
        print(f"Failed to read CSV: {e}", file=sys.stderr)
        raise

    arr = df.values
    # If we got a 1D-ish structure, make sure it's 2D
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)

    rows, cols = arr.shape

    # Case A: already 45 columns — good
    if cols == period:
        return arr.astype(float, copy=False)

    # Case B: single row with multiple-of-45 columns — split into rows
    if rows == 1 and cols % period == 0:
        n_obs = cols // period
        return arr.reshape(n_obs, period).astype(float, copy=False)

    # Case C: total elements multiple of 45 — flatten and reshape safely
    total = rows * cols
    if total % period == 0:
        return arr.reshape(total // period, period).astype(float, copy=False)

    raise ValueError(
        f"Data shape {arr.shape} is incompatible with period={period}. "
        f"Total elements {total} is not a multiple of {period}."
    )


def extract_blocks(data: np.ndarray, offset: int, nqpos: int, nqvel: int):
    # Slices in Python are end-exclusive
    qpos_slice = slice(offset, offset + nqpos)
    qvel_slice = slice(offset + nqpos, offset + nqpos + nqvel)

    if data.shape[1] < (offset + nqpos + nqvel):
        raise ValueError(
            f"Each observation has only {data.shape[1]} columns but need at least "
            f"{offset + nqpos + nqvel} for qpos+qvel slices."
        )

    qpos = data[:, qpos_slice]
    qvel = data[:, qvel_slice]
    return qpos, qvel


def plot_series(t, series, title, ylabel, labels=None, save_path=None, show=True):
    plt.figure()
    # series: shape (T, N)
    n_j = series.shape[1]
    plt.figure(figsize=(10, 6))  # Increased height (default is usually 6)
    for j in range(n_j):
        lab = labels[j] if labels and j < len(labels) else f"joint{j+1}"
        plt.plot(t, series[:, j], label=lab)
    plt.title(title)
    plt.xlabel("sample")
    plt.gca().set_ylim(auto=True)
    plt.gca().set_yticks(np.linspace(np.min(series), np.max(series), num=20))
    plt.ylabel(ylabel)
    plt.legend(loc="best")
    plt.grid(True)
    # plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved: {save_path}")
    if show:
        plt.show()


def main():
    p = argparse.ArgumentParser(description="Plot qpos and qvel for 12 joints from 45-wide observations.")
    p.add_argument("csv", type=Path, help="Path to CSV file")
    p.add_argument("--period", type=int, default=45, help="Values per observation (default: 45)")
    p.add_argument("--offset", type=int, default=9, help="Starting index of qpos (default: 9)")
    p.add_argument("--nqpos", type=int, default=12, help="Number of qpos joints (default: 12)")
    p.add_argument("--nqvel", type=int, default=12, help="Number of qvel joints (default: 12)")
    p.add_argument("--save-prefix", type=str, default=None, help="Prefix to save PNGs (e.g., 'run1'). If not set, figures are only shown.")
    p.add_argument("--no-show", action="store_true", help="Do not display interactive windows (useful on servers).")
    args = p.parse_args()

    data = load_observations(args.csv, period=args.period)
    qpos, qvel = extract_blocks(data, offset=args.offset, nqpos=args.nqpos, nqvel=args.nqvel)

    T = np.arange(data.shape[0])
    joint_labels = [f"joint{i+1}" for i in range(qpos.shape[1])]

    # Prepare save paths if requested
    if args.save_prefix:
        qpos_png = Path(f"{args.save_prefix}_qpos.png")
        qvel_png = Path(f"{args.save_prefix}_qvel.png")
    else:
        qpos_png = None
        qvel_png = None

    show = not args.no_show

    plot_series(T, qpos, title="qpos (12 joints)", ylabel="position", labels=joint_labels, save_path=qpos_png, show=show)
    plot_series(T, qvel, title="qvel (12 joints)", ylabel="velocity", labels=joint_labels, save_path=qvel_png, show=show)


if __name__ == "__main__":
    main()
