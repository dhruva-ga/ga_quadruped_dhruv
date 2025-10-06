#!/usr/bin/env python3
import os, glob
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt

def load_stack_dir(dirpath: Path):
    files = sorted(glob.glob(str(dirpath / "*.npy")))
    if not files:
        return None, None, None
    arrs = []
    for f in files:
        a = np.load(f)
        if a.ndim == 0:
            a = a.reshape(1)
        elif a.ndim > 1:
            a = a.reshape(-1)
        arrs.append(a.astype(float))
    if not arrs:
        return None, None, None

    max_d = max(a.shape[-1] for a in arrs)
    padded = []
    for a in arrs:
        if a.shape[-1] < max_d:
            pad = np.full((max_d - a.shape[-1],), np.nan)
            a = np.concatenate([a, pad], axis=0)
        padded.append(a)
    data = np.vstack(padded)

    xs = []
    for f in files:
        s = os.path.basename(f).rsplit(".", 1)[0]
        try:
            xs.append(float(s))
        except Exception:
            xs.append(np.nan)
    if not np.isnan(xs).any():
        x = np.array(xs) - float(xs[0])
        xlabel = "time (s from first file)"
    else:
        x = np.arange(data.shape[0], dtype=float)
        xlabel = "sample index"
    return data, x, xlabel

def main():
    artifacts_dir = Path("/home/radon12/Documents/ga_quadruped/logs/param/1758550983.2430685/artifacts/")
    plots_dir = artifacts_dir / "__plots"
    plots_dir.mkdir(exist_ok=True)

    subdirs = [p for p in sorted(artifacts_dir.iterdir()) if p.is_dir() and not p.name.startswith("__")]
    print(f"Found {len(subdirs)} artifact folders in {artifacts_dir}")

    # Load all data first
    loaded = []
    for sd in subdirs:
        print(subdirs)
        data, x, xlabel = load_stack_dir(sd)
        if data is not None:
            loaded.append((sd, data, x, xlabel))
        else:
            print(f"- {sd.name}: no .npy files")

    if not loaded:
        print("Nothing to plot.")
        return

    # global x-limits for shared axis
    x_min = min(np.nanmin(x) for _, _, x, _ in loaded)
    x_max = max(np.nanmax(x) for _, _, x, _ in loaded)

    fig, axes = plt.subplots(
        nrows=len(loaded), ncols=1,
        figsize=(12, 3 * len(loaded)),
        sharex=True  # <<< important
    )
    if len(loaded) == 1:
        axes = [axes]

    last_xlabel = "sample index"
    for ax, (sd, data, x, xlabel) in zip(axes, loaded):
        N, D = data.shape
        for d in range(D):
            ax.plot(x, data[:, d], label=f"ch{d}")
        ax.set_title(f"{sd.name} — {N} samples × {D} dims")
        ax.set_ylabel("value")
        try:
            ax.legend(loc="best", ncols=min(4, max(1, D)))
        except TypeError:
            ax.legend(loc="best")
        last_xlabel = xlabel
        ax.set_xlim(x_min, x_max)

    axes[-1].set_xlabel(last_xlabel)
    plt.tight_layout()

    plt.show()
    plt.close()

if __name__ == "__main__":
    main()
