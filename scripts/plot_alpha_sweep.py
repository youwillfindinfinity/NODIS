"""
plot_alpha_sweep.py
-------------------
Plots MCC vs prior weight (alpha) for PIGLasso with a perfect oracle prior,
across all four network topologies. SSGLasso (no prior) shown as baseline.

Reads from results/alpha_sweep/ produced by benchmarks/run_alpha_sweep.py.

Usage:
    cd NODIS/
    python scripts/plot_alpha_sweep.py
    python scripts/plot_alpha_sweep.py --out figures/alpha_sweep.pdf
"""

import argparse
import os
import sys
import warnings

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(__file__))
import plot_style
plot_style.apply()

ALPHA_VALS  = [round(a / 10, 1) for a in range(1, 10)]
TOPOLOGIES  = ["scale-free", "hub", "cluster", "random"]

TOPO_PALETTE = {
    "cluster":    "#4C72B0",
    "hub":        "#F78154",
    "random":     "#4D9078",
    "scale-free": "#B4436C",
}
TOPO_MARKERS = {
    "cluster":    "o",
    "hub":        "s",
    "random":     "^",
    "scale-free": "D",
}
TOPO_LABELS = {
    "cluster":    "Cluster",
    "hub":        "Hub",
    "random":     "Random",
    "scale-free": "Scale-free",
}

SWEEP_DIR    = os.path.join(os.path.dirname(__file__), "..", "results", "alpha_sweep")
FIGURES_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")


def alpha_tag(alpha: float) -> str:
    return f"a{round(alpha * 10):02d}"


def load_sweep() -> pd.DataFrame:
    """Load all CSVs from alpha_sweep subdirectories into one DataFrame."""
    frames = []

    for alpha in ALPHA_VALS:
        subdir = os.path.join(SWEEP_DIR, alpha_tag(alpha))
        if not os.path.isdir(subdir):
            continue
        for fname in os.listdir(subdir):
            if not fname.endswith(".csv"):
                continue
            try:
                df = pd.read_csv(os.path.join(subdir, fname))
                df["alpha"] = alpha
                frames.append(df)
            except Exception:
                pass

    if not frames:
        raise FileNotFoundError(
            f"No results found in {SWEEP_DIR}.\n"
            "Run benchmarks/run_alpha_sweep.py first."
        )
    return pd.concat(frames, ignore_index=True)


def load_baseline() -> pd.DataFrame:
    """Load SSGLasso baseline from alpha_sweep/baseline/."""
    subdir = os.path.join(SWEEP_DIR, "baseline")
    if not os.path.isdir(subdir):
        return pd.DataFrame()
    frames = []
    for fname in os.listdir(subdir):
        if fname.endswith(".csv"):
            try:
                frames.append(pd.read_csv(os.path.join(subdir, fname)))
            except Exception:
                pass
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def build_figure(sweep: pd.DataFrame, baseline: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.subplots_adjust(left=0.10, right=0.93, top=0.82, bottom=0.13)

    end_annotations = []  # (mean_val, label_text, color) — staggered after loop

    for topo in TOPOLOGIES:
        color  = TOPO_PALETTE[topo]
        marker = TOPO_MARKERS[topo]
        label  = TOPO_LABELS[topo]

        means, sds = [], []
        for alpha in ALPHA_VALS:
            vals = sweep.loc[
                (sweep["topology"] == topo) & (sweep["alpha"] == alpha), "mcc"
            ].dropna()
            means.append(vals.mean() if len(vals) else np.nan)
            sds.append(vals.std()   if len(vals) else np.nan)

        means = np.array(means)
        sds   = np.array(sds)
        mask  = ~np.isnan(means)
        xs    = np.array(ALPHA_VALS)

        ax.plot(xs[mask], means[mask],
                color=color, marker=marker, markersize=5,
                linewidth=1.8, label=label, zorder=3)
        ax.fill_between(xs[mask],
                        (means - sds)[mask], (means + sds)[mask],
                        color=color, alpha=0.12, zorder=2)
        ax.errorbar(xs[mask], means[mask], yerr=sds[mask],
                    fmt="none", ecolor=color, elinewidth=0.9,
                    capsize=4, capthick=0.9, zorder=4)

        if mask[-1]:
            end_annotations.append((means[mask][-1], f"{means[mask][-1]:.2f}", color))

    # SSGLasso baselines per topology
    if not baseline.empty:
        for topo in TOPOLOGIES:
            vals = baseline.loc[baseline["topology"] == topo, "mcc"].dropna()
            if len(vals):
                ax.axhline(vals.mean(),
                           color=TOPO_PALETTE[topo], linewidth=1.2,
                           linestyle="--", alpha=0.55, zorder=1)

        ax.plot([], [], color="#888888", linewidth=1.2, linestyle="--",
                label="SSGLasso baseline (no prior)")

    ax.axvline(0.5, color="#888888", linewidth=1.0, linestyle=":",
               alpha=0.7, zorder=1)
    ax.text(0.5 + 0.01, ax.get_ylim()[0] + 0.01, "α = 0.5\n(default)",
            fontsize=8, color="#888888", va="bottom")

    ax.set_xlabel("Prior weight (α)")
    ax.set_ylabel("MCC")
    ax.set_xticks(ALPHA_VALS)
    ax.set_xlim(0.05, 0.97)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.2f}"))

    # Stagger final-point annotations — enforce minimum gap then re-centre
    MIN_SEP = 0.030
    end_annotations.sort(key=lambda t: t[0])
    positions = [a[0] for a in end_annotations]
    for i in range(1, len(positions)):
        if positions[i] - positions[i - 1] < MIN_SEP:
            positions[i] = positions[i - 1] + MIN_SEP
    # Re-centre so the spread stays near the actual data values
    mid_orig = np.median([a[0] for a in end_annotations])
    mid_placed = np.median(positions)
    positions = [p + (mid_orig - mid_placed) for p in positions]

    for (y_val, text, color), y_pos in zip(end_annotations, positions):
        ax.text(0.915, y_pos, text,
                ha="left", va="center", fontsize=9,
                fontweight="bold", color=color,
                clip_on=False)

    ax.legend(frameon=False, loc="upper center",
              bbox_to_anchor=(0.5, 1.18), ncol=5,
              borderaxespad=0, columnspacing=1.2, handlelength=1.5)

    return fig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=os.path.join(FIGURES_DIR, "alpha_sweep.pdf"))
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    print("Loading alpha sweep results …")
    sweep = load_sweep()
    print(f"  {len(sweep)} rows loaded across "
          f"{sweep['alpha'].nunique()} alpha values, "
          f"{sweep['topology'].nunique()} topologies")

    baseline = load_baseline()
    if baseline.empty:
        print("  [WARN] No SSGLasso baseline found — run without --skip-baseline")
    else:
        print(f"  {len(baseline)} baseline rows loaded")

    print("Building figure …")
    fig = build_figure(sweep, baseline)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight")
    print(f"Saved → {args.out}")

    if args.out.endswith(".pdf"):
        png_out = args.out.replace(".pdf", ".png")
        fig.savefig(png_out, dpi=150, bbox_inches="tight")
        print(f"Saved → {png_out}")

    plt.close(fig)


if __name__ == "__main__":
    main()
