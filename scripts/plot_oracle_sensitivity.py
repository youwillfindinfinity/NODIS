"""
plot_oracle_sensitivity.py
--------------------------
Visualises the noisy-oracle prior sensitivity sweep for PIGLasso.
Produces two separate single-panel figures: one for AUPR, one for MCC.

Noise levels:
  n00 = 0.0  (perfect oracle)
  n01 = 0.1  (10% edges flipped)
  n02 = 0.2  (20% edges flipped)
  n03 = 0.3  (30% edges flipped)
  SSGLasso = no prior baseline (dashed horizontal)

Usage:
    cd NODIS/
    python scripts/plot_oracle_sensitivity.py
    python scripts/plot_oracle_sensitivity.py --out-aupr figures/oracle_aupr.pdf --out-mcc figures/oracle_mcc.pdf
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(__file__))
import plot_style
plot_style.apply()

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
SUMMARY_CSV = os.path.join(RESULTS_DIR, "metrics_summary.csv")
FIGURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")

TOPOS = ["cluster", "hub", "random", "scale-free"]
TOPO_PALETTE = {
    "cluster":    "#4C72B0",
    "hub":        "#F78154",
    "random":     "#5CAD6E",
    "scale-free": "#B4436C",
}
TOPO_MARKERS = {
    "cluster":    "o",
    "hub":        "s",
    "random":     "^",
    "scale-free": "D",
}

# Oracle noise levels in order (x-axis) — 0% to 100% in steps of 10%
ORACLE_METHODS = [f"piglasso_oracle_n{i:02d}" for i in range(11)]
NOISE_LABELS   = ["0%\n(perfect)", "10%", "20%", "30%", "40%",
                  "50%\n(random)", "60%", "70%", "80%", "90%", "100%"]
NOISE_X        = np.arange(len(ORACLE_METHODS))

SSGLASSO_COLOR = "#7B2D8B"   # purple, matches main benchmark plot


# ---------------------------------------------------------------------------
# Panel helper
# ---------------------------------------------------------------------------

def _oracle_panel(ax, data, metric, ylabel, title):
    """
    Line plot: metric mean (±SEM) vs. noise level, one line per topology.
    Dashed horizontal line = SSGLasso (no prior) grand mean per topology.
    """
    syn = data[
        (data["benchmark"] == "synthetic") &
        (data["config"] == "n513p164")
    ]

    for topo in TOPOS:
        color  = TOPO_PALETTE[topo]
        marker = TOPO_MARKERS[topo]

        # Oracle sweep line
        mus, sems = [], []
        for m in ORACLE_METHODS:
            vals = syn.loc[
                (syn["method"] == m) & (syn["topology"] == topo), metric
            ].dropna()
            mus.append(vals.mean()  if len(vals) else np.nan)
            sems.append(vals.sem()  if len(vals) else 0.0)

        mu  = np.array(mus)
        sem = np.array(sems)

        ax.plot(NOISE_X, mu, color=color, lw=2.0, marker=marker,
                ms=6, zorder=4, label=topo)
        ax.fill_between(NOISE_X, mu - sem, mu + sem,
                        color=color, alpha=0.12, zorder=3)

        # SSGLasso baseline (dashed horizontal)
        base = syn.loc[
            (syn["method"] == "ssglasso") & (syn["topology"] == topo), metric
        ].dropna().mean()
        if not np.isnan(base):
            ax.axhline(base, color=color, lw=1.0, linestyle="--",
                       alpha=0.6, zorder=2)

    ax.set_xticks(NOISE_X)
    ax.set_xticklabels(NOISE_LABELS)
    ax.set_xlabel("Prior noise level (fraction of edges flipped)")
    ax.set_ylabel(ylabel)
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda v, _: f"{v:.2f}")
    )


# ---------------------------------------------------------------------------
# Build single-panel figure
# ---------------------------------------------------------------------------

def build_single_figure(df: pd.DataFrame, metric: str, ylabel: str,
                        panel_title: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(11, 5))
    fig.subplots_adjust(left=0.12, right=0.97, top=0.80, bottom=0.18)

    _oracle_panel(ax, df, metric, ylabel, title=panel_title)

    # Legend
    topo_handles = [
        Line2D([0], [0], color=TOPO_PALETTE[t], lw=2.0,
               marker=TOPO_MARKERS[t], ms=6,
               label=t)
        for t in TOPOS
    ]
    baseline_handle = Line2D([0], [0], color="grey", lw=1.0,
                              linestyle="--", label="SSGLasso baseline (no prior)")

    fig.legend(handles=topo_handles + [baseline_handle],
               loc="upper center", ncol=5, frameon=False, fontsize=13,
               bbox_to_anchor=(0.5, 1.00),
               handlelength=1.8, columnspacing=2.0)


    return fig


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _save(fig, path, dpi):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    print(f"Saved → {path}")
    if path.endswith(".pdf"):
        png = path.replace(".pdf", ".png")
        fig.savefig(png, dpi=150, bbox_inches="tight")
        print(f"Saved → {png}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-aupr", default=os.path.join(FIGURES_DIR, "oracle_sensitivity_aupr.pdf"))
    parser.add_argument("--out-mcc",  default=os.path.join(FIGURES_DIR, "oracle_sensitivity_mcc.pdf"))
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    print("Loading data …")
    df = pd.read_csv(SUMMARY_CSV)
    oracle_methods = [m for m in df["method"].unique() if "oracle" in m]
    print(f"  {len(df):,} rows — oracle methods: {sorted(oracle_methods)}")

    if not oracle_methods:
        print("ERROR: no piglasso_oracle_* rows found in metrics_summary.csv")
        return

    print("Building AUPR figure …")
    fig_aupr = build_single_figure(df, "aupr", "AUPR",
                                   "AUPR vs. prior noise level (n=513, p=164)")
    _save(fig_aupr, args.out_aupr, args.dpi)

    print("Building MCC figure …")
    fig_mcc = build_single_figure(df, "mcc", "MCC",
                                  "MCC vs. prior noise level (n=513, p=164)")
    _save(fig_mcc, args.out_mcc, args.dpi)


if __name__ == "__main__":
    main()
