"""
plot_oracle_sensitivity.py  —  Nature-grade oracle prior sensitivity sweep.

Two single-panel figures: AUPR and MCC vs. noise level (n=513, p=164).

Usage:
    cd NODIS/
    python scripts/plot_oracle_sensitivity.py
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

from plot_style import FULL_W, TOPO_PALETTE, TOPO_MARKERS, TOPO_LABELS, panel_label, save

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
SUMMARY_CSV = os.path.join(RESULTS_DIR, "metrics_summary.csv")
FIGURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")

TOPOS          = ["cluster", "hub", "random", "scale-free"]
ORACLE_METHODS = [f"piglasso_oracle_n{i:02d}" for i in range(11)]
NOISE_LABELS   = ["0%\n(perfect)", "10%", "20%", "30%", "40%",
                  "50%\n(random)", "60%", "70%", "80%", "90%", "100%"]
NOISE_X        = np.arange(len(ORACLE_METHODS))
SSGLASSO_COLOR = "#0072B2"


def _oracle_panel(ax, data, metric, ylabel):
    syn = data[(data["benchmark"] == "synthetic") & (data["config"] == "n513p164")]

    for topo in TOPOS:
        color  = TOPO_PALETTE[topo]
        marker = TOPO_MARKERS[topo]
        mus, sems = [], []
        for m in ORACLE_METHODS:
            vals = syn.loc[(syn["method"] == m) & (syn["topology"] == topo), metric].dropna()
            mus.append(vals.mean()  if len(vals) else np.nan)
            sems.append(vals.sem() if len(vals) else 0.0)
        mu, sem = np.array(mus), np.array(sems)
        ax.plot(NOISE_X, mu, color=color, lw=1.5, marker=marker,
                ms=4, zorder=4, label=TOPO_LABELS[topo])
        ax.fill_between(NOISE_X, mu - sem, mu + sem, color=color, alpha=0.12, zorder=3)

        base = syn.loc[(syn["method"] == "ssglasso") & (syn["topology"] == topo),
                       metric].dropna().mean()
        if not np.isnan(base):
            ax.axhline(base, color=color, lw=0.7, linestyle="--", alpha=0.55, zorder=2)

    ax.set_xticks(NOISE_X)
    ax.set_xticklabels(NOISE_LABELS)
    ax.set_xlabel("Prior noise level (fraction of edges flipped)")
    ax.set_ylabel(ylabel)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.2f}"))


def build_single_figure(df, metric, ylabel):
    fig, ax = plt.subplots(figsize=(FULL_W, 3.2))
    fig.subplots_adjust(left=0.10, right=0.97, top=0.84, bottom=0.18)
    _oracle_panel(ax, df, metric, ylabel)

    topo_handles = [
        Line2D([0], [0], color=TOPO_PALETTE[t], lw=1.5,
               marker=TOPO_MARKERS[t], ms=4, label=TOPO_LABELS[t])
        for t in TOPOS
    ]
    baseline = Line2D([0], [0], color="grey", lw=0.7, linestyle="--",
                      label="SSGLasso baseline")
    fig.legend(handles=topo_handles + [baseline],
               loc="upper center", ncol=5, frameon=False, fontsize=7,
               bbox_to_anchor=(0.5, 1.00),
               handlelength=1.5, columnspacing=1.5)
    panel_label(ax, "A")
    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-aupr", default=os.path.join(FIGURES_DIR, "oracle_sensitivity_aupr.pdf"))
    parser.add_argument("--out-mcc",  default=os.path.join(FIGURES_DIR, "oracle_sensitivity_mcc.pdf"))
    parser.add_argument("--dpi", type=int, default=600)
    args = parser.parse_args()

    print("Loading data …")
    df = pd.read_csv(SUMMARY_CSV)
    oracle_found = [m for m in df["method"].unique() if "oracle" in m]
    print(f"  {len(df):,} rows — oracle methods: {sorted(oracle_found)}")
    if not oracle_found:
        print("ERROR: no piglasso_oracle_* rows found"); return

    print("Building AUPR figure …")
    save(build_single_figure(df, "aupr", "AUPR"), args.out_aupr, args.dpi)
    print("Building MCC figure …")
    save(build_single_figure(df, "mcc",  "MCC"),  args.out_mcc,  args.dpi)


if __name__ == "__main__":
    main()
