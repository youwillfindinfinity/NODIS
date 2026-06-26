"""
plot_scalability.py
--------------------
Nature-grade dual-panel scalability figure (double column, 180 mm wide).

Panels:
  A  AUPR vs. problem size — line + SEM band across all configs
  B  Wall time vs. problem size — log-scale line plot

Usage:
    cd NODIS/
    python scripts/plot_scalability.py [--out figures/scalability.pdf]
"""

import argparse
import os
import sys

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec

sys.path.insert(0, os.path.dirname(__file__))
import plot_style
plot_style.apply()

from plot_style import FULL_W, METHOD_PALETTE, METHOD_LABELS, PIG_METHODS, panel_label, save

METHODS  = ["desparsified", "glasso", "gglasso", "ssglasso", "piglasso_oracle_n02"]
PALETTE  = METHOD_PALETTE
LABELS   = METHOD_LABELS
ZO_PIG   = 5
ZO_BASE  = 2

# Ordered configs that represent growing problem size
ALL_CONFIGS = ["n100p50", "n237p78", "n513p164", "n1026p328"]
CONFIG_XLABS = ["n=100\np=50", "n=237\np=78", "n=513\np=164", "n=1026\np=328"]

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
SUMMARY_CSV = os.path.join(RESULTS_DIR, "metrics_summary.csv")
FIGURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")


def _line_panel(ax, df, metric, ylabel, title, log_y=False):
    syn      = df[df["benchmark"] == "synthetic"]
    present  = [c for c in ALL_CONFIGS if c in syn["config"].unique()]
    x        = np.arange(len(present))
    methods  = [m for m in METHODS if m in syn["method"].unique()]

    for m in methods:
        mus, sems = [], []
        for cfg in present:
            vals = syn.loc[(syn["method"] == m) & (syn["config"] == cfg),
                           metric].dropna()
            mus.append(vals.mean() if len(vals) else np.nan)
            sems.append(vals.sem()  if len(vals) else np.nan)
        mu, sem = np.array(mus), np.array(sems)
        mask = ~np.isnan(mu)
        color = PALETTE.get(m, "#888888")
        lw, ms = (1.5, 5.5) if m in PIG_METHODS else (0.9, 3.2)
        mk = "D" if m in PIG_METHODS else "o"
        zo = ZO_PIG if m in PIG_METHODS else ZO_BASE
        ax.plot(x[mask], mu[mask], color=color, lw=lw, marker=mk, ms=ms,
                zorder=zo, label=LABELS.get(m, m), solid_capstyle="round")
        ax.fill_between(x[mask], (mu - sem)[mask], (mu + sem)[mask],
                        color=color, alpha=0.12, zorder=zo - 1)

    ax.set_xticks(np.arange(len(present)))
    ax.set_xticklabels([CONFIG_XLABS[ALL_CONFIGS.index(c)] for c in present])
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=4, fontweight="bold")
    if log_y:
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda v, _: f"{v:.0f}s" if v >= 1 else f"{v:.2f}s")
        )
    else:
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.legend(loc="lower left" if not log_y else "lower right",
              fontsize=6.5, handlelength=1.2, labelspacing=0.3)


def build_figure(df):
    fig = plt.figure(figsize=(FULL_W, 3.0))
    gs  = GridSpec(1, 2, figure=fig,
                   wspace=0.40,
                   left=0.09, right=0.97,
                   top=0.88, bottom=0.18)

    ax_A = fig.add_subplot(gs[0, 0])
    ax_B = fig.add_subplot(gs[0, 1])

    _line_panel(ax_A, df, "aupr",         "AUPR",           "AUPR vs. problem size")
    _line_panel(ax_B, df, "wall_seconds", "Wall time (s)",  "Runtime vs. problem size",
                log_y=True)

    panel_label(ax_A, "A")
    panel_label(ax_B, "B")

    methods_present = [m for m in METHODS if m in df["method"].unique()]
    handles = [
        mpatches.Patch(facecolor=PALETTE.get(m, "#888"), edgecolor="none",
                       label=LABELS.get(m, m))
        for m in methods_present
    ]
    fig.legend(handles=handles, loc="upper center", ncol=len(handles),
               frameon=False, fontsize=7, bbox_to_anchor=(0.5, 1.00),
               handlelength=1.2, handleheight=0.8, columnspacing=1.5)
    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=os.path.join(FIGURES_DIR, "scalability.pdf"))
    parser.add_argument("--dpi", type=int, default=600)
    args = parser.parse_args()

    df = pd.read_csv(SUMMARY_CSV, low_memory=False)
    save(build_figure(df), args.out, args.dpi)


if __name__ == "__main__":
    main()
