"""
plot_mcc_vs_n.py
-----------------
Nature-grade MCC vs. sample size figure (single column, 88 mm wide).
SSGLasso vs. PIGLasso at fixed p=160.

Usage:
    cd NODIS/
    python scripts/plot_mcc_vs_n.py [--out figures/mcc_vs_n.pdf]
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

from plot_style import FULL_W, METHOD_PALETTE, panel_label, save

N_VALS  = [100, 300, 500, 700, 900, 1100, 1300, 1500]
P       = 160
CONFIGS = [f"n{n}p{P}" for n in N_VALS]

METHODS = {
    "ssglasso":            ("SSGLasso", METHOD_PALETTE["ssglasso"],            "o", -1),
    "piglasso_oracle_n02": ("PIGLasso", METHOD_PALETTE["piglasso_oracle_n02"], "o", +1),
}

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
SUMMARY_CSV = os.path.join(RESULTS_DIR, "metrics_summary.csv")
FIGURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")


def build_figure(df):
    sub = df[(df["benchmark"] == "synthetic") & (df["config"].isin(CONFIGS))].copy()

    fig, ax = plt.subplots(figsize=(FULL_W, 3.2))
    fig.subplots_adjust(left=0.09, right=0.97, top=0.88, bottom=0.15)

    for method, (label, color, marker, lbl_sign) in METHODS.items():
        means, sds = [], []
        for n in N_VALS:
            vals = sub.loc[
                (sub["method"] == method) & (sub["config"] == f"n{n}p{P}"), "mcc"
            ].dropna()
            means.append(vals.mean() if len(vals) else np.nan)
            sds.append(vals.std()   if len(vals) else np.nan)
        means = np.array(means)
        sds   = np.array(sds)
        mask  = ~np.isnan(means)

        ax.plot(np.array(N_VALS)[mask], means[mask],
                color=color, marker=marker, markersize=4,
                linewidth=1.5, label=label, zorder=3)
        ax.fill_between(np.array(N_VALS)[mask],
                        (means - sds)[mask], (means + sds)[mask],
                        color=color, alpha=0.15, zorder=2)
        ax.errorbar(np.array(N_VALS)[mask], means[mask], yerr=sds[mask],
                    fmt="none", ecolor=color, elinewidth=0.7,
                    capsize=3, capthick=0.7, zorder=4)

        for n, m_val, s in zip(np.array(N_VALS)[mask], means[mask], sds[mask]):
            y_txt = (m_val + s + 0.05) if lbl_sign > 0 else (m_val - s - 0.015)
            va    = "bottom" if lbl_sign > 0 else "top"
            ax.text(n, y_txt, f"{m_val:.2f}", ha="center", va=va,
                    fontsize=6, fontweight="bold", color=color)

    ax.axhline(0.5, color="#009E73", linewidth=1.0, linestyle="--",
               label="Random baseline (MCC=0.5)", zorder=1)
    ax.axvline(513, color="#0072B2", linewidth=1.0, linestyle=":",
               label="GSE182616 (n=513)", zorder=1)

    ax.set_xlabel("Sample size (n)")
    ax.set_ylabel("MCC")
    ax.set_xlim(0, N_VALS[-1] + 100)
    ax.set_ylim(0.15, 1.05)
    ax.set_xticks(N_VALS)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.1f}"))
    ax.legend(frameon=False, loc="lower right", fontsize=7,
              handlelength=1.3, labelspacing=0.3)

    panel_label(ax, "A")
    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=os.path.join(FIGURES_DIR, "mcc_vs_n.pdf"))
    parser.add_argument("--dpi", type=int, default=600)
    args = parser.parse_args()
    print("Loading data …")
    df = pd.read_csv(SUMMARY_CSV, low_memory=False)
    print(f"  n-sweep configs found: {sorted(df.loc[df['config'].isin(CONFIGS), 'config'].unique())}")
    print("Building figure …")
    save(build_figure(df), args.out, args.dpi)


if __name__ == "__main__":
    main()
