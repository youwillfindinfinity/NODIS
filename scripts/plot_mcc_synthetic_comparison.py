"""
plot_mcc_synthetic_comparison.py
---------------------------------
Nature-grade 2-panel MCC comparison (double column, 180 mm wide).

Panels:
  A  Grand-mean MCC — lollipop chart (dot + stem to axis)
  B  MCC per topology — violin + jitter strip (strip shows replicate spread)

Usage:
    cd NODIS/
    python scripts/plot_mcc_synthetic_comparison.py [--out figures/mcc_synthetic_comparison.pdf]
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

from plot_style import (
    FULL_W, METHOD_PALETTE, METHOD_LABELS, PIG_METHODS, panel_label, save,
)

METHODS_ORDER      = ["glasso", "desparsified", "gglasso", "ssglasso", "piglasso_oracle_n02"]
PALETTE            = METHOD_PALETTE
LABELS             = METHOD_LABELS
TOPOS              = ["cluster", "hub", "random", "scale-free"]
CONFIGS_BENCHMARK  = ["n513p164"]

ZO_PIG  = 5
ZO_BASE = 2

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
SUMMARY_CSV = os.path.join(RESULTS_DIR, "metrics_summary.csv")
FIGURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")


def _grand_means(df):
    small3   = df[(df["benchmark"] == "synthetic") & (df["config"].isin(CONFIGS_BENCHMARK))]
    seed_col = "seed_offset" if "seed_offset" in small3.columns else None
    rows = []
    for m in METHODS_ORDER:
        sub = small3.loc[small3["method"] == m]
        if sub.empty:
            continue
        if seed_col and sub[seed_col].nunique() > 1:
            per_seed = sub.groupby(seed_col)["mcc"].mean()
            rows.append({"method": m, "MCC": per_seed.mean(), "SD": per_seed.std()})
        else:
            vals = sub["mcc"].dropna()
            rows.append({"method": m, "MCC": vals.mean(), "SD": vals.std()})
    return pd.DataFrame(rows)


def _per_topology_vals(df):
    small3   = df[(df["benchmark"] == "synthetic") & (df["config"].isin(CONFIGS_BENCHMARK))]
    methods = [m for m in METHODS_ORDER if m in small3["method"].unique()]
    out = {}
    for m in methods:
        out[m] = {}
        for topo in TOPOS:
            vals = small3.loc[(small3["method"] == m) & (small3["topology"] == topo),
                              "mcc"].dropna().values
            out[m][topo] = vals
    return out, methods


def _lollipop(ax, grand):
    """Panel A: horizontal lollipop chart for grand mean MCC."""
    grand = grand.sort_values("MCC", ascending=True).reset_index(drop=True)
    for i, row in grand.iterrows():
        m, v, sd = row["method"], row["MCC"], row["SD"]
        color = PALETTE.get(m, "#888888")
        zo    = ZO_PIG if m in PIG_METHODS else ZO_BASE
        lw    = 1.8 if m in PIG_METHODS else 1.0
        ms    = 7.0 if m in PIG_METHODS else 5.5
        mk    = "D" if m in PIG_METHODS else "o"
        # stem
        ax.plot([0, v], [i, i], color=color, lw=lw, zorder=zo - 1,
                solid_capstyle="round", alpha=0.55)
        # CI bar
        ax.errorbar(v, i, xerr=sd, fmt="none", ecolor=color,
                    elinewidth=lw * 0.7, capsize=3.0, capthick=0.7, zorder=zo)
        # dot
        ax.scatter(v, i, color=color, s=ms ** 2, marker=mk,
                   edgecolors="white", linewidths=0.5, zorder=zo + 1)
        # value label
        ax.text(v + sd + 0.018, i, f"{v:.3f}", va="center", ha="left", fontsize=7,
                color="#222222",
                fontweight="bold" if m in PIG_METHODS else "normal",
                zorder=zo + 2)
    ax.set_yticks(np.arange(len(grand)))
    ax.set_yticklabels([LABELS.get(m, m) for m in grand["method"]])
    ax.set_xlabel("MCC  (mean ± SD)")
    ax.set_xlim(0, 1.15)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.2f}"))
    ax.axvline(0, color="#cccccc", linewidth=0.4)
    ax.set_title("Grand-mean MCC  (n=513, p=164)", pad=4, fontweight="bold")


def _violin_strip_topology(ax, val_dict, methods):
    """Panel B: violin + jitter strip per topology × method."""
    n_meth  = len(methods)
    gw, bw  = 0.86, 0.86 / n_meth
    offsets = np.linspace(-gw / 2 + bw / 2, gw / 2 - bw / 2, n_meth)
    rng = np.random.default_rng(42)

    for ti, topo in enumerate(TOPOS):
        for mi, m in enumerate(methods):
            vals = val_dict[m].get(topo, np.array([]))
            if len(vals) < 4:
                continue
            xp    = ti + offsets[mi]
            color = PALETTE.get(m, "#888888")
            zo    = ZO_PIG if m in PIG_METHODS else ZO_BASE

            vp = ax.violinplot(vals, positions=[xp], widths=bw * 0.80,
                               showmedians=False, showextrema=False)
            for body in vp["bodies"]:
                body.set_facecolor(color)
                body.set_alpha(0.50 if m in PIG_METHODS else 0.28)
                body.set_edgecolor(color); body.set_linewidth(0.4); body.set_zorder(zo)
                for path in body.get_paths():
                    path.vertices[:, 1] = np.clip(path.vertices[:, 1], 0.0, 1.0)
            # jitter
            jit = rng.uniform(-bw * 0.18, bw * 0.18, len(vals))
            ax.scatter(xp + jit, vals, color=color, s=2.2, alpha=0.28,
                       edgecolors="none", zorder=zo + 1)
            # IQR + median
            q25, med, q75 = np.percentile(vals, [25, 50, 75])
            ax.vlines(xp, q25, q75, color=color,
                      linewidth=1.8 if m in PIG_METHODS else 0.9,
                      zorder=zo + 2, capstyle="round")
            ax.scatter(xp, med, color="white", s=9,
                       edgecolors=color, linewidths=0.8, zorder=zo + 3)
            # mean label
            mu = vals.mean()
            ax.text(xp, q75 + 0.03, f"{mu:.2f}", ha="center", va="bottom",
                    fontsize=5.0, color=color,
                    fontweight="bold" if m in PIG_METHODS else "normal")

    ax.set_xticks(np.arange(len(TOPOS)))
    ax.set_xticklabels(TOPOS)
    ax.set_xlabel("Network topology")
    ax.set_ylabel("MCC")
    ax.set_ylim(-0.05, 1.15)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.2f}"))
    ax.axhline(0, color="#cccccc", linewidth=0.4)
    ax.set_title("MCC per topology  (n=513, p=164)", pad=4, fontweight="bold")


def build_figure(df):
    grand          = _grand_means(df)
    val_dict, mths = _per_topology_vals(df)

    fig = plt.figure(figsize=(FULL_W, 3.2))
    gs  = GridSpec(1, 2, figure=fig,
                   width_ratios=[1, 1.9],
                   wspace=0.42,
                   left=0.10, right=0.97,
                   top=0.88, bottom=0.16)

    ax_A = fig.add_subplot(gs[0, 0])
    ax_B = fig.add_subplot(gs[0, 1])

    _lollipop(ax_A, grand)
    _violin_strip_topology(ax_B, val_dict, mths)
    panel_label(ax_A, "A")
    panel_label(ax_B, "B")

    methods_present = [m for m in METHODS_ORDER if m in df["method"].unique()]
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
    parser.add_argument("--out", default=os.path.join(FIGURES_DIR, "mcc_synthetic_comparison.pdf"))
    parser.add_argument("--dpi", type=int, default=600)
    args = parser.parse_args()
    print("Loading data …")
    df = pd.read_csv(SUMMARY_CSV)
    print(f"  {len(df):,} rows — methods: {sorted(df['method'].unique())}")
    print("Building figure …")
    save(build_figure(df), args.out, args.dpi)


if __name__ == "__main__":
    main()
