"""
plot_mcc_synthetic_comparison.py
---------------------------------
Publication-quality 2-panel MCC comparison from mcc_synthetic_comparison.csv.
Matches the visual style of benchmark_comparison.pdf.

Panels:
  A  Grand-mean MCC — horizontal bar chart
  B  MCC per topology — grouped bar chart

Usage:
    cd NODIS/
    python scripts/plot_mcc_synthetic_comparison.py
    python scripts/plot_mcc_synthetic_comparison.py --out figures/mcc_synthetic_comparison.pdf
"""

import argparse
import os
import warnings

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Global style — identical to benchmark_comparison.py
# ---------------------------------------------------------------------------
matplotlib.rcParams.update({
    "font.family":        "sans-serif",
    "font.sans-serif":    ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size":          8,
    "axes.labelsize":     9,
    "axes.titlesize":     9.5,
    "xtick.labelsize":    7.5,
    "ytick.labelsize":    7.5,
    "legend.fontsize":    8,
    "figure.dpi":         300,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.linewidth":     0.8,
    "xtick.major.width":  0.8,
    "ytick.major.width":  0.8,
    "xtick.major.size":   3.5,
    "ytick.major.size":   3.5,
    "pdf.fonttype":       42,
    "ps.fonttype":        42,
})

# ---------------------------------------------------------------------------
# Colour + method metadata — identical to benchmark_comparison.py
# ---------------------------------------------------------------------------
METHODS_ORDER = ["glasso", "desparsified", "gglasso", "piglasso", "piglasso_corr"]

PALETTE = {
    "desparsified": "#5B9BD5",
    "glasso":       "#70AD47",
    "gglasso":      "#FFC000",
    "piglasso":     "#7B2D8B",   # purple — SSGLasso (no prior)
    "piglasso_corr":"#C00000",   # crimson — PIGLasso (with prior)
}
LABELS = {
    "desparsified": "Desparsified",
    "glasso":       "GLasso",
    "gglasso":      "GGLasso",
    "piglasso":     "SSGLasso",
    "piglasso_corr":"PIGLasso",
}
PIG_METHODS = {"piglasso", "piglasso_corr"}
ZO_PIG  = 5
ZO_BASE = 2

TOPOS = ["cluster", "hub", "random", "scale-free"]


# ---------------------------------------------------------------------------
# Panel A — Grand-mean MCC horizontal bar chart
# ---------------------------------------------------------------------------

def _grand_mean_bars(ax, grand: pd.DataFrame, title: str):
    # Sort ascending so best method ends up at top
    grand = grand.sort_values("MCC", ascending=True).reset_index(drop=True)
    methods = grand["method"].tolist()
    values  = grand["MCC"].tolist()
    n = len(methods)

    for i, (m, v) in enumerate(zip(methods, values)):
        color = PALETTE[m]
        lw    = 2.0 if m in PIG_METHODS else 0.8
        ec    = "#7B0000" if m == "piglasso_corr" else ("#4A1A5C" if m == "piglasso" else "#444444")
        zo    = (ZO_PIG if m == "piglasso_corr" else ZO_PIG - 1) if m in PIG_METHODS else ZO_BASE

        ax.barh(i, v, color=color, edgecolor=ec, linewidth=lw,
                zorder=zo, height=0.6)
        if m in PIG_METHODS:
            ax.barh(i, v, color="none", edgecolor=ec,
                    linewidth=2.2, zorder=zo + 1, height=0.6)

        # Value label inside bar
        ax.text(v - 0.005, i, f"{v:.3f}", va="center", ha="right",
                fontsize=7.5,
                color="white" if m in PIG_METHODS else "#333333",
                fontweight="bold" if m in PIG_METHODS else "normal",
                zorder=zo + 2)

    ax.set_yticks(np.arange(n))
    ax.set_yticklabels([LABELS[m] for m in methods])
    ax.set_xlabel("MCC")
    ax.set_xlim(0, 0.85)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.2f}"))
    ax.set_title(title, pad=6, fontweight="bold")
    ax.axvline(0, color="#bbbbbb", linewidth=0.6)


# ---------------------------------------------------------------------------
# Panel B — MCC per topology grouped bar chart
# ---------------------------------------------------------------------------

def _per_topology_bars(ax, per_topo: pd.DataFrame, title: str):
    methods = [m for m in METHODS_ORDER if m in per_topo["method"].unique()]

    n_topo  = len(TOPOS)
    n_meth  = len(methods)
    gw      = 0.82
    bw      = gw / n_meth
    offsets = np.linspace(-gw / 2 + bw / 2, gw / 2 - bw / 2, n_meth)

    for ti, topo in enumerate(TOPOS):
        for mi, m in enumerate(methods):
            row = per_topo[(per_topo["topology"] == topo) & (per_topo["method"] == m)]
            if row.empty:
                continue
            v     = float(row["MCC"].iloc[0])
            xp    = ti + offsets[mi]
            color = PALETTE[m]
            lw    = 1.8 if m in PIG_METHODS else 0.7
            ec    = "#7B0000" if m == "piglasso_corr" else ("#4A1A5C" if m == "piglasso" else "#444444")
            zo    = (ZO_PIG if m == "piglasso_corr" else ZO_PIG - 1) if m in PIG_METHODS else ZO_BASE

            ax.bar(xp, v, bw * 0.90, color=color, edgecolor=ec,
                   linewidth=lw, zorder=zo)
            if m in PIG_METHODS:
                ax.bar(xp, v, bw * 0.90, color="none", edgecolor=ec,
                       linewidth=2.0, zorder=zo + 1)

            ax.text(xp, v + 0.008, f"{v:.2f}", ha="center", va="bottom",
                    fontsize=6.0,
                    color="#7B0000" if m == "piglasso_corr" else ("#4A1A5C" if m == "piglasso" else "#444444"),
                    fontweight="bold" if m in PIG_METHODS else "normal")

    ax.set_xticks(np.arange(n_topo))
    ax.set_xticklabels([t.replace("-", "\u2011") for t in TOPOS])
    ax.set_ylabel("MCC")
    ax.set_ylim(0, 0.85)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.2f}"))
    ax.set_title(title, pad=6, fontweight="bold")
    ax.axhline(0, color="#bbbbbb", linewidth=0.6)


# ---------------------------------------------------------------------------
# Build figure
# ---------------------------------------------------------------------------

def build_figure(df: pd.DataFrame) -> plt.Figure:
    grand    = df[df["scope"] == "grand_mean"].copy()
    per_topo = df[df["scope"] == "per_topology"].copy()

    fig = plt.figure(figsize=(12, 5))
    gs  = GridSpec(1, 2, figure=fig,
                   wspace=0.38,
                   left=0.08, right=0.97,
                   top=0.85, bottom=0.12)

    ax_A = fig.add_subplot(gs[0, 0])
    _grand_mean_bars(ax_A, grand,
                     title="A   Grand-mean MCC (synthetic, all topologies)")

    ax_B = fig.add_subplot(gs[0, 1])
    _per_topology_bars(ax_B, per_topo,
                       title="B   MCC by network topology")

    # Shared legend — both piglasso (SSGLasso) and piglasso_corr (PIGLasso) shown
    methods_legend = [m for m in METHODS_ORDER if m in df["method"].unique()]
    handles = []
    for m in methods_legend:
        lw    = 2.2 if m in PIG_METHODS else 1.0
        ec    = "#7B0000" if m == "piglasso_corr" else ("#4A1A5C" if m == "piglasso" else PALETTE[m])
        label = LABELS[m] + ("  \u2605" if m in PIG_METHODS else "")
        handles.append(mpatches.Patch(facecolor=PALETTE[m], edgecolor=ec,
                                      linewidth=lw, label=label))

    fig.legend(handles=handles, loc="upper center",
               ncol=len(handles), frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, 1.00),
               handlelength=1.5, handleheight=0.95, columnspacing=2.0)

    fig.suptitle("Synthetic benchmark \u2014 MCC comparison",
                 y=1.05, fontsize=11, fontweight="bold")

    return fig


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        default=os.path.join(os.path.dirname(__file__), "..", "mcc_synthetic_comparison.csv"),
    )
    parser.add_argument("--out", default="figures/mcc_synthetic_comparison.pdf")
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    print("Loading data \u2026")
    df = pd.read_csv(args.csv)
    print(f"  {len(df)} rows \u2014 methods: {sorted(df['method'].unique())}")

    print("Building figure \u2026")
    fig = build_figure(df)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight")
    print(f"Saved \u2192 {args.out}")

    if args.out.endswith(".pdf"):
        png_out = args.out.replace(".pdf", ".png")
        fig.savefig(png_out, dpi=150, bbox_inches="tight")
        print(f"Saved \u2192 {png_out}")

    plt.close(fig)


if __name__ == "__main__":
    main()
