"""
plot_mcc_synthetic_comparison.py
---------------------------------
Publication-quality 2-panel MCC comparison.
Reads directly from results/metrics_summary.csv.

Panels:
  A  Grand-mean MCC — horizontal bar chart
  B  MCC per topology — grouped bar chart

Methods shown:
  glasso, desparsified, gglasso, piglasso (SSGLasso),
  piglasso_corr (PIGLasso), piglasso_oracle_n02 (PIGLasso oracle 20%)

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
# Global style
# ---------------------------------------------------------------------------
matplotlib.rcParams.update({
    "font.family":        "sans-serif",
    "font.sans-serif":    ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size":          12,
    "axes.labelsize":     13,
    "axes.titlesize":     14,
    "xtick.labelsize":    12,
    "ytick.labelsize":    12,
    "legend.fontsize":    12,
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
# Colour + method metadata
# ---------------------------------------------------------------------------
METHODS_ORDER = ["glasso", "desparsified", "gglasso",
                 "ssglasso", "piglasso_oracle_n02"]

PALETTE = {
    "glasso":              "#4C72B0",
    "desparsified":        "#F78154",
    "gglasso":             "#5CAD6E",
    "ssglasso":            "#F2C14E",   # SSGLasso (no prior)
    "piglasso_oracle_n02": "#B4436C",   # PIGLasso (with prior)
}
LABELS = {
    "glasso":              "GLasso",
    "desparsified":        "Desparsified",
    "gglasso":             "GGLasso",
    "ssglasso":            "SSGLasso",
    "piglasso_oracle_n02": "PIGLasso",
}
# Methods drawn with highlighted style
PIG_METHODS   = {"ssglasso", "piglasso_oracle_n02"}
ORACLE_METHOD = "piglasso_oracle_n02"

ZO_PIG  = 5
ZO_BASE = 2

TOPOS = ["cluster", "hub", "random", "scale-free"]
CONFIGS_SMALL3 = ["n513p164"]

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
SUMMARY_CSV = os.path.join(RESULTS_DIR, "metrics_summary.csv")
FIGURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def _grand_means(df: pd.DataFrame) -> pd.DataFrame:
    """Mean ± SD MCC across seed replications (grand mean per seed, then SD across seeds)."""
    small3 = df[
        (df["benchmark"] == "synthetic") &
        (df["config"].isin(CONFIGS_SMALL3))
    ]
    seed_col = "seed_offset" if "seed_offset" in small3.columns else None
    rows = []
    for m in METHODS_ORDER:
        sub = small3.loc[small3["method"] == m, :]
        if sub.empty:
            continue
        if seed_col and sub[seed_col].nunique() > 1:
            # Mean per seed replication, then SD across seeds
            per_seed = sub.groupby(seed_col)["mcc"].mean()
            rows.append({"method": m, "MCC": per_seed.mean(), "SD": per_seed.std()})
        else:
            vals = sub["mcc"].dropna()
            rows.append({"method": m, "MCC": vals.mean(), "SD": vals.std()})
    return pd.DataFrame(rows)


def _per_topology(df: pd.DataFrame) -> pd.DataFrame:
    """Mean ± SD MCC per (method, topology) across seed replications."""
    small3 = df[
        (df["benchmark"] == "synthetic") &
        (df["config"].isin(CONFIGS_SMALL3))
    ]
    seed_col = "seed_offset" if "seed_offset" in small3.columns else None
    rows = []
    for m in METHODS_ORDER:
        for topo in TOPOS:
            sub = small3.loc[
                (small3["method"] == m) & (small3["topology"] == topo), :
            ]
            if sub.empty:
                continue
            if seed_col and sub[seed_col].nunique() > 1:
                per_seed = sub.groupby(seed_col)["mcc"].mean()
                rows.append({"method": m, "topology": topo,
                             "MCC": per_seed.mean(), "SD": per_seed.std()})
            else:
                vals = sub["mcc"].dropna()
                rows.append({"method": m, "topology": topo,
                             "MCC": vals.mean(), "SD": vals.std()})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Panel A — Grand-mean MCC horizontal bar chart
# ---------------------------------------------------------------------------

def _grand_mean_bars(ax, grand: pd.DataFrame, title: str):
    grand = grand.sort_values("MCC", ascending=True).reset_index(drop=True)
    methods = grand["method"].tolist()
    values  = grand["MCC"].tolist()
    sds     = grand["SD"].tolist()
    n = len(methods)

    for i, (m, v, sd) in enumerate(zip(methods, values, sds)):
        color = PALETTE[m]
        lw    = 0.8
        ec    = "#444444"
        zo    = ZO_PIG if m in PIG_METHODS else ZO_BASE
        ax.barh(i, v, color=color, edgecolor=ec, linewidth=lw,
                zorder=zo, height=0.6)
        ax.errorbar(v, i, xerr=sd, fmt="none", ecolor="#333333",
                    elinewidth=0.9, capsize=3, capthick=0.9, zorder=zo + 1)

        ax.text(v + 0.012, i, f"{v:.3f}", va="center", ha="left",
                fontsize=11,
                color="#333333",
                fontweight="bold" if m in PIG_METHODS else "normal",
                zorder=zo + 2)

    ax.set_yticks(np.arange(n))
    ax.set_yticklabels([LABELS[m] for m in methods])
    ax.set_xlabel("MCC")
    ax.set_ylabel("Inference method")
    ax.set_xlim(0, 1.02)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.2f}"))
    ax.axvline(0, color="#bbbbbb", linewidth=0.6)


# ---------------------------------------------------------------------------
# Panel B — MCC per topology grouped bar chart
# ---------------------------------------------------------------------------

def _per_topology_bars(ax, per_topo: pd.DataFrame, title: str):
    methods = [m for m in METHODS_ORDER if m in per_topo["method"].unique()]
    n_topo  = len(TOPOS)
    n_meth  = len(methods)
    gw      = 0.90
    bw      = gw / n_meth
    offsets = np.linspace(-gw / 2 + bw / 2, gw / 2 - bw / 2, n_meth)

    for ti, topo in enumerate(TOPOS):
        for mi, m in enumerate(methods):
            row = per_topo[(per_topo["topology"] == topo) & (per_topo["method"] == m)]
            if row.empty:
                continue
            v     = float(row["MCC"].iloc[0])
            sd    = float(row["SD"].iloc[0])
            xp    = ti + offsets[mi]
            color = PALETTE[m]
            lw    = 0.7
            ec    = "#444444"
            zo    = ZO_PIG if m in PIG_METHODS else ZO_BASE
            ax.bar(xp, v, bw * 0.90, color=color, edgecolor=ec,
                   linewidth=lw, zorder=zo)
            ax.errorbar(xp, v, yerr=sd, fmt="none", ecolor="#333333",
                        elinewidth=0.8, capsize=2, capthick=0.8, zorder=zo + 1)

            ax.text(xp, v + sd + 0.015, f"{v:.2f}", ha="center", va="bottom",
                    fontsize=9,
                    color="#444444",
                    fontweight="bold" if m in PIG_METHODS else "normal")

    ax.set_xticks(np.arange(n_topo))
    ax.set_xticklabels([t for t in TOPOS])
    ax.set_xlabel("Network topology")
    ax.set_ylabel("MCC")
    ax.set_ylim(0, 1.10)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.2f}"))
    ax.axhline(0, color="#bbbbbb", linewidth=0.6)


# ---------------------------------------------------------------------------
# Build figure
# ---------------------------------------------------------------------------

def build_figure(df: pd.DataFrame) -> plt.Figure:
    grand    = _grand_means(df)
    per_topo = _per_topology(df)

    fig = plt.figure(figsize=(16, 5))
    gs  = GridSpec(1, 2, figure=fig,
                   width_ratios=[1, 1.8],
                   wspace=0.35,
                   left=0.08, right=0.97,
                   top=0.85, bottom=0.12)

    ax_A = fig.add_subplot(gs[0, 0])
    _grand_mean_bars(ax_A, grand,
                     title="A   Grand-mean MCC (synthetic, all topologies)")

    ax_B = fig.add_subplot(gs[0, 1])
    _per_topology_bars(ax_B, per_topo,
                       title="B   MCC by network topology")

    # Shared legend
    methods_legend = [m for m in METHODS_ORDER if m in df["method"].unique()]
    handles = []
    for m in methods_legend:
        lw    = 1.0
        ec    = PALETTE[m]
        label = LABELS[m].replace("\n", " ")
        handles.append(mpatches.Patch(facecolor=PALETTE[m], edgecolor=ec,
                                      linewidth=lw, label=label))

    fig.legend(handles=handles, loc="upper center",
               ncol=len(handles), frameon=False, fontsize=13,
               bbox_to_anchor=(0.5, 1.00),
               handlelength=1.5, handleheight=0.95, columnspacing=1.8)


    return fig


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=os.path.join(FIGURES_DIR, "mcc_synthetic_comparison.pdf"))
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    print("Loading data \u2026")
    df = pd.read_csv(SUMMARY_CSV)
    methods_found = sorted(df["method"].unique())
    print(f"  {len(df):,} rows \u2014 methods: {methods_found}")

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
