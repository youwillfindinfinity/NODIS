"""
plot_benchmark_comparison.py
----------------------------
Publication-quality 6-panel benchmark comparison.
Highlights PIGLasso with crimson throughout.

Panels:
  A  Radar chart — grand mean across AUPR, AUROC, MCC, F1_opt (synthetic)
  B  MCC per topology — violin+IQR (synthetic, 3 configs pooled)
  C  Scalability — AUPR vs. problem size (line plot with CI bands)
  D  Diffusion recovery — normalised Spearman by topology & method
  E  DREAM5 — AUPR vs. gene-set size (line, p=200/500/1000)
  F  Computational cost — median wall time at n=513, p=164 (log-scale)

Usage:
    cd NODIS/
    python scripts/plot_benchmark_comparison.py
    python scripts/plot_benchmark_comparison.py --out figures/benchmark_comparison.pdf
"""

import argparse
import math
import os
import warnings

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.path import Path

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Global style
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
# Colour + method metadata
# ---------------------------------------------------------------------------
METHODS_MAIN = ["desparsified", "glasso", "gglasso", "ssglasso", "piglasso_corr"]

PALETTE = {
    "desparsified": "#5B9BD5",
    "glasso":       "#70AD47",
    "gglasso":      "#FFC000",
    "ssglasso":     "#7B2D8B",   # purple — SSGLasso (no prior)
    "piglasso_corr":"#C00000",   # crimson — PIGLasso (with prior)
}
LABELS = {
    "desparsified": "Desparsified",
    "glasso":       "GLasso",
    "gglasso":      "GGLasso",
    "ssglasso":     "SSGLasso",
    "piglasso_corr":"PIGLasso",
}
PIG_METHODS = {"ssglasso", "piglasso_corr"}
ZO_PIG   = 5
ZO_BASE  = 2

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
FIGURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")
SUMMARY_CSV = os.path.join(RESULTS_DIR, "metrics_summary.csv")


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _load() -> pd.DataFrame:
    df = pd.read_csv(SUMMARY_CSV)
    # Normalise method labels: treat piglasso_corr as piglasso for dream5 panel
    return df


def _methods_present(df, pool):
    present = df["method"].unique()
    return [m for m in pool if m in present]


# ---------------------------------------------------------------------------
# Panel A — Radar chart
# ---------------------------------------------------------------------------

def _radar(ax, data, methods, metrics, metric_labels, title):
    """
    Filled radar chart.  Each method is a closed polygon over the metrics.
    Values normalised so that best method = 1 on each axis.
    """
    n = len(metrics)
    angles = np.linspace(0, 2 * math.pi, n, endpoint=False).tolist()
    angles += angles[:1]  # close polygon

    ax.set_theta_offset(math.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_labels, size=8.5, fontweight="bold")
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], size=6.5, color="grey")
    ax.tick_params(pad=7)
    ax.spines["polar"].set_color("#cccccc")
    ax.grid(color="#dddddd", linewidth=0.7)

    # Compute per-metric grand means (across all 3 configs, 4 topos, 50 reps)
    small3 = data[data["config"].isin(["n100p50", "n237p78", "n513p164"])]
    raw = {}
    for m in methods:
        sub = small3[small3["method"] == m]
        raw[m] = [sub[met].mean() for met in metrics]

    # Normalise: scale so max across methods = 1 per metric
    maxes = [max(raw[m][i] for m in methods if raw[m][i] == raw[m][i])
             for i in range(n)]
    maxes = [v if v > 0 else 1.0 for v in maxes]
    norm = {m: [raw[m][i] / maxes[i] for i in range(n)] for m in methods}

    for m in methods:
        vals = norm[m] + norm[m][:1]
        color = PALETTE[m]
        lw    = 2.4 if m in PIG_METHODS else 1.2
        # piglasso_corr (PIGLasso) on top; ssglasso (SSGLasso) just below with dashed line
        zo    = (ZO_PIG if m == "piglasso_corr" else ZO_PIG - 1) if m in PIG_METHODS else ZO_BASE
        ls    = "--" if m == "ssglasso" else "-"
        alpha_fill = 0.15 if m in PIG_METHODS else 0.06
        ax.plot(angles, vals, color=color, linewidth=lw, linestyle=ls,
                zorder=zo, label=LABELS[m])
        ax.fill(angles, vals, color=color, alpha=alpha_fill, zorder=zo - 1)
        # Mark vertices for both PIG methods
        if m in PIG_METHODS:
            ax.scatter(angles[:-1], vals[:-1], color=color, s=25,
                       zorder=zo + 1, edgecolors="white", linewidths=0.8)

    ax.set_title(title, pad=18, fontweight="bold", size=9.5)


# ---------------------------------------------------------------------------
# Panel B — MCC per topology  (violin + IQR)
# ---------------------------------------------------------------------------

def _violin_topology(ax, data, metric, methods, ylabel, title, ylim=None):
    small3 = data[data["config"].isin(["n100p50", "n237p78", "n513p164"])]
    topos = ["cluster", "hub", "random", "scale-free"]
    n_topo  = len(topos)
    n_meth  = len(methods)
    gw      = 0.88
    bw      = gw / n_meth
    offsets = np.linspace(-gw / 2 + bw / 2, gw / 2 - bw / 2, n_meth)

    for ti, topo in enumerate(topos):
        for mi, m in enumerate(methods):
            vals = small3.loc[
                (small3["method"] == m) & (small3["topology"] == topo), metric
            ].dropna().values
            if len(vals) < 4:
                continue
            xp    = ti + offsets[mi]
            color = PALETTE[m]
            lw    = 1.8 if m in PIG_METHODS else 0.7
            zo    = ZO_PIG if m in PIG_METHODS else ZO_BASE
            alpha = 0.75 if m in PIG_METHODS else 0.45

            vp = ax.violinplot(vals, positions=[xp], widths=bw * 0.88,
                               showmedians=False, showextrema=False)
            for body in vp["bodies"]:
                body.set_facecolor(color)
                body.set_alpha(alpha)
                body.set_edgecolor(color)
                body.set_linewidth(lw)
                body.set_zorder(zo)
                # Hard-clip violins to [0,1]
                for path in body.get_paths():
                    path.vertices[:, 1] = np.clip(path.vertices[:, 1], 0.0, 1.0)

            q25, med, q75 = np.percentile(vals, [25, 50, 75])
            iqr_lw = lw * 2.0 if m in PIG_METHODS else lw * 1.4
            ax.vlines(xp, q25, q75, color=color, linewidth=iqr_lw,
                      zorder=zo + 1, capstyle="round")
            ax.scatter(xp, med, color="white", s=14,
                       edgecolors=color, linewidths=lw, zorder=zo + 2)

    ax.set_xticks(np.arange(n_topo))
    ax.set_xticklabels([t for t in topos])  # non-breaking hyphen
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=4, fontweight="bold")
    if ylim:
        ax.set_ylim(*ylim)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.2f}"))
    ax.axhline(0, color="#bbbbbb", linewidth=0.6, linestyle=":")


# ---------------------------------------------------------------------------
# Panel C — Scalability line plot
# ---------------------------------------------------------------------------

CONFIGS      = ["n100p50", "n237p78", "n513p164"]
CONFIG_XLABS = ["n=100\np=50", "n=237\np=78", "n=513\np=164"]


def _scalability(ax, data, metric, methods, ylabel, title):
    small3 = data[data["config"].isin(CONFIGS)]
    x = np.arange(len(CONFIGS))
    for m in methods:
        mus, sems = [], []
        for cfg in CONFIGS:
            vals = small3.loc[
                (small3["method"] == m) & (small3["config"] == cfg), metric
            ].dropna()
            mus.append(vals.mean() if len(vals) else np.nan)
            sems.append(vals.sem()  if len(vals) else 0.0)
        mu  = np.array(mus)
        sem = np.array(sems)
        color = PALETTE[m]
        lw    = 2.4 if m in PIG_METHODS else 1.3
        ms    = 8   if m in PIG_METHODS else 4.5
        mk    = "D" if m in PIG_METHODS else "o"
        zo    = ZO_PIG if m in PIG_METHODS else ZO_BASE
        mec   = "#7B0000" if m in PIG_METHODS else color
        mew   = 1.4 if m in PIG_METHODS else 0.5
        ax.plot(x, mu, color=color, lw=lw, marker=mk, ms=ms, zorder=zo,
                label=LABELS[m], markeredgecolor=mec, markeredgewidth=mew,
                solid_capstyle="round")
        ax.fill_between(x, mu - sem, mu + sem, color=color, alpha=0.13,
                        zorder=zo - 1)

    ax.set_xticks(x)
    ax.set_xticklabels(CONFIG_XLABS)
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=4, fontweight="bold")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.legend(loc="lower right", frameon=False, fontsize=7.5,
              handlelength=1.6, labelspacing=0.4)


# ---------------------------------------------------------------------------
# Panel D — Diffusion recovery per topology (grouped bar)
# ---------------------------------------------------------------------------

def _diffusion_topology(ax, data, methods, title):
    """
    Grouped bar: DiffSp_norm per topology (small3 configs only).
    """
    small3_diff = data[
        (data["benchmark"] == "diffusion") &
        (data["config"].isin(CONFIGS))
    ]
    topos   = ["cluster", "hub", "random", "scale-free"]
    n_topo  = len(topos)
    n_meth  = len(methods)
    gw      = 0.80
    bw      = gw / n_meth
    offsets = np.linspace(-gw / 2 + bw / 2, gw / 2 - bw / 2, n_meth)

    for ti, topo in enumerate(topos):
        for mi, m in enumerate(methods):
            vals = small3_diff.loc[
                (small3_diff["method"] == m) &
                (small3_diff["topology"] == topo),
                "diffusion_spearman_norm"
            ].dropna()
            if vals.empty:
                continue
            mu  = vals.mean()
            sem = vals.sem()
            xp    = ti + offsets[mi]
            color = PALETTE[m]
            lw    = 1.8 if m in PIG_METHODS else 0.7
            ec    = "#7B0000" if m in PIG_METHODS else "#444444"
            zo    = ZO_PIG if m in PIG_METHODS else ZO_BASE
            ax.bar(xp, mu, bw * 0.88, color=color, edgecolor=ec,
                   linewidth=lw, zorder=zo)
            ax.errorbar(xp, mu, yerr=sem, fmt="none", color="black",
                        capsize=2, capthick=0.7, linewidth=0.7, zorder=zo + 1)
            if m in PIG_METHODS:
                ax.bar(xp, mu, bw * 0.88, color="none", edgecolor=ec,
                       linewidth=2.0, zorder=zo + 1)

    ax.set_xticks(np.arange(n_topo))
    ax.set_xticklabels([t for t in topos])
    ax.axhline(0, color="#999999", linewidth=0.8, linestyle="--", alpha=0.7)
    ax.set_ylabel("Normalised Spearman ρ")
    ax.set_title(title, pad=4, fontweight="bold")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.2f}"))


# ---------------------------------------------------------------------------
# Panel E — DREAM5 line plot (AUPR vs p)
# ---------------------------------------------------------------------------

def _dream5_line(ax, data, title):
    """
    Line plot: AUPR vs. gene-set size (p = 200, 500, 1000).
    Uses piglasso_corr as PIGLasso representative (no plain piglasso in dream5).
    """
    d5 = data[(data["benchmark"] == "dream5") & (data["network"] == 1)].copy()
    if d5.empty:
        ax.text(0.5, 0.5, "DREAM5 data\nnot available",
                ha="center", va="center", transform=ax.transAxes, color="grey")
        ax.set_title(title, pad=4, fontweight="bold")
        return

    # Remove piglasso_string (duplicate) for cleanliness
    d5 = d5[d5["method"] != "piglasso_string"]

    # Use piglasso_corr directly as PIGLasso; plain piglasso has no dream5 data
    disp_methods = ["desparsified", "glasso", "gglasso", "piglasso_corr"]
    ps = [200, 500, 1000]
    x  = np.arange(len(ps))

    for m in disp_methods:
        sub = d5[d5["method"] == m]
        mus = [sub.loc[sub["p"] == p, "aupr"].mean() for p in ps]
        color = PALETTE[m]
        lw    = 2.4 if m in PIG_METHODS else 1.3
        ms    = 7   if m in PIG_METHODS else 4
        mk    = "D" if m in PIG_METHODS else "o"
        zo    = ZO_PIG if m in PIG_METHODS else ZO_BASE
        mec   = "#7B0000" if m in PIG_METHODS else color
        ax.plot(x, mus, color=color, lw=lw, marker=mk, ms=ms, zorder=zo,
                label=LABELS[m], markeredgecolor=mec, markeredgewidth=1.2 if m in PIG_METHODS else 0.5)

    # Coverage annotation band
    coverage = [0.408, 0.133, 0.087]
    ax2 = ax.twinx()
    ax2.fill_between(x, 0, coverage, color="#dddddd", alpha=0.45,
                     label="Gold-std coverage")
    ax2.set_ylim(0, 1.0)
    ax2.set_yticks([0, 0.25, 0.5])
    ax2.set_yticklabels(["0%", "25%", "50%"], color="grey", size=6.5)
    ax2.set_ylabel("Partial corr\ncoverage", color="grey", size=7)
    ax2.tick_params(colors="grey")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_color("#cccccc")

    ax.set_xticks(x)
    ax.set_xticklabels(["p=200", "p=500", "p=1000"])
    ax.set_ylabel("AUPR")
    ax.set_ylim(0.0, 0.25)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.2f}"))
    ax.set_title(title, pad=4, fontweight="bold")
    ax.legend(loc="upper right", frameon=False, fontsize=7, handlelength=1.4)


# ---------------------------------------------------------------------------
# Panel F — Wall time (log-scale horizontal bars)
# ---------------------------------------------------------------------------

def _wall_time(ax, data, title):
    syn513 = data[
        (data["benchmark"] == "synthetic") & (data["config"] == "n513p164")
    ]
    methods = [m for m in METHODS_MAIN if m in syn513["method"].unique()]
    n = len(methods)

    for i, m in enumerate(reversed(methods)):
        vals = syn513.loc[syn513["method"] == m, "wall_seconds"].dropna()
        if vals.empty:
            continue
        med  = vals.median()
        p25  = vals.quantile(0.25)
        p75  = vals.quantile(0.75)
        color = PALETTE[m]
        lw    = 2.0 if m in PIG_METHODS else 0.8
        ec    = "#7B0000" if m in PIG_METHODS else "#444444"
        zo    = ZO_PIG if m in PIG_METHODS else ZO_BASE

        ax.barh(i, med, color=color, edgecolor=ec, linewidth=lw,
                zorder=zo, height=0.55)
        ax.errorbar(med, i, xerr=[[med - p25], [p75 - med]],
                    fmt="none", color="#333333", capsize=2.5,
                    capthick=0.8, linewidth=0.8, zorder=zo + 1)
        if m in PIG_METHODS:
            ax.barh(i, med, color="none", edgecolor=ec,
                    linewidth=2.2, zorder=zo + 1, height=0.55)
        # Value label
        ax.text(med * 0.5, i, f"{med:.0f}s", va="center", ha="center",
                fontsize=6.5, color="white" if m in PIG_METHODS else "#333333",
                fontweight="bold" if m in PIG_METHODS else "normal")

    ax.set_yticks(np.arange(n))
    ax.set_yticklabels([LABELS[m] for m in reversed(methods)])
    ax.set_xscale("log")
    ax.set_xlabel("Wall time (s) — n=513, p=164  [log scale]")
    ax.set_title(title, pad=4, fontweight="bold")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda v, _: f"{v:.0f}s" if v >= 1 else f"{v:.2f}s"
    ))


# ---------------------------------------------------------------------------
# Build full figure
# ---------------------------------------------------------------------------

def build_figure(df: pd.DataFrame) -> plt.Figure:
    syn  = df[df["benchmark"] == "synthetic"]
    diff = df[df["benchmark"] == "diffusion"]

    syn_methods  = [m for m in METHODS_MAIN if m in syn["method"].unique()]
    diff_methods = [m for m in METHODS_MAIN if m in diff["method"].unique()]

    fig = plt.figure(figsize=(18, 12))
    gs  = GridSpec(2, 3, figure=fig,
                   hspace=0.50, wspace=0.40,
                   left=0.06, right=0.97,
                   top=0.91, bottom=0.09)

    # Panel A — radar (polar)
    ax_A = fig.add_subplot(gs[0, 0], projection="polar")
    _radar(ax_A, syn, syn_methods,
           metrics=["aupr", "auroc", "mcc", "f1_opt"],
           metric_labels=["AUPR", "AUROC", "MCC", "F1$_{opt}$"],
           title="A   Synthetic — overall performance")

    # Panel B — MCC per topology
    ax_B = fig.add_subplot(gs[0, 1])
    _violin_topology(ax_B, syn, "mcc", syn_methods,
                     ylabel="MCC", ylim=(-0.05, 1.05),
                     title="B   Synthetic — MCC by topology")

    # Panel C — scalability
    ax_C = fig.add_subplot(gs[0, 2])
    _scalability(ax_C, syn, "aupr", syn_methods,
                 ylabel="AUPR",
                 title="C   Scalability — AUPR vs. problem size")

    # Panel D — diffusion per topology
    ax_D = fig.add_subplot(gs[1, 0])
    _diffusion_topology(ax_D, df, diff_methods,
                        title="D   Network diffusion recovery by topology")

    # Panel E — DREAM5 line
    ax_E = fig.add_subplot(gs[1, 1])
    _dream5_line(ax_E, df, title="E   DREAM5 — AUPR vs. gene-set size")

    # Panel F — wall time
    ax_F = fig.add_subplot(gs[1, 2])
    _wall_time(ax_F, df, title="F   Computational cost (n=513, p=164)")

    # Shared legend
    handles = []
    for m in METHODS_MAIN:
        if m not in syn_methods:
            continue
        lw = 2.2 if m in PIG_METHODS else 1.0
        ec = "#7B0000" if m in PIG_METHODS else PALETTE[m]
        label = LABELS[m]
        handles.append(mpatches.Patch(facecolor=PALETTE[m], edgecolor=ec,
                                      linewidth=lw, label=label))

    fig.legend(handles=handles, loc="upper center", ncol=len(METHODS_MAIN),
               frameon=False, fontsize=9.5, bbox_to_anchor=(0.5, 1.00),
               handlelength=1.5, handleheight=0.95, columnspacing=2.0)

    fig.suptitle("NODIS Benchmark — PIGLasso vs. all methods",
                 y=1.04, fontsize=12, fontweight="bold")

    return fig


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="NODIS benchmark comparison figure")
    parser.add_argument("--out", default=os.path.join(FIGURES_DIR, "benchmark_comparison.pdf"))
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    print("Loading data …")
    df = pd.read_csv(SUMMARY_CSV)
    print(f"  {len(df):,} rows — methods: {sorted(df['method'].unique())}")

    print("Building figure …")
    fig = build_figure(df)

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
