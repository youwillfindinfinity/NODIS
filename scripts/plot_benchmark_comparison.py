"""
plot_benchmark_comparison.py
----------------------------
Nature-grade 6-panel benchmark comparison (double-column, 180 mm wide).

Panels:
  A  Grouped dot plot — grand-mean AUPR / AUROC / MCC / F1_opt per method
  B  MCC per topology — violin + strip jitter
  C  Scalability — AUPR vs. problem size (line with SEM band)
  D  Diffusion recovery — dot plot with SEM by topology
  E  DREAM5 — AUPR vs. gene-set size (line)
  F  Computational cost — log-scale horizontal bar (median + IQR)

Usage:
    cd NODIS/
    python scripts/plot_benchmark_comparison.py [--out figures/benchmark_comparison.pdf]
"""

import argparse
import os
import sys
import warnings

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(__file__))
import plot_style
plot_style.apply()

from plot_style import (
    FULL_W, METHOD_PALETTE, METHOD_LABELS, PIG_METHODS,
    panel_label, save,
)

warnings.filterwarnings("ignore")

METHODS_MAIN = ["desparsified", "glasso", "gglasso", "ssglasso", "piglasso_oracle_n02"]
PALETTE      = METHOD_PALETTE
LABELS       = METHOD_LABELS

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
FIGURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")
SUMMARY_CSV = os.path.join(RESULTS_DIR, "metrics_summary.csv")

ZO_PIG  = 5
ZO_BASE = 2
CONFIGS      = ["n100p50", "n237p78", "n513p164"]
CONFIG_XLABS = ["n=100\np=50", "n=237\np=78", "n=513\np=164"]


def _grouped_dot(ax, data, methods, metrics, metric_labels, title):
    """Panel A: Cleveland-style grouped dot plot for multi-metric comparison."""
    small3 = data[(data["benchmark"] == "synthetic") & (data["config"].isin(CONFIGS))]
    n_met  = len(metrics)
    n_meth = len(methods)
    spacing = 1.0
    group_w = 0.7

    offsets = np.linspace(-group_w / 2, group_w / 2, n_meth)

    for mi, m in enumerate(metrics):
        base_y = mi * spacing
        ax.axhline(base_y, color="#e8e8e8", linewidth=0.5, zorder=0)
        for mti, mth in enumerate(methods):
            vals = small3.loc[small3["method"] == mth, m].dropna()
            if vals.empty:
                continue
            mu, sem = vals.mean(), vals.sem()
            y = base_y + offsets[mti]
            color = PALETTE.get(mth, "#888888")
            lw = 1.5 if mth in PIG_METHODS else 0.8
            zo = ZO_PIG if mth in PIG_METHODS else ZO_BASE
            ax.plot([0, mu], [y, y], color=color, lw=lw * 0.5, zorder=zo - 1,
                    solid_capstyle="round", alpha=0.35)
            ax.errorbar(mu, y, xerr=sem, fmt="none", color=color,
                        elinewidth=lw * 0.7, capsize=1.5, capthick=0.5, zorder=zo)
            ms = 5.5 if mth in PIG_METHODS else 3.5
            mk = "D" if mth in PIG_METHODS else "o"
            ax.scatter(mu, y, color=color, s=ms ** 2, marker=mk,
                       edgecolors="white", linewidths=0.4, zorder=zo + 1)
            ax.text(mu + 0.015, y, f"{mu:.2f}", va="center", ha="left",
                    fontsize=5.0, color=color)

    ax.set_yticks([i * spacing for i in range(n_met)])
    ax.set_yticklabels(metric_labels, fontsize=7)
    ax.set_xlim(0, 1.18)
    ax.set_xlabel("Score (0 – 1)")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.1f}"))
    ax.set_title(title, pad=4, fontweight="bold")
    ax.invert_yaxis()


def _violin_topology(ax, data, metric, methods, ylabel, title, ylim=None):
    """Panel B: violin + jitter strip plot per topology."""
    small3  = data[(data["benchmark"] == "synthetic") & (data["config"].isin(CONFIGS))]
    topos   = ["cluster", "hub", "random", "scale-free"]
    n_meth  = len(methods)
    gw, bw  = 0.88, 0.88 / n_meth
    offsets = np.linspace(-gw / 2 + bw / 2, gw / 2 - bw / 2, n_meth)

    rng = np.random.default_rng(42)
    for ti, topo in enumerate(topos):
        for mi, m in enumerate(methods):
            vals = small3.loc[
                (small3["method"] == m) & (small3["topology"] == topo), metric
            ].dropna().values
            if len(vals) < 4:
                continue
            xp, color = ti + offsets[mi], PALETTE.get(m, "#888888")
            zo = ZO_PIG if m in PIG_METHODS else ZO_BASE
            vp = ax.violinplot(vals, positions=[xp], widths=bw * 0.82,
                               showmedians=False, showextrema=False)
            for body in vp["bodies"]:
                body.set_facecolor(color)
                body.set_alpha(0.55 if m in PIG_METHODS else 0.30)
                body.set_edgecolor(color); body.set_linewidth(0.4); body.set_zorder(zo)
                for path in body.get_paths():
                    path.vertices[:, 1] = np.clip(path.vertices[:, 1], 0.0, 1.0)
            # jitter strip
            jitter = rng.uniform(-bw * 0.20, bw * 0.20, len(vals))
            ax.scatter(xp + jitter, vals, color=color, s=2.5, alpha=0.30,
                       edgecolors="none", zorder=zo + 1)
            # IQR bar + median dot
            q25, med, q75 = np.percentile(vals, [25, 50, 75])
            ax.vlines(xp, q25, q75, color=color,
                      linewidth=1.8 if m in PIG_METHODS else 1.0,
                      zorder=zo + 2, capstyle="round")
            ax.scatter(xp, med, color="white", s=9,
                       edgecolors=color, linewidths=0.8, zorder=zo + 3)

    ax.set_xticks(np.arange(len(topos)))
    ax.set_xticklabels(topos)
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=4, fontweight="bold")
    if ylim:
        ax.set_ylim(*ylim)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.2f}"))
    ax.axhline(0, color="#cccccc", linewidth=0.4, linestyle=":")


def _scalability(ax, data, metric, methods, ylabel, title):
    small3 = data[(data["benchmark"] == "synthetic") & (data["config"].isin(CONFIGS))]
    x = np.arange(len(CONFIGS))
    for m in methods:
        mus, sems = [], []
        for cfg in CONFIGS:
            vals = small3.loc[
                (small3["method"] == m) & (small3["config"] == cfg), metric
            ].dropna()
            mus.append(vals.mean() if len(vals) else np.nan)
            sems.append(vals.sem()  if len(vals) else 0.0)
        mu, sem = np.array(mus), np.array(sems)
        color = PALETTE.get(m, "#888888")
        lw, ms = (1.5, 5.5) if m in PIG_METHODS else (0.9, 3.0)
        mk = "D" if m in PIG_METHODS else "o"
        zo = ZO_PIG if m in PIG_METHODS else ZO_BASE
        ax.plot(x, mu, color=color, lw=lw, marker=mk, ms=ms, zorder=zo,
                label=LABELS.get(m, m), solid_capstyle="round")
        ax.fill_between(x, mu - sem, mu + sem, color=color, alpha=0.12, zorder=zo - 1)
    ax.set_xticks(x)
    ax.set_xticklabels(CONFIG_XLABS)
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=4, fontweight="bold")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.legend(loc="lower right", fontsize=6.5, handlelength=1.2, labelspacing=0.3)


def _diffusion_dot(ax, data, methods, title):
    """Panel D: dot plot (mean ± SEM) for diffusion recovery by topology."""
    diff    = data[(data["benchmark"] == "diffusion") & (data["config"].isin(CONFIGS))]
    topos   = ["cluster", "hub", "random", "scale-free"]
    n_meth  = len(methods)
    gw, bw  = 0.80, 0.80 / n_meth
    offsets = np.linspace(-gw / 2 + bw / 2, gw / 2 - bw / 2, n_meth)

    for ti, topo in enumerate(topos):
        for mi, m in enumerate(methods):
            vals = diff.loc[
                (diff["method"] == m) & (diff["topology"] == topo),
                "diffusion_spearman_norm"
            ].dropna()
            if vals.empty:
                continue
            mu, sem = vals.mean(), vals.sem()
            xp, color = ti + offsets[mi], PALETTE.get(m, "#888888")
            zo = ZO_PIG if m in PIG_METHODS else ZO_BASE
            ms = 5.0 if m in PIG_METHODS else 3.2
            mk = "D" if m in PIG_METHODS else "o"
            ax.plot([xp, xp], [0, mu], color=color, lw=0.5, alpha=0.30, zorder=zo - 1)
            ax.errorbar(xp, mu, yerr=sem, fmt="none", color=color,
                        elinewidth=0.8, capsize=1.5, capthick=0.5, zorder=zo)
            ax.scatter(xp, mu, color=color, s=ms ** 2, marker=mk,
                       edgecolors="white", linewidths=0.4, zorder=zo + 1)

    ax.set_xticks(np.arange(len(topos)))
    ax.set_xticklabels(topos)
    ax.axhline(0, color="#999999", linewidth=0.4, linestyle="--", alpha=0.7)
    ax.set_ylabel("Normalised Spearman ρ")
    ax.set_title(title, pad=4, fontweight="bold")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.2f}"))


def _dream5_line(ax, data, title):
    d5 = data[(data["benchmark"] == "dream5") & (data["network"] == 1)].copy()
    if d5.empty:
        ax.text(0.5, 0.5, "DREAM5 data\nnot available",
                ha="center", va="center", transform=ax.transAxes, color="#888888")
        ax.set_title(title, pad=4, fontweight="bold")
        return
    d5 = d5[d5["method"] != "piglasso_string"]
    disp = ["desparsified", "glasso", "gglasso", "piglasso_corr"]
    ps, x = [200, 500, 1000], np.arange(3)
    for m in disp:
        sub  = d5[d5["method"] == m]
        mus  = [sub.loc[sub["p"] == p, "aupr"].mean() for p in ps]
        color = PALETTE.get(m, "#888888")
        lw, ms = (1.5, 5.5) if m in PIG_METHODS else (0.9, 3.0)
        mk = "D" if m in PIG_METHODS else "o"
        zo = ZO_PIG if m in PIG_METHODS else ZO_BASE
        ax.plot(x, mus, color=color, lw=lw, marker=mk, ms=ms, zorder=zo,
                label=LABELS.get(m, m))
    coverage = [0.408, 0.133, 0.087]
    ax2 = ax.twinx()
    ax2.fill_between(x, 0, coverage, color="#e0e0e0", alpha=0.5)
    ax2.set_ylim(0, 1.0)
    ax2.set_yticks([0, 0.25, 0.5])
    ax2.set_yticklabels(["0%", "25%", "50%"], color="#888888", size=6)
    ax2.set_ylabel("Coverage", color="#888888", size=6.5)
    ax2.tick_params(colors="#888888", width=0.4)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_color("#cccccc")
    ax2.spines["right"].set_linewidth(0.4)
    ax.set_xticks(x); ax.set_xticklabels(["p=200", "p=500", "p=1000"])
    ax.set_ylabel("AUPR"); ax.set_ylim(0.0, 0.25)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.2f}"))
    ax.set_title(title, pad=4, fontweight="bold")
    ax.legend(loc="upper right", fontsize=6, handlelength=1.2)


def _wall_time(ax, data, title):
    """Panel F: horizontal bar (median + IQR) on log scale."""
    syn513  = data[(data["benchmark"] == "synthetic") & (data["config"] == "n513p164")]
    methods = [m for m in METHODS_MAIN if m in syn513["method"].unique()]

    for i, m in enumerate(reversed(methods)):
        vals = syn513.loc[syn513["method"] == m, "wall_seconds"].dropna()
        if vals.empty:
            continue
        med   = vals.median()
        p25, p75 = vals.quantile(0.25), vals.quantile(0.75)
        color = PALETTE.get(m, "#888888")
        zo = ZO_PIG if m in PIG_METHODS else ZO_BASE
        ax.barh(i, med, color=color, edgecolor="none",
                zorder=zo, height=0.52, alpha=0.85)
        ax.errorbar(med, i, xerr=[[med - p25], [p75 - med]],
                    fmt="none", color="#333333", capsize=2.0,
                    capthick=0.6, linewidth=0.6, zorder=zo + 1)
        ax.text(med * 1.25, i, f"{med:.0f} s", va="center", ha="left",
                fontsize=6, color="#222222")
    ax.set_yticks(np.arange(len(methods)))
    ax.set_yticklabels([LABELS.get(m, m) for m in reversed(methods)])
    ax.set_xscale("log")
    ax.set_xlabel("Wall time (s)  [log scale]")
    ax.set_title(title, pad=4, fontweight="bold")
    ax.xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda v, _: f"{v:.0f}s" if v >= 1 else f"{v:.2f}s")
    )
    ax.axvline(60, color="#cccccc", linewidth=0.4, linestyle=":", zorder=0)
    ax.text(63, len(methods) - 0.5, "1 min", color="#aaaaaa", fontsize=5.5, va="top")


def build_figure(df: pd.DataFrame) -> plt.Figure:
    syn  = df[df["benchmark"] == "synthetic"]
    diff = df[df["benchmark"] == "diffusion"]
    syn_methods  = [m for m in METHODS_MAIN if m in syn["method"].unique()]
    diff_methods = [m for m in METHODS_MAIN if m in diff["method"].unique()]

    fig = plt.figure(figsize=(FULL_W, 5.8))
    gs  = GridSpec(2, 3, figure=fig,
                   hspace=0.65, wspace=0.48,
                   left=0.08, right=0.97,
                   top=0.90, bottom=0.10)

    ax_A = fig.add_subplot(gs[0, 0])
    ax_B = fig.add_subplot(gs[0, 1])
    ax_C = fig.add_subplot(gs[0, 2])
    ax_D = fig.add_subplot(gs[1, 0])
    ax_E = fig.add_subplot(gs[1, 1])
    ax_F = fig.add_subplot(gs[1, 2])

    _grouped_dot(ax_A, syn, syn_methods,
                 ["aupr", "auroc", "mcc", "f1_opt"],
                 ["AUPR", "AUROC", "MCC", "F1$_{opt}$"],
                 "Overall performance")
    _violin_topology(ax_B, syn, "mcc", syn_methods,
                     "MCC", "MCC by topology", (-0.05, 1.05))
    _scalability(ax_C, syn, "aupr", syn_methods, "AUPR", "Scalability")
    _diffusion_dot(ax_D, df, diff_methods, "Diffusion recovery")
    _dream5_line(ax_E, df, "DREAM5 — AUPR")
    _wall_time(ax_F, df, "Computational cost")

    for ax, ltr in [(ax_A, "A"), (ax_B, "B"), (ax_C, "C"),
                    (ax_D, "D"), (ax_E, "E"), (ax_F, "F")]:
        panel_label(ax, ltr)

    handles = [
        mpatches.Patch(facecolor=PALETTE.get(m, "#888"), edgecolor="none",
                       label=LABELS.get(m, m))
        for m in METHODS_MAIN if m in syn_methods
    ]
    fig.legend(handles=handles, loc="upper center", ncol=len(handles),
               frameon=False, fontsize=7, bbox_to_anchor=(0.5, 1.00),
               handlelength=1.2, handleheight=0.8, columnspacing=1.5)
    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=os.path.join(FIGURES_DIR, "benchmark_comparison.pdf"))
    parser.add_argument("--dpi", type=int, default=600)
    args = parser.parse_args()
    print("Loading data …")
    df = pd.read_csv(SUMMARY_CSV)
    print(f"  {len(df):,} rows — methods: {sorted(df['method'].unique())}")
    print("Building figure …")
    save(build_figure(df), args.out, args.dpi)


if __name__ == "__main__":
    main()
