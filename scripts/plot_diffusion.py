"""
plot_diffusion.py  —  Nature-grade 4-panel diffusion & knockout analysis.

Panels:
  A  Heatmap — DiffSp_norm (method × topology), n513p164
  B  Violin + jitter — per delta-mode DiffSp_norm (n513p164)
  C  Lollipop — Knockout top-10 recall per topology
  D  Scatter — DiffSp_norm vs. MCC with per-method regression lines

Usage:
    python scripts/plot_diffusion.py [--out figures/diffusion_analysis.pdf]
"""

import argparse
import os
import sys

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(__file__))
import plot_style
plot_style.apply()

from plot_style import (
    FULL_W, METHOD_PALETTE, METHOD_LABELS, PIG_METHODS,
    panel_label, save,
)

METHODS = ["desparsified", "glasso", "gglasso", "ssglasso"]
PALETTE = METHOD_PALETTE
LABELS  = METHOD_LABELS
TOPOS   = ["cluster", "hub", "random", "scale-free"]
SMALL3  = ["n100p50", "n237p78", "n513p164"]
ZO_PIG  = 5
ZO_BASE = 2

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
SUMMARY_CSV = os.path.join(RESULTS_DIR, "metrics_summary.csv")
FIGURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")


def _heatmap(ax, diff):
    methods = [m for m in METHODS if m in diff["method"].unique()]
    d513    = diff[diff["config"] == "n513p164"]
    mat     = np.full((len(methods), len(TOPOS)), np.nan)
    for mi, m in enumerate(methods):
        for ti, t in enumerate(TOPOS):
            vals = d513.loc[(d513["method"] == m) & (d513["topology"] == t),
                            "diffusion_spearman_norm"].dropna()
            mat[mi, ti] = vals.mean() if len(vals) else np.nan

    cmap = LinearSegmentedColormap.from_list(
        "nodis_heat", ["#f7fbff", "#6baed6", "#2171b5", "#08306b"], N=256)
    im = ax.imshow(mat, cmap=cmap, vmin=0.0, vmax=0.75, aspect="auto")
    ax.set_xticks(np.arange(len(TOPOS)))
    ax.set_yticks(np.arange(len(methods)))
    ax.set_xticklabels(TOPOS)
    ax.set_yticklabels([LABELS.get(m, m) for m in methods])
    for mi in range(len(methods)):
        for ti in range(len(TOPOS)):
            v = mat[mi, ti]
            if np.isnan(v):
                continue
            txt_color = "white" if v > 0.50 else "#222222"
            ax.text(ti, mi, f"{v:.2f}", ha="center", va="center",
                    fontsize=6.5, color=txt_color)
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Normalised Spearman ρ", size=7)
    cb.ax.tick_params(labelsize=6)
    ax.set_xlabel("Network topology")
    ax.set_ylabel("Method")
    ax.spines[:].set_visible(False)


def _delta_violin(ax, diff):
    """Panel B: violin + jitter per delta-mode."""
    d       = diff[diff["config"] == "n513p164"].copy()
    methods = [m for m in METHODS if m in d["method"].unique()]
    dmodes  = sorted(d["delta_mode"].dropna().unique())
    n_meth  = len(methods)
    gw, bw  = 0.82, 0.82 / n_meth
    offsets = np.linspace(-gw / 2 + bw / 2, gw / 2 - bw / 2, n_meth)
    rng = np.random.default_rng(42)
    dm_lbl = {"fiedler": "Fiedler", "hub": "Hub", "random": "Random"}

    for di, dm in enumerate(dmodes):
        for mi, m in enumerate(methods):
            vals = d.loc[(d["method"] == m) & (d["delta_mode"] == dm),
                         "diffusion_spearman_norm"].dropna().values
            if len(vals) < 2:
                continue
            xp    = di + offsets[mi]
            color = PALETTE.get(m, "#888888")
            zo    = ZO_PIG if m in PIG_METHODS else ZO_BASE

            if len(vals) >= 4:
                vp = ax.violinplot(vals, positions=[xp], widths=bw * 0.78,
                                   showmedians=False, showextrema=False)
                for body in vp["bodies"]:
                    body.set_facecolor(color)
                    body.set_alpha(0.50 if m in PIG_METHODS else 0.28)
                    body.set_edgecolor(color); body.set_linewidth(0.4); body.set_zorder(zo)

            jit = rng.uniform(-bw * 0.16, bw * 0.16, len(vals))
            ax.scatter(xp + jit, vals, color=color, s=2.5, alpha=0.35,
                       edgecolors="none", zorder=zo + 1)
            q25, med, q75 = np.percentile(vals, [25, 50, 75])
            ax.vlines(xp, q25, q75, color=color,
                      linewidth=1.6 if m in PIG_METHODS else 0.9,
                      zorder=zo + 2, capstyle="round")
            ax.scatter(xp, med, color="white", s=8,
                       edgecolors=color, linewidths=0.7, zorder=zo + 3)

    ax.set_xticks(np.arange(len(dmodes)))
    ax.set_xticklabels([dm_lbl.get(d, d.capitalize()) for d in dmodes])
    ax.set_xlabel("Δ-signal mode")
    ax.set_ylabel("Normalised Spearman ρ")
    ax.axhline(0, color="#cccccc", linewidth=0.4, linestyle="--", alpha=0.7)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.2f}"))


def _knockout_lollipop(ax, diff):
    """Panel C: lollipop (dot + stem) for knockout top-10 recall per topology."""
    d513    = diff[diff["config"] == "n513p164"]
    methods = [m for m in METHODS if m in d513["method"].unique()]
    n_meth  = len(methods)
    gw, bw  = 0.82, 0.82 / n_meth
    offsets = np.linspace(-gw / 2 + bw / 2, gw / 2 - bw / 2, n_meth)

    for ti, topo in enumerate(TOPOS):
        for mi, m in enumerate(methods):
            vals = d513.loc[(d513["method"] == m) & (d513["topology"] == topo),
                            "knockout_top10_recall"].dropna()
            if vals.empty:
                continue
            mu, sem = vals.mean(), vals.sem()
            xp, color = ti + offsets[mi], PALETTE.get(m, "#888888")
            zo = ZO_PIG if m in PIG_METHODS else ZO_BASE
            ms = 5.0 if m in PIG_METHODS else 3.2
            mk = "D" if m in PIG_METHODS else "o"
            ax.plot([xp, xp], [0, mu], color=color, lw=0.5, alpha=0.28, zorder=zo - 1)
            ax.errorbar(xp, mu, yerr=sem, fmt="none", color=color,
                        elinewidth=0.8, capsize=1.5, capthick=0.5, zorder=zo)
            ax.scatter(xp, mu, color=color, s=ms ** 2, marker=mk,
                       edgecolors="white", linewidths=0.4, zorder=zo + 1)

    ax.set_xticks(np.arange(len(TOPOS)))
    ax.set_xticklabels(TOPOS)
    ax.set_xlabel("Network topology")
    ax.set_ylabel("Knockout top-10 recall")
    ax.set_ylim(0, 1.10)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.axhline(0, color="#cccccc", linewidth=0.4)


def _diffusion_vs_mcc_scatter(ax, df):
    """Panel D: scatter DiffSp_norm vs. MCC with per-method regression lines."""
    diff    = df[df["benchmark"] == "diffusion"]
    syn     = df[df["benchmark"] == "synthetic"]
    methods = [m for m in METHODS if m in diff["method"].unique()]

    for m in methods:
        xs, ys = [], []
        for cfg in SMALL3:
            for topo in TOPOS:
                d_val = diff.loc[(diff["method"] == m) & (diff["config"] == cfg) &
                                 (diff["topology"] == topo),
                                 "diffusion_spearman_norm"].mean()
                a_val = syn.loc[(syn["method"] == m) & (syn["config"] == cfg) &
                                (syn["topology"] == topo), "mcc"].mean()
                if np.isnan(d_val) or np.isnan(a_val):
                    continue
                xs.append(a_val); ys.append(d_val)
        xs, ys = np.array(xs), np.array(ys)
        color = PALETTE.get(m, "#888888")
        zo    = ZO_PIG if m in PIG_METHODS else ZO_BASE
        ax.scatter(xs, ys, color=color, s=18, edgecolors="white",
                   linewidths=0.3, zorder=zo, label=LABELS.get(m, m), alpha=0.40)
        if len(xs) >= 3:
            fit = np.polyfit(xs, ys, 1)
            xl  = np.linspace(xs.min(), xs.max(), 60)
            ax.plot(xl, np.polyval(fit, xl), color=color, lw=2.0,
                    linestyle="--", alpha=0.55, zorder=zo - 1)

    ax.axhline(0, color="#cccccc", linewidth=0.4, linestyle="--", alpha=0.6)
    ax.set_xlabel("MCC (edge recovery)")
    ax.set_ylabel("Normalised Spearman ρ")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.2f}"))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.2f}"))
    ax.legend(loc="lower right", fontsize=6.5, handlelength=1.2, labelspacing=0.3)


def _method_legend_handles(methods):
    return [mpatches.Patch(facecolor=PALETTE.get(m, "#888"), edgecolor="none",
                           label=LABELS.get(m, m))
            for m in methods]


def build_figure(df):
    diff    = df[df["benchmark"] == "diffusion"]
    methods = [m for m in METHODS if m in diff["method"].unique()]

    fig = plt.figure(figsize=(FULL_W, 5.2))
    gs  = GridSpec(2, 2, figure=fig,
                   hspace=0.58, wspace=0.42,
                   left=0.08, right=0.97,
                   top=0.90, bottom=0.10)

    ax_A = fig.add_subplot(gs[0, 0])
    ax_B = fig.add_subplot(gs[0, 1])
    ax_C = fig.add_subplot(gs[1, 0])
    ax_D = fig.add_subplot(gs[1, 1])

    _heatmap(ax_A, diff)
    _delta_violin(ax_B, diff)
    _knockout_lollipop(ax_C, diff)
    _diffusion_vs_mcc_scatter(ax_D, df)

    for ax, ltr in [(ax_A, "A"), (ax_B, "B"), (ax_C, "C"), (ax_D, "D")]:
        panel_label(ax, ltr)

    fig.legend(handles=_method_legend_handles(methods),
               loc="upper center", ncol=len(methods),
               frameon=False, fontsize=7, bbox_to_anchor=(0.5, 1.00),
               handlelength=1.2, handleheight=0.8, columnspacing=1.5)
    return fig


# Individual panel builders for supplementary use
def build_panel_a(df):
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    fig.subplots_adjust(left=0.16, right=0.88, top=0.95, bottom=0.15)
    _heatmap(ax, df[df["benchmark"] == "diffusion"])
    return fig


def build_panel_b(df):
    diff    = df[df["benchmark"] == "diffusion"]
    methods = [m for m in METHODS if m in diff["method"].unique()]
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    fig.subplots_adjust(left=0.16, right=0.97, top=0.84, bottom=0.15)
    _delta_violin(ax, diff)
    fig.legend(handles=_method_legend_handles(methods), loc="upper center",
               ncol=3, frameon=False, fontsize=6.5, bbox_to_anchor=(0.5, 1.00),
               handlelength=1.2, columnspacing=1.2)
    return fig


def build_panel_c(df):
    diff    = df[df["benchmark"] == "diffusion"]
    methods = [m for m in METHODS if m in diff["method"].unique()]
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    fig.subplots_adjust(left=0.16, right=0.97, top=0.84, bottom=0.15)
    _knockout_lollipop(ax, diff)
    fig.legend(handles=_method_legend_handles(methods), loc="upper center",
               ncol=3, frameon=False, fontsize=6.5, bbox_to_anchor=(0.5, 1.00),
               handlelength=1.2, columnspacing=1.2)
    return fig


def build_panel_d(df):
    diff    = df[df["benchmark"] == "diffusion"]
    methods = [m for m in METHODS if m in diff["method"].unique()]
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    fig.subplots_adjust(left=0.16, right=0.97, top=0.84, bottom=0.15)
    _diffusion_vs_mcc_scatter(ax, df)
    ax.get_legend().remove()
    fig.legend(handles=_method_legend_handles(methods), loc="upper center",
               ncol=3, frameon=False, fontsize=6.5, bbox_to_anchor=(0.5, 1.00),
               handlelength=1.2, columnspacing=1.2)
    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=os.path.join(FIGURES_DIR, "diffusion_analysis.pdf"))
    parser.add_argument("--dpi", type=int, default=600)
    args = parser.parse_args()

    df = pd.read_csv(SUMMARY_CSV)
    save(build_figure(df), args.out, args.dpi)

    base = args.out.replace(".pdf", "")
    for ltr, builder in [("_A", build_panel_a), ("_B", build_panel_b),
                          ("_C", build_panel_c), ("_D", build_panel_d)]:
        save(builder(df), base + ltr + ".pdf", args.dpi)


if __name__ == "__main__":
    main()
