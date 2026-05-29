"""
plot_diffusion.py
-----------------
Detailed 4-panel diffusion & knockout analysis figure.

Panels:
  A  Heatmap — DiffSp_norm (method × topology), config n513p164
  B  Per-delta-mode DiffSp_norm (box, PIGLasso vs all, n513p164)
  C  Knockout top-10 recall per topology (grouped bar)
  D  Scatter — DiffSp_norm vs AUPR (methods coloured; all configs)

Usage:
    python scripts/plot_diffusion.py
    python scripts/plot_diffusion.py --out figures/diffusion_analysis.pdf
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.gridspec import GridSpec

sys.path.insert(0, os.path.dirname(__file__))
import plot_style
plot_style.apply()

PALETTE = {
    "glasso":       "#4C72B0",
    "desparsified": "#F78154",
    "gglasso":      "#5CAD6E",
    "ssglasso":     "#F2C14E",
    "piglasso":     "#B4436C",
}
LABELS = {
    "desparsified": "Desparsified",
    "glasso":       "GLasso",
    "gglasso":      "GGLasso",
    "ssglasso":     "SSGLasso",
    "piglasso":     "PIGLasso",
}
METHODS = ["desparsified", "glasso", "gglasso", "ssglasso", "piglasso"]
PIG_METHODS = {"ssglasso", "piglasso"}
ZO_PIG = 5
ZO_BASE = 2

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
SUMMARY_CSV = os.path.join(RESULTS_DIR, "metrics_summary.csv")
FIGURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")
SMALL3 = ["n100p50", "n237p78", "n513p164"]
TOPOS  = ["cluster", "hub", "random", "scale-free"]


# ---------------------------------------------------------------------------
# Panel A — DiffSp_norm heatmap (method × topology)
# ---------------------------------------------------------------------------

def _heatmap(ax, diff):
    methods  = [m for m in METHODS if m in diff["method"].unique()]
    small3   = diff[diff["config"] == "n513p164"]
    mat      = np.full((len(methods), len(TOPOS)), np.nan)
    for mi, m in enumerate(methods):
        for ti, t in enumerate(TOPOS):
            vals = small3.loc[
                (small3["method"] == m) & (small3["topology"] == t),
                "diffusion_spearman_norm"
            ].dropna()
            mat[mi, ti] = vals.mean() if len(vals) else np.nan

    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list(
        "nodis", ["#F2F2F2", "#4C72B0", "#4D9078", "#F2C14E", "#F78154", "#B4436C"], N=256)
    im = ax.imshow(mat, cmap=cmap, vmin=0.0, vmax=0.75, aspect="auto", alpha=0.90)
    ax.set_xticks(np.arange(len(TOPOS)))
    ax.set_yticks(np.arange(len(methods)))
    ax.set_xticklabels([t for t in TOPOS])
    ax.set_yticklabels([LABELS[m] for m in methods])

    # Annotate cells
    for mi in range(len(methods)):
        for ti in range(len(TOPOS)):
            v = mat[mi, ti]
            if np.isnan(v):
                continue
            txt_color = "white" if (v < 0.15 or v > 0.55) else "black"
            ax.text(ti, mi, f"{v:.2f}", ha="center", va="center",
                    fontsize=10, color=txt_color, fontweight="normal")

    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Normalised diffusion recovery (Spearman ρ)", size=13)
    cb.ax.tick_params(labelsize=12)

    ax.set_xlabel("Network topology", fontsize=13)
    ax.set_ylabel("Inference method", fontsize=13)
    ax.spines[:].set_visible(False)


# ---------------------------------------------------------------------------
# Panel B — Per-delta-mode box (n513p164)
# ---------------------------------------------------------------------------

def _delta_box(ax, diff):
    d = diff[diff["config"] == "n513p164"].copy()
    methods = [m for m in METHODS if m in d["method"].unique()]
    delta_modes = sorted(d["delta_mode"].dropna().unique())
    n_delta = len(delta_modes)
    n_meth  = len(methods)
    gw      = 0.80
    bw      = gw / n_meth
    offsets = np.linspace(-gw / 2 + bw / 2, gw / 2 - bw / 2, n_meth)

    for di, dm in enumerate(delta_modes):
        for mi, m in enumerate(methods):
            vals = d.loc[
                (d["method"] == m) & (d["delta_mode"] == dm),
                "diffusion_spearman_norm"
            ].dropna().values
            if len(vals) < 2:
                continue
            xp    = di + offsets[mi]
            color = PALETTE[m]
            zo    = ZO_PIG if m in PIG_METHODS else ZO_BASE

            bp = ax.boxplot(vals, positions=[xp], widths=bw * 0.80,
                            patch_artist=True,
                            medianprops=dict(color="white", linewidth=1.5, zorder=zo + 2),
                            whiskerprops=dict(color=color, linewidth=0.7),
                            capprops=dict(color=color, linewidth=0.7),
                            flierprops=dict(marker=".", color=color, ms=3, alpha=0.5))
            for patch in bp["boxes"]:
                patch.set_facecolor(color)
                patch.set_alpha(1.0)
                patch.set_edgecolor(color)
                patch.set_linewidth(0.7)
                patch.set_zorder(zo)

    ax.set_xticks(np.arange(n_delta))
    dm_labels = {"fiedler": "Fiedler", "hub": "Hub", "random": "Random"}
    ax.set_xticklabels([dm_labels.get(d, d.capitalize()) for d in delta_modes])
    ax.set_xlabel("Δ-signal mode")
    ax.set_ylabel("Normalised diffusion recovery (Spearman ρ)")
    ax.axhline(0, color="#999999", linewidth=0.8, linestyle="--", alpha=0.7)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.2f}"))


# ---------------------------------------------------------------------------
# Panel C — Knockout top-10 recall per topology
# ---------------------------------------------------------------------------

def _knockout_bar(ax, diff):
    small3   = diff[diff["config"] == "n513p164"]
    methods  = [m for m in METHODS if m in small3["method"].unique()]
    n_topo   = len(TOPOS)
    n_meth   = len(methods)
    gw       = 0.82
    bw       = gw / n_meth
    offsets  = np.linspace(-gw / 2 + bw / 2, gw / 2 - bw / 2, n_meth)

    for ti, topo in enumerate(TOPOS):
        for mi, m in enumerate(methods):
            vals = small3.loc[
                (small3["method"] == m) & (small3["topology"] == topo),
                "knockout_top10_recall"
            ].dropna()
            if vals.empty:
                continue
            mu  = vals.mean()
            sem = vals.sem()
            xp    = ti + offsets[mi]
            color = PALETTE[m]
            lw    = 0.7
            ec    = "#444444"
            zo    = ZO_PIG if m in PIG_METHODS else ZO_BASE
            ax.bar(xp, mu, bw * 0.86, color=color, edgecolor=ec,
                   linewidth=lw, zorder=zo)
            ax.errorbar(xp, mu, yerr=sem, fmt="none", color="black",
                        capsize=2, capthick=0.7, linewidth=0.7, zorder=zo + 1)

    ax.set_xticks(np.arange(n_topo))
    ax.set_xticklabels([t for t in TOPOS])
    ax.set_xlabel("Network topology")
    ax.set_ylabel("Knockout top-10 recall")
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0%}"))


# ---------------------------------------------------------------------------
# Panel D — DiffSp_norm vs AUPR scatter
# ---------------------------------------------------------------------------

def _diffusion_vs_aupr(ax, df):
    """
    Each point = one (method, config, topology) cell mean.
    PIGLasso points plotted larger with crimson edge.
    x-axis: MCC (uses full confusion matrix, more appropriate since
    diffusion runs on the thresholded binary network).
    """
    diff = df[df["benchmark"] == "diffusion"]
    syn  = df[df["benchmark"] == "synthetic"]
    methods = [m for m in METHODS if m in diff["method"].unique()]

    for m in methods:
        xs, ys = [], []
        for cfg in SMALL3:
            for topo in TOPOS:
                d_val = diff.loc[
                    (diff["method"] == m) & (diff["config"] == cfg) &
                    (diff["topology"] == topo),
                    "diffusion_spearman_norm"
                ].mean()
                a_val = syn.loc[
                    (syn["method"] == m) & (syn["config"] == cfg) &
                    (syn["topology"] == topo),
                    "mcc"
                ].mean()
                if np.isnan(d_val) or np.isnan(a_val):
                    continue
                xs.append(a_val)
                ys.append(d_val)

        color = PALETTE[m]
        zo    = ZO_PIG if m in PIG_METHODS else ZO_BASE
        ax.scatter(xs, ys, color=color, s=35, edgecolors=color, linewidths=0.5,
                   zorder=zo, label=LABELS[m], alpha=0.75)

    ax.axhline(0, color="#999999", linewidth=0.7, linestyle="--", alpha=0.6)
    ax.set_xlabel("MCC (edge recovery)")
    ax.set_ylabel("Normalised diffusion recovery\n(Spearman ρ)")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.2f}"))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.2f}"))
    ax.legend(loc="lower right", frameon=False, fontsize=11)



# ---------------------------------------------------------------------------
# Build figure
# ---------------------------------------------------------------------------

def build_figure(df: pd.DataFrame) -> plt.Figure:
    diff = df[df["benchmark"] == "diffusion"]

    fig = plt.figure(figsize=(16, 11))
    gs  = GridSpec(2, 2, figure=fig,
                   hspace=0.48, wspace=0.38,
                   left=0.07, right=0.97,
                   top=0.93, bottom=0.09)

    ax_A = fig.add_subplot(gs[0, 0])
    ax_B = fig.add_subplot(gs[0, 1])
    ax_C = fig.add_subplot(gs[1, 0])
    ax_D = fig.add_subplot(gs[1, 1])

    _heatmap(ax_A, diff)
    _delta_box(ax_B, diff)
    _knockout_bar(ax_C, diff)
    _diffusion_vs_aupr(ax_D, df)

    # Shared legend
    import matplotlib.patches as mpatches
    handles = []
    methods = [m for m in METHODS if m in diff["method"].unique()]
    for m in methods:
        lw = 2.2 if m in PIG_METHODS else 1.0
        ec = PALETTE[m] if m in PIG_METHODS else PALETTE[m]
        label = LABELS[m]
        handles.append(mpatches.Patch(facecolor=PALETTE[m], edgecolor=ec,
                                      linewidth=lw, label=label))

    fig.legend(handles=handles, loc="upper center", ncol=len(methods),
               frameon=False, fontsize=13, bbox_to_anchor=(0.5, 1.00),
               handlelength=1.5, handleheight=0.95, columnspacing=2.0)

    return fig


# ---------------------------------------------------------------------------
# Shared legend helper
# ---------------------------------------------------------------------------

def _add_method_legend(fig, methods):
    import matplotlib.patches as mpatches
    handles = [
        mpatches.Patch(facecolor=PALETTE[m], edgecolor=PALETTE[m],
                       linewidth=1.0, label=LABELS[m])
        for m in methods
    ]
    fig.legend(handles=handles, loc="upper center", ncol=len(handles),
               frameon=False, fontsize=13, bbox_to_anchor=(0.5, 1.00),
               handlelength=1.5, handleheight=0.95, columnspacing=2.0)


# ---------------------------------------------------------------------------
# Individual panel figures
# ---------------------------------------------------------------------------

def build_panel_a(df: pd.DataFrame) -> plt.Figure:
    diff = df[df["benchmark"] == "diffusion"]
    fig, ax = plt.subplots(figsize=(7, 5))
    fig.subplots_adjust(left=0.14, right=0.88, top=0.95, bottom=0.14)
    _heatmap(ax, diff)
    return fig


def build_panel_b(df: pd.DataFrame) -> plt.Figure:
    diff = df[df["benchmark"] == "diffusion"]
    fig, ax = plt.subplots(figsize=(7, 5))
    fig.subplots_adjust(left=0.14, right=0.97, top=0.80, bottom=0.14)
    _delta_box(ax, diff)
    methods = [m for m in METHODS if m in diff["method"].unique()]
    _add_method_legend(fig, methods)
    return fig


def build_panel_c(df: pd.DataFrame) -> plt.Figure:
    diff = df[df["benchmark"] == "diffusion"]
    fig, ax = plt.subplots(figsize=(7, 5))
    fig.subplots_adjust(left=0.14, right=0.97, top=0.80, bottom=0.14)
    _knockout_bar(ax, diff)
    methods = [m for m in METHODS if m in diff["method"].unique()]
    _add_method_legend(fig, methods)
    return fig


def build_panel_d(df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 5))
    fig.subplots_adjust(left=0.14, right=0.97, top=0.80, bottom=0.14)
    _diffusion_vs_aupr(ax, df)
    # Panel D has its own legend inside; remove it and add shared one above
    ax.get_legend().remove()
    diff = df[df["benchmark"] == "diffusion"]
    methods = [m for m in METHODS if m in diff["method"].unique()]
    _add_method_legend(fig, methods)
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
    parser.add_argument("--out", default=os.path.join(FIGURES_DIR, "diffusion_analysis.pdf"))
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    df = pd.read_csv(SUMMARY_CSV)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    # Combined 4-panel figure
    fig = build_figure(df)
    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight")
    print(f"Saved → {args.out}")
    if args.out.endswith(".pdf"):
        png = args.out.replace(".pdf", ".png")
        fig.savefig(png, dpi=150, bbox_inches="tight")
        print(f"Saved → {png}")
    plt.close(fig)

    # Individual panels
    base = args.out.replace(".pdf", "")
    for label, builder in [("_A", build_panel_a), ("_B", build_panel_b),
                            ("_C", build_panel_c), ("_D", build_panel_d)]:
        _save(builder(df), base + label + ".pdf", args.dpi)


if __name__ == "__main__":
    main()
