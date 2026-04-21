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
import warnings

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.gridspec import GridSpec

warnings.filterwarnings("ignore")

matplotlib.rcParams.update({
    "font.family":     "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size":       8,
    "axes.labelsize":  9,
    "axes.titlesize":  9.5,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.dpi":      300,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth":  0.8,
    "pdf.fonttype":    42,
    "ps.fonttype":     42,
})

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
SMALL3 = ["n100p50", "n237p78", "n513p164"]
TOPOS  = ["cluster", "hub", "random", "scale-free"]


# ---------------------------------------------------------------------------
# Panel A — DiffSp_norm heatmap (method × topology)
# ---------------------------------------------------------------------------

def _heatmap(ax, diff):
    methods  = [m for m in METHODS if m in diff["method"].unique()]
    small3   = diff[diff["config"].isin(SMALL3)]
    mat      = np.full((len(methods), len(TOPOS)), np.nan)
    for mi, m in enumerate(methods):
        for ti, t in enumerate(TOPOS):
            vals = small3.loc[
                (small3["method"] == m) & (small3["topology"] == t),
                "diffusion_spearman_norm"
            ].dropna()
            mat[mi, ti] = vals.mean() if len(vals) else np.nan

    im = ax.imshow(mat, cmap="RdYlGn", vmin=0.0, vmax=0.75, aspect="auto")
    ax.set_xticks(np.arange(len(TOPOS)))
    ax.set_yticks(np.arange(len(methods)))
    ax.set_xticklabels([t for t in TOPOS], size=8)
    ax.set_yticklabels([LABELS[m] for m in methods], size=8)

    # Annotate cells
    for mi in range(len(methods)):
        for ti in range(len(TOPOS)):
            v = mat[mi, ti]
            if np.isnan(v):
                continue
            is_pig = methods[mi] in PIG_METHODS
            # Bold + star for PIGLasso; mark best per column
            best_in_col = max(mat[m_i, ti] for m_i in range(len(methods))
                              if not np.isnan(mat[m_i, ti]))
            txt_color = "white" if (v < 0.15 or v > 0.55) else "black"
            ax.text(ti, mi, f"{v:.2f}", ha="center", va="center",
                    fontsize=8, color=txt_color,
                    fontweight="bold" if is_pig else "normal")

    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Normalised Spearman ρ", size=8)
    cb.ax.tick_params(labelsize=7)

    # Highlight PIGLasso rows
    for pig_m, pig_color in [("piglasso", "#B4436C"), ("ssglasso", "#F78154")]:
        if pig_m in methods:
            pig_idx = methods.index(pig_m)
            rect = plt.Rectangle((-0.5, pig_idx - 0.5), len(TOPOS), 1,
                                  fill=False, edgecolor=pig_color,
                                  linewidth=2.2, clip_on=False, zorder=5)
            ax.add_patch(rect)

    ax.set_title("A   Diffusion recovery — normalised Spearman ρ\n"
                 "(mean across 3 configs × 3 δ-modes × 50 reps)",
                 pad=6, fontweight="bold")
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
            lw    = 1.8 if m in PIG_METHODS else 0.7
            zo    = ZO_PIG if m in PIG_METHODS else ZO_BASE

            bp = ax.boxplot(vals, positions=[xp], widths=bw * 0.80,
                            patch_artist=True,
                            medianprops=dict(color="white", linewidth=1.5),
                            whiskerprops=dict(color=color, linewidth=lw),
                            capprops=dict(color=color, linewidth=lw),
                            flierprops=dict(marker=".", color=color, ms=3, alpha=0.5))
            for patch in bp["boxes"]:
                patch.set_facecolor(color)
                patch.set_alpha(0.75 if m in PIG_METHODS else 0.45)
                patch.set_edgecolor(color)
                patch.set_linewidth(lw)
                patch.set_zorder(zo)
            if m in PIG_METHODS:
                for patch in bp["boxes"]:
                    patch.set_alpha(0.85)

    ax.set_xticks(np.arange(n_delta))
    dm_labels = {"fiedler": "Fiedler", "hub": "Hub", "random": "Random"}
    ax.set_xticklabels([dm_labels.get(d, d.capitalize()) for d in delta_modes])
    ax.set_xlabel("Δ-signal mode")
    ax.set_ylabel("Normalised Spearman ρ")
    ax.axhline(0, color="#999999", linewidth=0.8, linestyle="--", alpha=0.7)
    ax.set_title("B   DiffSp_norm by δ-mode (n=513, p=164)",
                 pad=4, fontweight="bold")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.2f}"))


# ---------------------------------------------------------------------------
# Panel C — Knockout top-10 recall per topology
# ---------------------------------------------------------------------------

def _knockout_bar(ax, diff):
    small3   = diff[diff["config"].isin(SMALL3)]
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
            lw    = 1.8 if m in PIG_METHODS else 0.7
            ec    = PALETTE[m] if m in PIG_METHODS else "#444444"
            zo    = ZO_PIG if m in PIG_METHODS else ZO_BASE
            ax.bar(xp, mu, bw * 0.86, color=color, edgecolor=ec,
                   linewidth=lw, zorder=zo)
            ax.errorbar(xp, mu, yerr=sem, fmt="none", color="black",
                        capsize=2, capthick=0.7, linewidth=0.7, zorder=zo + 1)
            if m in PIG_METHODS:
                ax.bar(xp, mu, bw * 0.86, color="none", edgecolor=ec,
                       linewidth=2.0, zorder=zo + 1)

    ax.set_xticks(np.arange(n_topo))
    ax.set_xticklabels([t for t in TOPOS])
    ax.set_ylabel("Knockout top-10 recall")
    ax.axhline(10 / 50, color="#aaaaaa", linewidth=0.7, linestyle=":",
               label=f"Random ({10/50:.2f})")
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_title("C   Knockout top-10 recall by topology",
                 pad=4, fontweight="bold")


# ---------------------------------------------------------------------------
# Panel D — DiffSp_norm vs AUPR scatter
# ---------------------------------------------------------------------------

def _diffusion_vs_aupr(ax, df):
    """
    Each point = one (method, config, topology) cell mean.
    PIGLasso points plotted larger with crimson edge.
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
                    "aupr"
                ].mean()
                if np.isnan(d_val) or np.isnan(a_val):
                    continue
                xs.append(a_val)
                ys.append(d_val)

        color = PALETTE[m]
        ms    = 60  if m in PIG_METHODS else 30
        ec    = PALETTE[m] if m in PIG_METHODS else color
        ew    = 1.5 if m in PIG_METHODS else 0.5
        zo    = ZO_PIG if m in PIG_METHODS else ZO_BASE
        ax.scatter(xs, ys, color=color, s=ms, edgecolors=ec, linewidths=ew,
                   zorder=zo, label=LABELS[m],
                   alpha=0.85 if m in PIG_METHODS else 0.65)

    # Annotate PIGLasso advantage region
    ax.axhline(0, color="#999999", linewidth=0.7, linestyle="--", alpha=0.6)
    ax.set_xlabel("AUPR (edge recovery)")
    ax.set_ylabel("Normalised Spearman ρ\n(diffusion recovery)")
    ax.set_title("D   Edge recovery vs. diffusion recovery\n(per topology × config cell mean)",
                 pad=4, fontweight="bold")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.2f}"))
    ax.legend(loc="lower right", frameon=False, fontsize=7.5)

    # Quadrant annotation
    ax.text(0.97, 0.97,
            "High AUPR +\nhigh diffusion",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=6.5, color="grey", style="italic")
    ax.text(0.03, 0.97,
            "Low AUPR +\nhigh diffusion",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=6.5, color="grey", style="italic")


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
               frameon=False, fontsize=9.5, bbox_to_anchor=(0.5, 1.00),
               handlelength=1.5, handleheight=0.95, columnspacing=2.0)

    fig.suptitle("NODIS — Diffusion & Knockout Analysis",
                 y=1.04, fontsize=12, fontweight="bold")
    return fig


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="figures/diffusion_analysis.pdf")
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    df = pd.read_csv(SUMMARY_CSV)
    fig = build_figure(df)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight")
    print(f"Saved → {args.out}")
    if args.out.endswith(".pdf"):
        png = args.out.replace(".pdf", ".png")
        fig.savefig(png, dpi=150, bbox_inches="tight")
        print(f"Saved → {png}")
    plt.close(fig)


if __name__ == "__main__":
    main()
