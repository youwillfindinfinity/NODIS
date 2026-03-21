"""
plot_benchmark_comparison.py
----------------------------
Publication-quality benchmark comparison: PIGLasso vs all methods.
Nature-style multi-panel figure.

Usage:
    python scripts/plot_benchmark_comparison.py
    python scripts/plot_benchmark_comparison.py --out figures/benchmark_comparison.pdf
"""

import argparse
import glob
import os
import warnings

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

warnings.filterwarnings("ignore")
matplotlib.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.minor.width": 0.5,
    "ytick.minor.width": 0.5,
    "xtick.major.size": 3.5,
    "ytick.major.size": 3.5,
    "pdf.fonttype": 42,   # embeds fonts in PDF
    "ps.fonttype": 42,
})

# ---------------------------------------------------------------------------
# Colour scheme  (Wong colourblind palette base, PIGLasso vivid red)
# ---------------------------------------------------------------------------
METHODS = ["desparsified", "glasso", "gglasso", "piglasso"]

PALETTE = {
    "desparsified": "#5B9BD5",   # steel blue
    "glasso":       "#70AD47",   # medium green
    "gglasso":      "#FFC000",   # amber
    "piglasso":     "#C00000",   # deep crimson  ← HIGHLIGHT
}
LABELS = {
    "desparsified": "Desparsified",
    "glasso":       "GLasso",
    "gglasso":      "GGLasso",
    "piglasso":     "PIGLasso",
}
PIGLASSO_ZORDER = 5
BASE_ZORDER = 2

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def load_synthetic() -> pd.DataFrame:
    """Load & concatenate all synthetic CSVs (metrics_summary.csv + piglasso)."""
    summary = os.path.join(RESULTS_DIR, "metrics_summary.csv")
    dfs = [pd.read_csv(summary)]

    pig_files = glob.glob(
        os.path.join(RESULTS_DIR, "piglasso", "synthetic", "results_*.csv")
    )
    if pig_files:
        dfs.append(pd.concat([pd.read_csv(f) for f in pig_files], ignore_index=True))

    df = pd.concat(dfs, ignore_index=True)
    df["config_label"] = df["config"].map(
        {"n100p50": "n=100, p=50", "n237p78": "n=237, p=78", "n513p164": "n=513, p=164"}
    )
    return df


def load_dream5() -> pd.DataFrame:
    rows = []
    for method in METHODS:
        path = os.path.join(RESULTS_DIR, method, "dream5")
        for f in glob.glob(os.path.join(path, "*.csv")):
            rows.append(pd.read_csv(f))
    if not rows:
        return pd.DataFrame()
    df = pd.concat(rows, ignore_index=True)
    df["method"] = df["method"].str.lower()
    return df


def load_sergio() -> pd.DataFrame:
    rows = []
    for method in METHODS:
        path = os.path.join(RESULTS_DIR, method, "sergio")
        for f in glob.glob(os.path.join(path, "*.csv")):
            rows.append(pd.read_csv(f))
    if not rows:
        return pd.DataFrame()
    df = pd.concat(rows, ignore_index=True)
    df["method"] = df["method"].str.lower()
    return df


def load_diffusion() -> pd.DataFrame:
    rows = []
    for method in METHODS:
        path = os.path.join(RESULTS_DIR, method, "diffusion")
        files = glob.glob(os.path.join(path, "*.csv"))
        for f in files:
            rows.append(pd.read_csv(f))
    if not rows:
        return pd.DataFrame()
    df = pd.concat(rows, ignore_index=True)
    df["method"] = df["method"].str.lower()
    return df


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _bar_group(ax, data, metric, methods_present, ylabel, title,
               ylim=None, chance_line=None, fmt_pct=True):
    """Grouped bar plot: one bar per method, mean ± SEM."""
    n = len(methods_present)
    x = np.arange(n)
    width = 0.65

    for i, m in enumerate(methods_present):
        vals = data.loc[data["method"] == m, metric].dropna()
        if vals.empty:
            continue
        mu, se = vals.mean(), vals.sem()
        color = PALETTE[m]
        lw = 2.0 if m == "piglasso" else 0.8
        ec = "#7B0000" if m == "piglasso" else "#333333"
        zorder = PIGLASSO_ZORDER if m == "piglasso" else BASE_ZORDER
        bar = ax.bar(x[i], mu, width * 0.7, color=color, edgecolor=ec,
                     linewidth=lw, zorder=zorder)
        ax.errorbar(x[i], mu, yerr=se, fmt="none", color="black",
                    capsize=2.5, capthick=0.8, linewidth=0.8, zorder=zorder + 1)
        if m == "piglasso":
            ax.bar(x[i], mu, width * 0.7, color="none", edgecolor=ec,
                   linewidth=2.2, zorder=zorder + 1)
            ax.text(x[i], mu + se + 0.01 * (ylim[1] - ylim[0] if ylim else 0.05),
                    "★", ha="center", va="bottom", fontsize=9,
                    color="#C00000", zorder=zorder + 2)

    if chance_line is not None:
        ax.axhline(chance_line, color="grey", linestyle="--",
                   linewidth=0.8, zorder=1, alpha=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[m] for m in methods_present],
                       rotation=30, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=4, fontweight="bold")
    if ylim:
        ax.set_ylim(*ylim)
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda v, _: f"{v:.0%}" if fmt_pct else f"{v:.2f}")
    )


def _violin_box(ax, data, metric, methods_present, ylabel, title,
                ylim=None, topology_order=None):
    """
    Split violin + box plot: one violin per method.
    Optionally facet by topology if topology_order provided.
    """
    if topology_order:
        n_topo = len(topology_order)
        n_meth = len(methods_present)
        group_width = 0.9
        bar_width = group_width / n_meth
        offsets = np.linspace(-group_width / 2 + bar_width / 2,
                               group_width / 2 - bar_width / 2, n_meth)

        for ti, topo in enumerate(topology_order):
            for mi, m in enumerate(methods_present):
                vals = data.loc[
                    (data["method"] == m) & (data["topology"] == topo), metric
                ].dropna().values
                if len(vals) < 3:
                    continue
                xpos = ti + offsets[mi]
                color = PALETTE[m]
                lw = 1.8 if m == "piglasso" else 0.7
                zorder = PIGLASSO_ZORDER if m == "piglasso" else BASE_ZORDER

                vp = ax.violinplot(vals, positions=[xpos],
                                   widths=bar_width * 0.9,
                                   showmedians=False, showextrema=False)
                for body in vp["bodies"]:
                    body.set_facecolor(color)
                    body.set_alpha(0.55 if m != "piglasso" else 0.80)
                    body.set_edgecolor(color)
                    body.set_linewidth(lw)
                    body.set_zorder(zorder)
                    # Clip KDE paths to [0, 1] to avoid artefacts
                    paths = body.get_paths()
                    for path in paths:
                        path.vertices[:, 1] = np.clip(path.vertices[:, 1], 0.0, 1.0)

                q25, med, q75 = np.percentile(vals, [25, 50, 75])
                ax.vlines(xpos, q25, q75, color=color, linewidth=lw * 1.5,
                          zorder=zorder + 1)
                ax.scatter(xpos, med, color="white", s=10,
                           edgecolors=color, linewidths=lw,
                           zorder=zorder + 2)

        ax.set_xticks(np.arange(n_topo))
        ax.set_xticklabels([t.replace("-", "\n") for t in topology_order])
        ax.set_xlabel("Topology")

    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=4, fontweight="bold")
    if ylim:
        ax.set_ylim(*ylim)
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda v, _: f"{v:.0%}")
    )


def _line_scalability(ax, data, metric, methods_present, ylabel, title,
                      configs=("n100p50", "n237p78", "n513p164"),
                      config_labels=("n=100\np=50", "n=237\np=78", "n=513\np=164")):
    """Line plot: metric vs. problem size (scalability)."""
    x = np.arange(len(configs))
    for m in methods_present:
        mu_list, se_list = [], []
        for cfg in configs:
            vals = data.loc[
                (data["method"] == m) & (data["config"] == cfg), metric
            ].dropna()
            mu_list.append(vals.mean() if not vals.empty else np.nan)
            se_list.append(vals.sem() if not vals.empty else 0.0)
        mu_arr = np.array(mu_list)
        se_arr = np.array(se_list)
        color = PALETTE[m]
        lw = 2.2 if m == "piglasso" else 1.2
        ms = 7 if m == "piglasso" else 4
        mk = "D" if m == "piglasso" else "o"
        zo = PIGLASSO_ZORDER if m == "piglasso" else BASE_ZORDER
        ax.plot(x, mu_arr, color=color, lw=lw, marker=mk, ms=ms,
                zorder=zo, label=LABELS[m],
                markeredgewidth=1.2 if m == "piglasso" else 0.5,
                markeredgecolor="#7B0000" if m == "piglasso" else color)
        ax.fill_between(x, mu_arr - se_arr, mu_arr + se_arr,
                        color=color, alpha=0.12, zorder=zo - 1)

    ax.set_xticks(x)
    ax.set_xticklabels(list(config_labels))
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=4, fontweight="bold")
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda v, _: f"{v:.0%}")
    )


def _wall_time_bar(ax, data, methods_present, title):
    """Horizontal log-scale bar chart: median wall time."""
    n = len(methods_present)
    y = np.arange(n)
    for i, m in enumerate(reversed(methods_present)):
        vals = data.loc[data["method"] == m, "wall_seconds"].dropna()
        if vals.empty:
            continue
        med = vals.median()
        p25, p75 = vals.quantile(0.25), vals.quantile(0.75)
        color = PALETTE[m]
        lw = 2.0 if m == "piglasso" else 0.8
        ec = "#7B0000" if m == "piglasso" else "#333333"
        zo = PIGLASSO_ZORDER if m == "piglasso" else BASE_ZORDER
        ax.barh(i, med, color=color, edgecolor=ec, linewidth=lw,
                zorder=zo, height=0.6)
        ax.errorbar(med, i, xerr=[[med - p25], [p75 - med]],
                    fmt="none", color="black", capsize=2.5,
                    capthick=0.8, linewidth=0.8, zorder=zo + 1)
        if m == "piglasso":
            ax.barh(i, med, color="none", edgecolor=ec, linewidth=2.2,
                    zorder=zo + 1, height=0.6)
            ax.text(p75 + 0.05 * p75, i, "★", va="center",
                    ha="left", fontsize=9, color="#C00000")

    ax.set_yticks(np.arange(n))
    ax.set_yticklabels([LABELS[m] for m in reversed(methods_present)])
    ax.set_xscale("log")
    ax.set_xlabel("Wall time (s)")
    ax.set_title(title, pad=4, fontweight="bold")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda v, _: f"{v:.0f}s" if v >= 1 else f"{v:.1f}s"
    ))


# ---------------------------------------------------------------------------
# Main figure
# ---------------------------------------------------------------------------

def build_figure(syn: pd.DataFrame, d5: pd.DataFrame,
                 ser: pd.DataFrame, diff: pd.DataFrame):
    """
    6-panel Nature-quality figure.

    A  Synthetic AUPR by topology (violin)
    B  Synthetic AUROC by topology (violin)
    C  Scalability: AUPR vs. problem size (line)
    D  DREAM5 AUPR bar
    E  SERGIO AUPR bar (if data available)
    F  Wall-time benchmark
    """
    fig = plt.figure(figsize=(18, 11))
    gs = GridSpec(2, 3, figure=fig,
                  hspace=0.52, wspace=0.38,
                  left=0.07, right=0.97,
                  top=0.92, bottom=0.10)

    ax_A = fig.add_subplot(gs[0, 0])
    ax_B = fig.add_subplot(gs[0, 1])
    ax_C = fig.add_subplot(gs[0, 2])
    ax_D = fig.add_subplot(gs[1, 0])
    ax_E = fig.add_subplot(gs[1, 1])
    ax_F = fig.add_subplot(gs[1, 2])

    topos = ["cluster", "hub", "random", "scale-free"]
    syn_methods = [m for m in METHODS if m in syn["method"].unique()]
    dream5_methods = [m for m in METHODS if not d5.empty and m in d5["method"].unique()]
    sergio_methods = [m for m in METHODS if not ser.empty and m in ser["method"].unique()]
    diff_methods = [m for m in METHODS if not diff.empty and m in diff["method"].unique()]

    # ------------------------------------------------------------------
    # Panel A: AUPR by topology
    # ------------------------------------------------------------------
    _violin_box(ax_A, syn, "aupr", syn_methods,
                ylabel="AUPR", title="A   Synthetic — AUPR by topology",
                ylim=(0.0, 1.05), topology_order=topos)
    ax_A.axhline(0.0, color="grey", linewidth=0.5, linestyle=":")

    # ------------------------------------------------------------------
    # Panel B: AUROC by topology
    # ------------------------------------------------------------------
    _violin_box(ax_B, syn, "auroc", syn_methods,
                ylabel="AUROC", title="B   Synthetic — AUROC by topology",
                ylim=(0.5, 1.05), topology_order=topos)
    ax_B.axhline(0.5, color="grey", linewidth=0.8, linestyle="--", alpha=0.6,
                 label="Random chance")

    # ------------------------------------------------------------------
    # Panel C: Scalability (AUPR vs problem size)
    # ------------------------------------------------------------------
    _line_scalability(ax_C, syn, "aupr", syn_methods,
                      ylabel="AUPR", title="C   Scalability — AUPR vs. problem size")
    ax_C.legend(loc="lower right", frameon=False, fontsize=7.5)

    # ------------------------------------------------------------------
    # Panel D: DREAM5
    # ------------------------------------------------------------------
    if not d5.empty and dream5_methods:
        _bar_group(ax_D, d5, "aupr", dream5_methods,
                   ylabel="AUPR",
                   title="D   DREAM5 (E. coli in silico)",
                   ylim=(0.0, 0.35), fmt_pct=False)
        ax_D.yaxis.set_major_formatter(mticker.FuncFormatter(
            lambda v, _: f"{v:.2f}"))
        ax_D.set_xlabel("Gene set: p=200, 500, 1000 (mean ± SEM)", fontsize=7)
    else:
        ax_D.text(0.5, 0.5, "DREAM5 data\nnot available",
                  ha="center", va="center", transform=ax_D.transAxes,
                  color="grey", fontsize=9)
        ax_D.set_title("D   DREAM5 (E. coli in silico)", pad=4, fontweight="bold")

    # ------------------------------------------------------------------
    # Panel E: Diffusion recovery (normalized Spearman) — all 4 methods present
    # ------------------------------------------------------------------
    if not diff.empty and diff_methods and "diffusion_spearman_norm" in diff.columns:
        _bar_group(ax_E, diff, "diffusion_spearman_norm", diff_methods,
                   ylabel="Normalised Spearman ρ",
                   title="E   Network diffusion recovery",
                   ylim=(-0.25, 1.05), fmt_pct=False)
        ax_E.yaxis.set_major_formatter(mticker.FuncFormatter(
            lambda v, _: f"{v:.2f}"))
        ax_E.axhline(0.0, color="grey", linestyle="--",
                     linewidth=0.8, alpha=0.6, label="Random baseline")
        ax_E.text(0.98, 0.97, "0 = random,  1 = perfect",
                  transform=ax_E.transAxes, ha="right", va="top",
                  fontsize=6.5, color="grey", style="italic")
    elif not ser.empty and sergio_methods:
        ser_npn = ser[ser["preprocessing"] == "log2_npn"] if "preprocessing" in ser.columns else ser
        if ser_npn.empty:
            ser_npn = ser
        _bar_group(ax_E, ser_npn, "aupr", sergio_methods,
                   ylabel="AUPR",
                   title="E   SERGIO (single-cell RNA-seq sim.)",
                   ylim=(0.0, 0.30), fmt_pct=False)
        ax_E.yaxis.set_major_formatter(mticker.FuncFormatter(
            lambda v, _: f"{v:.2f}"))
    else:
        ax_E.text(0.5, 0.5, "Diffusion / SERGIO\ndata not available",
                  ha="center", va="center", transform=ax_E.transAxes,
                  color="grey", fontsize=9)
        ax_E.set_title("E", pad=4, fontweight="bold")

    # ------------------------------------------------------------------
    # Panel F: Wall time
    # ------------------------------------------------------------------
    _wall_time_bar(ax_F, syn, syn_methods,
                   title="F   Computational cost (synthetic)")

    # ------------------------------------------------------------------
    # Shared legend at top
    # ------------------------------------------------------------------
    legend_handles = []
    for m in METHODS:
        if m not in syn_methods:
            continue
        lw = 2.2 if m == "piglasso" else 1.0
        ec = "#7B0000" if m == "piglasso" else PALETTE[m]
        patch = mpatches.Patch(
            facecolor=PALETTE[m], edgecolor=ec, linewidth=lw,
            label=LABELS[m] + (" ★" if m == "piglasso" else "")
        )
        legend_handles.append(patch)

    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=len(METHODS),
        frameon=False,
        fontsize=9,
        bbox_to_anchor=(0.5, 1.00),
        handlelength=1.4,
        handleheight=0.9,
        columnspacing=1.6,
    )

    # Super-title
    fig.suptitle(
        "NODIS Benchmark — PIGLasso vs. all methods",
        y=1.03, fontsize=11, fontweight="bold", ha="center"
    )

    return fig


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="NODIS benchmark comparison figure")
    parser.add_argument("--out", default="figures/benchmark_comparison.pdf",
                        help="Output file path (.pdf or .png)")
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    print("Loading synthetic data …")
    syn = load_synthetic()
    print(f"  {len(syn):,} rows — methods: {sorted(syn['method'].unique())}")

    print("Loading DREAM5 data …")
    d5 = load_dream5()
    print(f"  {len(d5):,} rows")

    print("Loading SERGIO data …")
    ser = load_sergio()
    print(f"  {len(ser):,} rows")

    print("Loading diffusion data …")
    diff = load_diffusion()
    print(f"  {len(diff):,} rows")

    print("Building figure …")
    fig = build_figure(syn, d5, ser, diff)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight")
    print(f"Saved → {args.out}")

    # Also save PNG alongside PDF for quick preview
    if args.out.endswith(".pdf"):
        png_out = args.out.replace(".pdf", ".png")
        fig.savefig(png_out, dpi=150, bbox_inches="tight")
        print(f"Saved → {png_out}")


if __name__ == "__main__":
    main()
