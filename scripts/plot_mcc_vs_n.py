"""
plot_mcc_vs_n.py
-----------------
MCC vs sample size (fixed p=160) for SSGLasso and PIGLasso.
Reads from results/metrics_summary.csv.

Usage:
    cd NODIS/
    python scripts/plot_mcc_vs_n.py
    python scripts/plot_mcc_vs_n.py --out figures/mcc_vs_n.pdf
"""

import argparse
import os
import warnings

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

matplotlib.rcParams.update({
    "font.family":       "sans-serif",
    "font.sans-serif":   ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size":         12,
    "axes.labelsize":    13,
    "axes.titlesize":    14,
    "xtick.labelsize":   12,
    "ytick.labelsize":   12,
    "legend.fontsize":   12,
    "figure.dpi":        300,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.linewidth":    0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.major.size":  3.5,
    "ytick.major.size":  3.5,
    "pdf.fonttype":      42,
    "ps.fonttype":       42,
})

N_VALS   = [100, 300, 500, 700, 900, 1100, 1300, 1500]
P        = 160
CONFIGS  = [f"n{n}p{P}" for n in N_VALS]

METHODS = {
    "ssglasso":            ("SSGLasso", "#F2C14E", "o", "top",    -1),
    "piglasso_oracle_n02": ("PIGLasso", "#B4436C", "o", "bottom", +1),
}
# label_va and label_side: "top"/−1 → label below point, "bottom"/+1 → label above cap

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
SUMMARY_CSV = os.path.join(RESULTS_DIR, "metrics_summary.csv")
FIGURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")


def build_figure(df: pd.DataFrame) -> plt.Figure:
    sub = df[
        (df["benchmark"] == "synthetic") &
        (df["config"].isin(CONFIGS))
    ].copy()

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.subplots_adjust(left=0.10, right=0.97, top=0.88, bottom=0.13)

    for method, (label, color, marker, lbl_va, lbl_sign) in METHODS.items():
        means, sds, ns = [], [], []
        for n in N_VALS:
            config = f"n{n}p{P}"
            vals = sub.loc[
                (sub["method"] == method) & (sub["config"] == config), "mcc"
            ].dropna()
            if len(vals) == 0:
                means.append(np.nan)
                sds.append(np.nan)
            else:
                means.append(vals.mean())
                sds.append(vals.std())
            ns.append(len(vals))

        means = np.array(means)
        sds   = np.array(sds)
        mask  = ~np.isnan(means)

        ax.plot(np.array(N_VALS)[mask], means[mask],
                color=color, marker=marker, markersize=5,
                linewidth=1.8, label=label, zorder=3)
        ax.fill_between(np.array(N_VALS)[mask],
                        (means - sds)[mask], (means + sds)[mask],
                        color=color, alpha=0.15, zorder=2)
        ax.errorbar(np.array(N_VALS)[mask], means[mask], yerr=sds[mask],
                    fmt="none", ecolor=color, elinewidth=0.9,
                    capsize=4, capthick=0.9, zorder=4)

        for n, m, s in zip(np.array(N_VALS)[mask], means[mask], sds[mask]):
            y_txt = (m + s + 0.018) if lbl_sign > 0 else (m - s - 0.018)
            ax.text(n, y_txt, f"{m:.2f}",
                    ha="center", va=lbl_va, fontsize=10,
                    fontweight="bold", color=color)

    ax.axhline(0.5, color="#bbbbbb", linewidth=0.8, linestyle="--",
               label="Random baseline (0.5)", zorder=1)

    ax.set_xlabel("Sample size (n)")
    ax.set_ylabel("MCC")
    ax.set_xlim(0, N_VALS[-1] + 100)
    ax.set_ylim(0.15, 1.05)
    ax.set_xticks(N_VALS)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.1f}"))
    ax.legend(frameon=False, loc="lower right")
    ax.set_title(f"MCC vs sample size  (p = {P}, all topologies)",
                 fontweight="bold", pad=8)

    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=os.path.join(FIGURES_DIR, "mcc_vs_n.pdf"))
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    print("Loading data …")
    df = pd.read_csv(SUMMARY_CSV, low_memory=False)
    configs_found = df.loc[df["config"].isin(CONFIGS), "config"].unique()
    print(f"  n-sweep configs found: {sorted(configs_found)}")

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