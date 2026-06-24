"""
plot_fig5_dream5.py
-------------------
Fig 5 — DREAM5 benchmark results.
Left:  AUPR for NODIS, GLasso-CV, GGLasso at p=200, 500, 1000 (E. coli in vivo).
Right: AUROC same layout.
Data from metrics_summary.csv (benchmark='dream5', network=1).
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial","DejaVu Sans"],
    "font.size": 9, "axes.labelsize": 9.5, "axes.titlesize": 10,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

METHODS = ["desparsified", "glasso", "gglasso"]
METHOD_LABELS = {"desparsified":"NODIS", "glasso":"GLasso-CV", "gglasso":"GGLasso"}
PALETTE = {"desparsified":"#5B9BD5", "glasso":"#70AD47", "gglasso":"#FFC000"}
MARKERS = {"desparsified":"o", "glasso":"s", "gglasso":"^"}
P_VALS = [200, 500, 1000]

RESULTS_CSV = os.path.join(os.path.dirname(__file__), "..", "results", "metrics_summary.csv")
FIG_OUT = os.path.join(os.path.dirname(__file__), "..", "paper", "Fig", "fig5_dream5.pdf")


def main():
    df = pd.read_csv(RESULTS_CSV, low_memory=False)
    d5 = df[(df["benchmark"] == "dream5") & (df["network"] == 1)].copy()
    d5["p"] = d5["p"].astype(float)

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.0), constrained_layout=True)
    x = np.arange(len(P_VALS))

    for ax, metric, ylabel, title in [
        (axes[0], "aupr",  "AUPR",  "A   DREAM5 E. coli — AUPR"),
        (axes[1], "auroc", "AUROC", "B   DREAM5 E. coli — AUROC"),
    ]:
        for method in METHODS:
            sub = d5[d5["method"] == method]
            vals = [sub.loc[sub["p"] == float(p), metric].mean() for p in P_VALS]
            ax.plot(x, vals, "o-", color=PALETTE[method], marker=MARKERS[method],
                    lw=1.8, ms=6, markeredgewidth=0.5, markeredgecolor="white",
                    label=METHOD_LABELS[method])
            for xi, v in zip(x, vals):
                if not np.isnan(v):
                    ax.text(xi, v + 0.005, f"{v:.3f}", ha="center", va="bottom",
                            fontsize=7, color=PALETTE[method])

        # Random baseline
        if metric == "auroc":
            ax.axhline(0.5, color="#aaaaaa", lw=0.7, ls="--", alpha=0.8,
                       label="Random (AUROC=0.5)")

        ax.set_xticks(x)
        ax.set_xticklabels([f"p={p}" for p in P_VALS])
        ax.set_xlabel("Gene-set size (p)")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontweight="bold", pad=4)
        ax.legend(frameon=False, fontsize=8, handlelength=1.4, loc="upper right")

    fig.suptitle(
        "DREAM5 Network 1 (E. coli in vivo) — undirected GGM inference",
        fontweight="bold", fontsize=10, y=1.02
    )

    os.makedirs(os.path.dirname(FIG_OUT), exist_ok=True)
    fig.savefig(FIG_OUT, dpi=300, bbox_inches="tight")
    fig.savefig(FIG_OUT.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    print(f"Saved → {FIG_OUT}")
    plt.close(fig)


if __name__ == "__main__":
    main()
