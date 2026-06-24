"""
plot_fig4_comparison.py
-----------------------
Fig 4 — Method comparison at n=237, p=78.
4 panels (one per topology): grouped bar chart of mean AUPR ± SD.
Methods: NODIS (desparsified), GLasso-CV, GGLasso.
50 replicates per (topology, method) from metrics_summary.csv.
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

TOPOLOGIES  = ["hub", "scale-free", "cluster", "random"]
TOPO_LABELS = {"hub":"Hub","scale-free":"Scale-free","cluster":"Cluster","random":"Random (ER)"}
METHODS     = ["desparsified", "glasso", "gglasso"]
METHOD_LABELS = {"desparsified":"NODIS", "glasso":"GLasso-CV", "gglasso":"GGLasso"}
PALETTE = {"desparsified":"#5B9BD5", "glasso":"#70AD47", "gglasso":"#FFC000"}

RESULTS_CSV = os.path.join(os.path.dirname(__file__), "..", "results", "metrics_summary.csv")
FIG_OUT = os.path.join(os.path.dirname(__file__), "..", "paper", "Fig", "fig4_comparison.pdf")


def main():
    df = pd.read_csv(RESULTS_CSV, low_memory=False)
    sub = df[
        (df["benchmark"] == "synthetic") &
        (df["config"] == "n237p78") &
        (df["method"].isin(METHODS))
    ].copy()

    fig, axes = plt.subplots(1, 4, figsize=(14, 4.2), constrained_layout=True,
                             sharey=True)

    n_methods = len(METHODS)
    bw = 0.22
    offsets = np.linspace(-(n_methods-1)*bw/2, (n_methods-1)*bw/2, n_methods)

    for ax, topo in zip(axes, TOPOLOGIES):
        for mi, method in enumerate(METHODS):
            vals = sub[(sub["topology"] == topo) & (sub["method"] == method)]["aupr"].dropna()
            mu = vals.mean() if len(vals) else np.nan
            sd = vals.std()  if len(vals) else 0.0
            xp = offsets[mi]
            ax.bar(xp, mu, bw*0.88, color=PALETTE[method], edgecolor="#444",
                   linewidth=0.6, zorder=2)
            ax.errorbar(xp, mu, yerr=sd, fmt="none", color="#333",
                        capsize=3, capthick=0.7, linewidth=0.8, zorder=3)
            ax.text(xp, mu + sd + 0.015, f"{mu:.2f}", ha="center", va="bottom",
                    fontsize=7.5, color="#444")

        ax.set_title(TOPO_LABELS[topo], fontweight="bold", pad=4)
        ax.set_xticks(offsets)
        ax.set_xticklabels([METHOD_LABELS[m] for m in METHODS], rotation=20,
                           ha="right", fontsize=7.5)
        ax.set_ylim(0, 1.12)
        ax.axhline(0, color="#bbbbbb", lw=0.5)

    axes[0].set_ylabel("Mean AUPR ± SD  (n=237, p=78, 50 reps)")
    fig.suptitle("Method comparison — AUPR by topology", fontweight="bold",
                 y=1.02, fontsize=11)

    handles = [mpl.patches.Patch(facecolor=PALETTE[m], edgecolor="#444",
                                 linewidth=0.6, label=METHOD_LABELS[m])
               for m in METHODS]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False,
               fontsize=9, bbox_to_anchor=(0.5, 1.00))

    os.makedirs(os.path.dirname(FIG_OUT), exist_ok=True)
    fig.savefig(FIG_OUT, dpi=300, bbox_inches="tight")
    fig.savefig(FIG_OUT.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    print(f"Saved → {FIG_OUT}")
    plt.close(fig)


if __name__ == "__main__":
    main()
