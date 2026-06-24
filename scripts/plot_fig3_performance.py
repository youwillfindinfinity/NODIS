"""
plot_fig3_performance.py
------------------------
Fig 3 — NODIS synthetic benchmark performance.
Left:  Mean AUPR vs n/p ratio per topology (±1 SD shading, 50 reps).
Right: Mean MCC  vs n/p ratio per topology (±1 SD shading, 50 reps).
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

TOPOLOGIES = ["hub", "scale-free", "cluster", "random"]
TOPO_LABELS = {"hub":"Hub","scale-free":"Scale-free","cluster":"Cluster","random":"Random (ER)"}
PALETTE = {"hub":"#5B9BD5","scale-free":"#E86343","cluster":"#70AD47","random":"#7B2D8B"}
MARKERS = {"hub":"o","scale-free":"s","cluster":"^","random":"D"}

RESULTS_CSV = os.path.join(os.path.dirname(__file__), "..", "results", "metrics_summary.csv")
FIG_OUT = os.path.join(os.path.dirname(__file__), "..", "paper", "Fig", "fig3_performance.pdf")

CONFIGS_ORDER = ["n100p50", "n237p78", "n513p164", "n1026p328"]
NP_RATIOS = {
    "n100p50": 100/50,
    "n237p78": 237/78,
    "n513p164": 513/164,
    "n1026p328": 1026/328,
}
NP_XLABELS = {
    "n100p50": "2.00\n(n=100,p=50)",
    "n237p78": "3.04\n(n=237,p=78)",
    "n513p164": "3.13\n(n=513,p=164)",
    "n1026p328": "3.13\n(n=1026,p=328)",
}


def main():
    df = pd.read_csv(RESULTS_CSV, low_memory=False)
    nodis = df[
        (df["method"] == "desparsified") &
        (df["benchmark"] == "synthetic") &
        (df["config"].isin(CONFIGS_ORDER))
    ].copy()
    nodis["n_over_p"] = nodis["n"] / nodis["p"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)

    for ax, metric, ylabel, title in [
        (axes[0], "aupr", "AUPR", "A   AUPR vs n/p ratio"),
        (axes[1], "mcc",  "MCC",  "B   MCC vs n/p ratio"),
    ]:
        x = np.arange(len(CONFIGS_ORDER))

        for topo in TOPOLOGIES:
            mus, sds = [], []
            for cfg in CONFIGS_ORDER:
                sub = nodis[(nodis["topology"] == topo) & (nodis["config"] == cfg)]
                vals = sub[metric].dropna()
                mus.append(vals.mean() if len(vals) else np.nan)
                sds.append(vals.std()  if len(vals) else np.nan)

            mu = np.array(mus); sd = np.array(sds)
            color = PALETTE[topo]; marker = MARKERS[topo]
            ax.plot(x, mu, "o-", color=color, lw=1.8, ms=6,
                    marker=marker, markeredgewidth=0.5, markeredgecolor="white",
                    label=TOPO_LABELS[topo], zorder=3)
            ax.fill_between(x, mu - sd, mu + sd, color=color, alpha=0.15, zorder=2)

        ax.set_xticks(x)
        ax.set_xticklabels([NP_XLABELS[c] for c in CONFIGS_ORDER], fontsize=7.5)
        ax.set_xlabel("n/p ratio")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontweight="bold", pad=4)
        ax.set_ylim(-0.02, 1.05)
        ax.axhline(0.5, color="#aaaaaa", lw=0.6, ls=":")

    handles = [
        mpl.lines.Line2D([0], [0], color=PALETTE[t], marker=MARKERS[t], lw=1.5,
                         ms=5, label=TOPO_LABELS[t])
        for t in TOPOLOGIES
    ]
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False,
               fontsize=9, bbox_to_anchor=(0.5, 1.04))

    os.makedirs(os.path.dirname(FIG_OUT), exist_ok=True)
    fig.savefig(FIG_OUT, dpi=300, bbox_inches="tight")
    fig.savefig(FIG_OUT.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    print(f"Saved → {FIG_OUT}")
    plt.close(fig)


if __name__ == "__main__":
    main()
