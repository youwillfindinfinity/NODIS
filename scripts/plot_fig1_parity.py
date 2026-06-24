"""
plot_fig1_parity.py
-------------------
Fig 1 — Implementation fidelity: NODIS vs reference B_NW_SL z-scores.
4 panels (one per topology), n=200, p=100, seed=0.
Scatter of |z-score| colored by true edge status.
"""
import sys, os, warnings
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.stats import pearsonr

from nodis.estimators.desparsified import DesparifiedGGM
from nodis.estimators._reference_bnwsl import fit_reference_bnwsl
from nodis.simulate.generator import generate

mpl.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial","DejaVu Sans"],
    "font.size": 9, "axes.labelsize": 10, "axes.titlesize": 10,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

TOPOLOGIES = ["hub", "scale-free", "cluster", "random"]
TOPO_LABELS = {"hub": "Hub", "scale-free": "Scale-free",
               "cluster": "Cluster", "random": "Random (ER)"}
N, P, SEED = 200, 100, 42

COLOR_NULL = "#5B9BD5"   # blue — true zero edges
COLOR_TRUE = "#E86343"   # orange — true non-zero edges

FIG_OUT = os.path.join(os.path.dirname(__file__), "..", "paper", "Fig", "fig1_parity.pdf")


def run_one(topology: str):
    data = generate(N, P, topology=topology, seed=SEED)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = DesparifiedGGM(n_jobs=1)
        model.fit(data.X)

    nodis_z = model.result_.z_scores

    ref = fit_reference_bnwsl(data.X, lambda_scale=1.0, standardise=True)
    ref_z = ref["z_scores"]

    tri = np.triu(np.ones((P, P), dtype=bool), k=1)
    true_adj = data.Omega.astype(bool)

    nodis_v = nodis_z[tri]
    ref_v   = ref_z[tri]
    labels  = true_adj[tri]

    r, _ = pearsonr(nodis_v, ref_v)
    return nodis_v, ref_v, labels, r


def main():
    fig, axes = plt.subplots(2, 2, figsize=(8, 7.5),
                             constrained_layout=True)
    axes = axes.flatten()

    for ax, topo in zip(axes, TOPOLOGIES):
        nodis_v, ref_v, labels, r = run_one(topo)

        # Plot null edges first (background), then true edges (foreground)
        ax.scatter(ref_v[~labels], nodis_v[~labels],
                   c=COLOR_NULL, s=5, alpha=0.35, linewidths=0, rasterized=True,
                   label="True null")
        ax.scatter(ref_v[labels], nodis_v[labels],
                   c=COLOR_TRUE, s=10, alpha=0.80, linewidths=0, rasterized=True,
                   label="True edge")

        # Identity line
        lim = max(np.abs(np.concatenate([ref_v, nodis_v]))) * 1.05
        ax.plot([-lim, lim], [-lim, lim], "k--", lw=0.8, alpha=0.6)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)

        ax.set_title(TOPO_LABELS[topo], pad=4, fontweight="bold")
        ax.set_xlabel("Reference B_NW_SL z-score")
        ax.set_ylabel("NODIS z-score")
        ax.text(0.05, 0.95, f"r = {r:.3f}",
                transform=ax.transAxes, va="top", ha="left",
                fontsize=9, fontweight="bold")

    # Shared legend on first axis
    handles = [
        mpl.patches.Patch(color=COLOR_NULL, label="True null edge"),
        mpl.patches.Patch(color=COLOR_TRUE, label="True non-null edge"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2,
               frameon=False, fontsize=9, bbox_to_anchor=(0.5, 1.01))

    os.makedirs(os.path.dirname(FIG_OUT), exist_ok=True)
    fig.savefig(FIG_OUT, dpi=300, bbox_inches="tight")
    fig.savefig(FIG_OUT.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    print(f"Saved → {FIG_OUT}")
    plt.close(fig)


if __name__ == "__main__":
    main()
