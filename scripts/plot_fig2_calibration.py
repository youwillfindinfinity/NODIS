"""
plot_fig2_calibration.py
------------------------
Fig 2 — Asymptotic validity.

Left:   Empirical FPR under all-null model (Theta=I) at 4 BH alpha levels.
        Lines per (n,p) config; dashed y=x reference.
Middle: Empirical FDR by topology at BH alpha=0.05 (from metrics_summary.csv).
Right:  Null-edge 95% CI coverage rate by n/p (should be >= 0.95).
"""
import sys, os, warnings
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

from nodis.estimators.desparsified import DesparifiedGGM
from nodis.inference.fdr import fdr_control as fdr_adjacency
from nodis.simulate.generator import generate
from scipy.stats import norm

mpl.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial","DejaVu Sans"],
    "font.size": 9, "axes.labelsize": 9.5, "axes.titlesize": 10,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

CONFIGS = [
    dict(n=100,  p=50,  label="n=100, p=50"),
    dict(n=237,  p=78,  label="n=237, p=78"),
    dict(n=513,  p=164, label="n=513, p=164"),
]
TOPOLOGIES = ["hub", "scale-free", "cluster", "random"]
TOPO_LABELS = {"hub":"Hub","scale-free":"Scale-free","cluster":"Cluster","random":"Random"}
ALPHA_LEVELS = [0.01, 0.05, 0.10, 0.20]
N_REPS = 20

RESULTS_CSV = os.path.join(os.path.dirname(__file__), "..", "results", "metrics_summary.csv")
FIG_OUT = os.path.join(os.path.dirname(__file__), "..", "paper", "Fig", "fig2_calibration.pdf")

PALETTE_CONF = ["#4477AA", "#EE6677", "#228833"]   # blue, red, green per config
PALETTE_TOPO = {"hub":"#5B9BD5","scale-free":"#E86343","cluster":"#70AD47","random":"#7B2D8B"}
Z_CRIT = norm.ppf(0.975)


def compute_null_fpr(n: int, p: int, alpha_levels: list, n_reps: int) -> np.ndarray:
    """FPR under Theta=I (no edges) at each alpha level."""
    fprs = np.zeros(len(alpha_levels))
    tri = np.triu(np.ones((p, p), dtype=bool), k=1)
    n_pairs = tri.sum()

    for rep in range(n_reps):
        X = np.random.default_rng(1000 + rep).standard_normal((n, p))
        model = DesparifiedGGM(n_jobs=1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X)
        pvals = model.result_.p_values[tri]
        for ai, alpha in enumerate(alpha_levels):
            adj = fdr_adjacency(model.result_.p_values, alpha=alpha, method="BH")
            fprs[ai] += adj[tri].sum() / n_pairs

    return fprs / n_reps


def compute_null_ci_coverage(n: int, p: int, n_reps: int) -> float:
    """Fraction of null-edge 95% CIs containing 0."""
    contained, total = 0, 0
    tri = np.triu(np.ones((p, p), dtype=bool), k=1)
    for rep in range(n_reps):
        data = generate(n, p, topology="hub", seed=rep)
        model = DesparifiedGGM(n_jobs=1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(data.X)
        omega_hat = model.result_.precision
        variance  = model.result_.variance
        se = np.sqrt(np.maximum(variance, 0.0) / n)
        lo = omega_hat - Z_CRIT * se
        hi = omega_hat + Z_CRIT * se
        null_adj = (~data.Omega.astype(bool)) & tri
        idx = np.where(null_adj)
        contained += int(np.sum((lo[idx] <= 0) & (0 <= hi[idx])))
        total += idx[0].size
    return contained / total if total else float("nan")


def main():
    # ---- Panel A: null FPR ----
    print("Computing null FPR …")
    null_fprs = {}
    for cfg in CONFIGS:
        fpr = compute_null_fpr(cfg["n"], cfg["p"], ALPHA_LEVELS, N_REPS)
        null_fprs[cfg["label"]] = fpr
        print(f"  {cfg['label']}: {fpr}")

    # ---- Panel C: CI coverage ----
    print("Computing CI coverage …")
    ci_covs = {}
    for cfg in CONFIGS:
        cov = compute_null_ci_coverage(cfg["n"], cfg["p"], N_REPS)
        ci_covs[cfg["label"]] = cov
        print(f"  {cfg['label']}: {cov:.4f}")

    # ---- Panel B: empirical FDR from metrics CSV ----
    df = pd.read_csv(RESULTS_CSV, low_memory=False)
    nodis = df[(df["method"] == "desparsified") & (df["benchmark"] == "synthetic")].copy()
    nodis["n_over_p"] = nodis["n"] / nodis["p"]
    fdr_data = {}
    for topo in TOPOLOGIES:
        sub = nodis[nodis["topology"] == topo]
        fdr_by_cfg = {}
        for cfg in CONFIGS:
            rows = sub[
                (sub["n"] == cfg["n"]) & (sub["p"] == cfg["p"])
            ]
            if len(rows) == 0:
                continue
            # FDR = fp / (tp + fp) per replicate, then mean
            tp = rows["tp"]; fp = rows["fp"]
            fdr_vals = fp / (tp + fp + 1e-12)
            fdr_by_cfg[cfg["label"]] = fdr_vals.mean()
        fdr_data[topo] = fdr_by_cfg

    # ---- Build figure ----
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), constrained_layout=True)

    # Panel A — Null FPR
    ax = axes[0]
    x = np.array(ALPHA_LEVELS)
    for i, cfg in enumerate(CONFIGS):
        ax.plot(x, null_fprs[cfg["label"]], "o-", color=PALETTE_CONF[i],
                lw=1.6, ms=5, label=cfg["label"])
    ax.plot([0, 0.22], [0, 0.22], "k--", lw=0.8, alpha=0.5, label="y = x")
    ax.set_xlim(0, 0.22)
    ax.set_ylim(-0.002, 0.06)
    ax.set_xlabel("Nominal BH α")
    ax.set_ylabel("Empirical FPR")
    ax.set_title("A   Null-model FPR calibration", fontweight="bold", pad=4)
    ax.legend(frameon=False, fontsize=7.5, handlelength=1.4)
    ax.set_xticks(ALPHA_LEVELS)

    # Panel B — Empirical FDR by topology
    ax = axes[1]
    n_topo = len(TOPOLOGIES)
    n_cfg  = len(CONFIGS)
    gw = 0.80; bw = gw / n_topo
    offsets = np.linspace(-gw/2 + bw/2, gw/2 - bw/2, n_topo)
    for ci, cfg in enumerate(CONFIGS):
        for ti, topo in enumerate(TOPOLOGIES):
            fdr_val = fdr_data.get(topo, {}).get(cfg["label"], np.nan)
            if np.isnan(fdr_val):
                continue
            xp = ci + offsets[ti]
            ax.bar(xp, fdr_val, bw * 0.88, color=PALETTE_TOPO[topo],
                   edgecolor="#555", linewidth=0.5, zorder=2)

    ax.axhline(0.05, color="k", lw=0.9, ls="--", alpha=0.7, label="α = 0.05")
    ax.set_xticks(np.arange(n_cfg))
    ax.set_xticklabels([c["label"] for c in CONFIGS], fontsize=7.5)
    ax.set_ylabel("Empirical FDR")
    ax.set_title("B   Empirical FDR by topology (BH α=0.05)", fontweight="bold", pad=4)
    ax.set_ylim(0, 0.07)
    handles = [mpl.patches.Patch(color=PALETTE_TOPO[t], label=TOPO_LABELS[t])
               for t in TOPOLOGIES]
    ax.legend(handles=handles, frameon=False, fontsize=7.5, handlelength=1.2,
              ncol=2, loc="upper right")

    # Panel C — Null-edge CI coverage
    ax = axes[2]
    cfg_labels = [c["label"] for c in CONFIGS]
    covs = [ci_covs[lb] for lb in cfg_labels]
    bars = ax.bar(np.arange(n_cfg), covs, color=PALETTE_CONF, edgecolor="#555",
                  linewidth=0.5, width=0.55)
    ax.axhline(0.95, color="k", lw=0.9, ls="--", alpha=0.7, label="Nominal 95%")
    ax.set_xticks(np.arange(n_cfg))
    ax.set_xticklabels(cfg_labels, fontsize=7.5)
    ax.set_ylabel("Null-edge CI coverage")
    ax.set_ylim(0.90, 1.005)
    ax.set_title("C   Null-edge 95% CI coverage", fontweight="bold", pad=4)
    ax.legend(frameon=False, fontsize=8)
    for bar, cov in zip(bars, covs):
        ax.text(bar.get_x() + bar.get_width()/2, cov + 0.001, f"{cov:.3f}",
                ha="center", va="bottom", fontsize=8)

    os.makedirs(os.path.dirname(FIG_OUT), exist_ok=True)
    fig.savefig(FIG_OUT, dpi=300, bbox_inches="tight")
    fig.savefig(FIG_OUT.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    print(f"Saved → {FIG_OUT}")
    plt.close(fig)


if __name__ == "__main__":
    main()
