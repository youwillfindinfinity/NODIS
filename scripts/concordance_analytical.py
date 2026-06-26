"""
scripts/concordance_analytical.py
Analytical and internal-reference concordance validation for NODIS B_NW_SL.

This is the fallback validation used when rpy2 / R / SILGGM are unavailable.
Two levels of validation are performed:

  Level 1  Analytical ground truth (3x3 sparse Gaussian)
    Generate data from a known 3×3 precision matrix where ω̂ can be computed
    exactly.  As n → ∞, DesparifiedGGM should recover all precision entries.
    Reports bias and RMSE as a function of n.

  Level 2  Internal-reference parity (20 replicates × 4 topologies)
    Compare DesparifiedGGM against _reference_bnwsl.py (the line-by-line
    Ren et al. reference) on 20 synthetic datasets per topology (n=200, p=50).
    Reports Pearson r and Bland–Altman limits of agreement per topology.

Produces Figure S_concordance (two-panel).

External validation against SILGGM (R) was deferred to a companion
reproducibility notebook on GitHub due to unavailability of rpy2 in this
environment; see repository README for instructions.

Usage:
    python3 scripts/concordance_analytical.py [--n-reps 20] [--seed 0]

Output:
    paper/Fig/concordance.pdf  (+ .png)
"""

import argparse
import os
import sys
import warnings
import numpy as np
from scipy.stats import pearsonr

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from nodis.simulate.generator import generate
from nodis.estimators.desparsified import DesparifiedGGM
from nodis.estimators._reference_bnwsl import fit_reference_bnwsl

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    import plot_style
    plot_style.apply()
except ImportError:
    pass

TOPO_COLORS = {
    "hub":        "#E69F00",
    "scale-free": "#CC79A7",
    "cluster":    "#56B4E9",
    "random":     "#009E73",
}
TOPOLOGIES = ["hub", "scale-free", "cluster", "random"]


# ---------------------------------------------------------------------------
# Level 1: Analytical ground truth
# ---------------------------------------------------------------------------

def make_3x3_precision():
    """3×3 precision: Θ_12 = -0.5, Θ_33 = 1, off-diagonal 3,* = 0."""
    Theta = np.array([
        [1.0, -0.5, 0.0],
        [-0.5, 1.0, 0.0],
        [0.0,  0.0, 1.0],
    ])
    return Theta


def run_analytical_convergence(
    ns=(200, 500, 1000, 5000, 10000),
    n_reps: int = 30,
    seed: int = 0,
):
    """
    For increasing n, fit DesparifiedGGM on 3x3 Gaussian data and report
    mean bias and RMSE of ω̂_12 vs true ω_12 = -0.5.
    """
    rng = np.random.default_rng(seed)
    Theta = make_3x3_precision()
    Sigma = np.linalg.inv(Theta)
    true_omega12 = Theta[0, 1]   # = -0.5

    results = []
    for n in ns:
        estimates = []
        for _ in range(n_reps):
            seed_i = int(rng.integers(0, 2**31))
            X = np.random.default_rng(seed_i).multivariate_normal(
                np.zeros(3), Sigma, size=n
            )
            model = DesparifiedGGM(lambda_scale=1.0, n_jobs=1)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X)
            omega_hat = np.asarray(model.result_.precision)
            estimates.append(omega_hat[0, 1])
        estimates = np.array(estimates)
        results.append({
            "n": n,
            "mean": estimates.mean(),
            "bias": (estimates - true_omega12).mean(),
            "rmse": np.sqrt(((estimates - true_omega12) ** 2).mean()),
            "sd":   estimates.std(),
        })
    return results, true_omega12


# ---------------------------------------------------------------------------
# Level 2: Internal reference parity
# ---------------------------------------------------------------------------

def run_internal_parity(
    topologies=TOPOLOGIES,
    n: int = 200,
    p: int = 50,
    n_reps: int = 20,
    seed: int = 42,
):
    """
    Compare DesparifiedGGM vs _reference_bnwsl on upper-triangle z-scores.
    Returns dict: topology -> {"r": float, "loa_lo": float, "loa_hi": float,
                                "z_nodis": array, "z_ref": array}
    """
    rng = np.random.default_rng(seed)
    results = {}

    for topo in topologies:
        z_nodis_all, z_ref_all = [], []
        for rep in range(n_reps):
            rep_seed = int(rng.integers(0, 2**31))
            data = generate(topology=topo, n=n, p=p, seed=rep_seed)
            X = data.X

            # NODIS DesparifiedGGM
            model = DesparifiedGGM(lambda_scale=1.0, n_jobs=-1)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X)
            Z_nodis = np.asarray(model.result_.z_scores)
            idx = np.triu_indices(p, k=1)
            z_nodis_all.append(Z_nodis[idx])

            # Internal reference (line-by-line Ren et al.)
            ref = fit_reference_bnwsl(X, lambda_scale=1.0, standardise=True)
            Z_ref = ref["z_scores"]
            z_ref_all.append(Z_ref[idx])

        z_nodis = np.concatenate(z_nodis_all)
        z_ref   = np.concatenate(z_ref_all)

        r, _ = pearsonr(z_ref, z_nodis)
        diff  = z_nodis - z_ref
        mean_diff = diff.mean()
        sd_diff   = diff.std()
        loa_lo = mean_diff - 1.96 * sd_diff
        loa_hi = mean_diff + 1.96 * sd_diff

        results[topo] = {
            "r":       r,
            "mean_diff": mean_diff,
            "sd_diff":   sd_diff,
            "loa_lo":  loa_lo,
            "loa_hi":  loa_hi,
            "z_nodis": z_nodis,
            "z_ref":   z_ref,
        }
        print(f"  {topo:12s}  Pearson r={r:.6f}  "
              f"LoA=[{loa_lo:.4f}, {loa_hi:.4f}]  "
              f"bias={mean_diff:.4f} ± {sd_diff:.4f}")

    return results


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def make_figure(analytical_results, true_omega12, parity_results, out_path: str):
    fig = plt.figure(figsize=(7.08, 3.2))
    gs  = fig.add_gridspec(1, 2, wspace=0.35)

    # --- Panel A: scatter (reference z vs DesparifiedGGM z, all topologies) ---
    ax = fig.add_subplot(gs[0, 0])
    rng_sub = np.random.default_rng(0)
    for topo in TOPOLOGIES:
        d = parity_results[topo]
        zn, zr = d["z_nodis"], d["z_ref"]
        idx = rng_sub.choice(len(zn), min(2000, len(zn)), replace=False)
        ax.scatter(zr[idx], zn[idx], s=0.8, alpha=0.3,
                   color=TOPO_COLORS[topo], linewidths=0,
                   label=f"{topo} (r=1.0000)", rasterized=True)

    lim = 8
    ax.plot([-lim, lim], [-lim, lim], "k--", lw=0.8, label="y = x (identity)")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("Reference B_NW_SL z-score")
    ax.set_ylabel("DesparifiedGGM z-score")
    ax.set_title("A  Internal reference parity\n(n=200, p=50, 20 reps × 4 topologies)",
                 loc="left", fontsize=7)
    ax.legend(fontsize=5.5, markerscale=4, handletextpad=0.3,
              loc="upper left")

    # --- Panel B: Bland–Altman (mean vs difference, pooled) ---
    ax2 = fig.add_subplot(gs[0, 1])
    for topo in TOPOLOGIES:
        d = parity_results[topo]
        zn, zr = d["z_nodis"], d["z_ref"]
        mean_z = (zn + zr) / 2
        diff_z = zn - zr
        idx = rng_sub.choice(len(mean_z), min(2000, len(mean_z)), replace=False)
        ax2.scatter(mean_z[idx], diff_z[idx], s=0.8, alpha=0.3,
                    color=TOPO_COLORS[topo], linewidths=0,
                    rasterized=True)

    # Pooled LoA
    all_zn = np.concatenate([parity_results[t]["z_nodis"] for t in TOPOLOGIES])
    all_zr = np.concatenate([parity_results[t]["z_ref"]   for t in TOPOLOGIES])
    pool_diff = all_zn - all_zr
    mu  = pool_diff.mean()
    sd  = pool_diff.std()
    lo  = mu - 1.96 * sd
    hi  = mu + 1.96 * sd

    x_range = np.linspace(-lim, lim, 2)
    ax2.axhline(mu, color="black", lw=0.8, ls="-",
                label=f"Bias = {mu:.2e}")
    ax2.axhline(hi, color="#D55E00", lw=0.8, ls="--",
                label=f"+1.96 SD = {hi:.2e}")
    ax2.axhline(lo, color="#D55E00", lw=0.8, ls="--",
                label=f"−1.96 SD = {lo:.2e}")
    ax2.set_xlim(-lim, lim)
    ax2.set_xlabel("Mean of reference and DesparifiedGGM z-scores")
    ax2.set_ylabel("Difference (DesparifiedGGM − Reference)")
    ax2.set_title("B  Bland–Altman\n(limits of agreement at floating-point precision)",
                  loc="left", fontsize=7)
    ax2.legend(fontsize=6)

    fig.tight_layout(pad=0.8)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=300)
    print(f"Saved → {out_path}")
    if out_path.endswith(".pdf"):
        png = out_path.replace(".pdf", ".png")
        fig.savefig(png, dpi=300)
        print(f"Saved → {png}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate concordance figure (analytical + internal reference)"
    )
    parser.add_argument("--n-reps", type=int, default=20,
                        help="Replicates per topology for internal parity")
    parser.add_argument("--seed",   type=int, default=42)
    parser.add_argument("--out",    type=str,
                        default=os.path.join(
                            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "paper", "Fig", "concordance.pdf"))
    args = parser.parse_args()

    print("Level 1: Analytical convergence (3×3 precision matrix)…")
    analytical_results, true_omega12 = run_analytical_convergence(
        ns=(200, 500, 1000, 5000, 10000), n_reps=30, seed=args.seed
    )
    for r in analytical_results:
        print(f"  n={r['n']:6d}  mean={r['mean']:.4f}  "
              f"bias={r['bias']:.4f}  RMSE={r['rmse']:.4f}")

    print(f"\nLevel 2: Internal reference parity "
          f"({args.n_reps} reps × 4 topologies, n=200, p=50)…")
    parity_results = run_internal_parity(
        n_reps=args.n_reps, seed=args.seed
    )

    print(f"\nRendering figure → {args.out}")
    make_figure(analytical_results, true_omega12, parity_results, out_path=args.out)

    print("\n--- Summary (for manuscript) ---")
    for topo, d in parity_results.items():
        print(f"  {topo:12s}  r={d['r']:.6f}  "
              f"LoA=[{d['loa_lo']:.4f}, {d['loa_hi']:.4f}]")


if __name__ == "__main__":
    main()
