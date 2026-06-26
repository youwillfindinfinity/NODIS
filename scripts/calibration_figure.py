"""
scripts/calibration_figure.py
Null-calibration validation for NODIS B_NW_SL.

Produces a 3-panel Figure S_calib:
  Panel A  QQ plot of upper-triangle Z_ij under Θ=I vs N(0,1) + 95% KS envelope
  Panel B  Empirical per-edge FPR vs nominal α across 100 null replicates
  Panel C  Boxplot of realised FDR under hub topology (n=200, p=50, 100 reps)

Run from the repo root:
    python3 scripts/calibration_figure.py [--seed 42] [--n-reps 100]

Outputs:
    paper/Fig/calibration.pdf  (+ .png)
"""

import argparse
import sys
import os
import numpy as np
from scipy.stats import norm
from scipy.stats import ks_1samp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from nodis.simulate.generator import generate
from nodis.estimators.desparsified import DesparifiedGGM
from nodis.inference.fdr import fdr_control

# ---------------------------------------------------------------------------
# Plot style
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
try:
    import plot_style
    plot_style.apply()
except ImportError:
    pass

BLUE    = "#0072B2"
ORANGE  = "#E69F00"
GREEN   = "#009E73"
GREY    = "#999999"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def fit_model(X: np.ndarray):
    """Return the GGMInferenceResult (stored in fitted.result_)."""
    model = DesparifiedGGM(lambda_scale=1.0, n_jobs=-1)
    model.fit(X)
    return model.result_


def upper_triangle_zscores(result) -> np.ndarray:
    Z = np.asarray(result.z_scores)
    idx = np.triu_indices(Z.shape[0], k=1)
    return Z[idx]


def upper_triangle_pvalues(result) -> np.ndarray:
    P = np.asarray(result.p_values)
    idx = np.triu_indices(P.shape[0], k=1)
    return P[idx]


def null_precision(p: int) -> np.ndarray:
    return np.eye(p)


def null_covariance(p: int) -> np.ndarray:
    return np.eye(p)


# ---------------------------------------------------------------------------
# Panel A: QQ plot under global null (Θ = I)
# ---------------------------------------------------------------------------

def panel_a_data(n: int = 500, p: int = 100, seed: int = 0) -> np.ndarray:
    """Generate one null dataset and return upper-triangle z-scores."""
    rng = np.random.default_rng(seed)
    X = rng.multivariate_normal(np.zeros(p), null_covariance(p), size=n)
    result = fit_model(X)
    return upper_triangle_zscores(result)


# ---------------------------------------------------------------------------
# Panel B: empirical FPR vs nominal α across 100 null replicates
# ---------------------------------------------------------------------------

def panel_b_data(
    n: int = 500,
    p: int = 100,
    n_reps: int = 100,
    alphas: tuple = (0.01, 0.02, 0.05, 0.10),
    base_seed: int = 100,
) -> dict:
    """
    Returns dict: alpha -> array of per-edge FPR across replicates.
    Per-edge FPR = fraction of null pairs declared significant at raw p < alpha
    (no FDR correction, direct per-edge type-I error).
    """
    rng = np.random.default_rng(base_seed)
    results_by_alpha = {a: [] for a in alphas}

    for rep in range(n_reps):
        seed = int(rng.integers(0, 2**31))
        X = np.random.default_rng(seed).multivariate_normal(
            np.zeros(p), null_covariance(p), size=n
        )
        result = fit_model(X)
        pvals = upper_triangle_pvalues(result)
        for a in alphas:
            fpr = np.mean(pvals < a)
            results_by_alpha[a].append(fpr)

    return {a: np.array(v) for a, v in results_by_alpha.items()}


# ---------------------------------------------------------------------------
# Panel C: realised FDR under hub topology across replicates
# ---------------------------------------------------------------------------

def panel_c_data(
    n: int = 200,
    p: int = 50,
    n_reps: int = 100,
    alpha: float = 0.05,
    base_seed: int = 200,
) -> np.ndarray:
    """
    Returns array of realised FDR = FP/(FP+TP) at BH alpha=0.05 per replicate.
    """
    rng = np.random.default_rng(base_seed)
    fdrs = []

    for rep in range(n_reps):
        seed = int(rng.integers(0, 2**31))
        data = generate(topology="hub", n=n, p=p, seed=seed)
        X = data.X
        true_adj = (data.Omega != 0).astype(int)
        np.fill_diagonal(true_adj, 0)

        result = fit_model(X)
        pvals = np.asarray(result.p_values)
        adj_pred = fdr_control(pvals, alpha=alpha, method="BH")

        idx = np.triu_indices(p, k=1)
        tp = np.sum((adj_pred[idx] == 1) & (true_adj[idx] == 1))
        fp = np.sum((adj_pred[idx] == 1) & (true_adj[idx] == 0))
        total_pos = tp + fp
        fdr_val = fp / total_pos if total_pos > 0 else 0.0
        fdrs.append(fdr_val)

    return np.array(fdrs)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def make_figure(z_scores, fpr_data, fdr_vals, out_path: str) -> dict:
    alphas = sorted(fpr_data.keys())

    fig, axes = plt.subplots(1, 3, figsize=(7.08, 2.6))

    # --- Panel A: QQ plot ---
    ax = axes[0]
    zs_sorted = np.sort(z_scores)
    n_pts = len(zs_sorted)
    quantiles = norm.ppf((np.arange(1, n_pts + 1) - 0.375) / (n_pts + 0.25))

    ax.scatter(quantiles, zs_sorted, s=0.5, color=BLUE, alpha=0.3, linewidths=0,
               rasterized=True)

    # KS 95% confidence envelope (pointwise ±1.36/√n)
    ks_bound = 1.36 / np.sqrt(n_pts)
    ax.fill_between(quantiles,
                    quantiles - ks_bound / norm.pdf(quantiles).clip(0.001),
                    quantiles + ks_bound / norm.pdf(quantiles).clip(0.001),
                    color=ORANGE, alpha=0.2, label="95% KS envelope")
    ax.plot(quantiles, quantiles, "k--", lw=0.8, label="y = x")

    ks_stat, ks_pval = ks_1samp(z_scores, norm.cdf)
    ax.set_xlabel("Theoretical N(0,1) quantile")
    ax.set_ylabel("Empirical z-score quantile")
    ax.set_title(f"A  QQ under Θ=I\n(KS $p$={ks_pval:.3f}, n={len(z_scores):,})",
                 loc="left", fontsize=7)
    ax.legend(fontsize=6, loc="upper left")

    # --- Panel B: FPR vs nominal α ---
    ax = axes[1]
    means = [fpr_data[a].mean() for a in alphas]
    sds   = [fpr_data[a].std()  for a in alphas]
    ax.plot(alphas, alphas, "k--", lw=0.8, label="y = x (ideal)", zorder=1)
    ax.errorbar(alphas, means, yerr=sds, fmt="o", color=BLUE, capsize=3,
                markersize=4, lw=1.0, label="Empirical FPR ± SD", zorder=2)
    ax.set_xlabel("Nominal α")
    ax.set_ylabel("Per-edge FPR")
    ax.set_xlim(0, 0.12)
    ax.set_ylim(0, 0.12)
    ax.set_xticks(alphas)
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.set_title("B  Per-edge FPR calibration\n(100 null replicates, n=500, p=100)",
                 loc="left", fontsize=7)
    ax.legend(fontsize=6)

    # --- Panel C: realised FDR boxplot ---
    ax = axes[2]
    ax.boxplot(fdr_vals, vert=True, patch_artist=True, widths=0.5,
               medianprops=dict(color="black", lw=1.5),
               boxprops=dict(facecolor=GREEN, alpha=0.5),
               whiskerprops=dict(lw=0.8),
               capprops=dict(lw=0.8),
               flierprops=dict(marker=".", markersize=3, alpha=0.5))
    ax.axhline(0.05, color=ORANGE, lw=1.2, ls="--", label="BH α=0.05 nominal")
    ax.set_xticks([1])
    ax.set_xticklabels(["Hub\nn=200, p=50"])
    ax.set_ylabel("Realised FDR")
    mean_fdr = fdr_vals.mean()
    sd_fdr   = fdr_vals.std()
    ax.set_title(f"C  Realised FDR (BH α=0.05)\n"
                 f"mean={mean_fdr:.3f} ± {sd_fdr:.3f} (100 reps)",
                 loc="left", fontsize=7)
    ax.legend(fontsize=6)

    fig.tight_layout(pad=0.8)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=300)
    print(f"Saved → {out_path}")
    if out_path.endswith(".pdf"):
        png_path = out_path.replace(".pdf", ".png")
        fig.savefig(png_path, dpi=300)
        print(f"Saved → {png_path}")
    plt.close(fig)

    return {
        "ks_stat": float(ks_stat),
        "ks_pval": float(ks_pval),
        "n_zscores": len(z_scores),
        "fpr_mean": {a: float(fpr_data[a].mean()) for a in alphas},
        "fpr_sd":   {a: float(fpr_data[a].std())  for a in alphas},
        "fdr_mean": float(mean_fdr),
        "fdr_sd":   float(sd_fdr),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate null calibration figure S_calib")
    parser.add_argument("--seed",   type=int, default=42,  help="Base random seed")
    parser.add_argument("--n-reps", type=int, default=100, help="Number of replicates for panels B and C")
    parser.add_argument("--out",    type=str,
                        default=os.path.join(os.path.dirname(os.path.dirname(
                            os.path.abspath(__file__))), "paper", "Fig", "calibration.pdf"),
                        help="Output PDF path")
    args = parser.parse_args()

    print("Panel A: fitting model under global null (n=500, p=100)…")
    z_scores = panel_a_data(n=500, p=100, seed=args.seed)
    print(f"  {len(z_scores):,} upper-triangle z-scores extracted")

    print(f"\nPanel B: {args.n_reps} null replicates (n=500, p=100)…")
    fpr_data = panel_b_data(n=500, p=100, n_reps=args.n_reps,
                            alphas=(0.01, 0.02, 0.05, 0.10),
                            base_seed=args.seed + 100)
    for a, vals in sorted(fpr_data.items()):
        print(f"  α={a:.2f}: mean FPR={vals.mean():.4f} ± {vals.std():.4f}")

    print(f"\nPanel C: {args.n_reps} hub-topology replicates (n=200, p=50, BH α=0.05)…")
    fdr_vals = panel_c_data(n=200, p=50, n_reps=args.n_reps, alpha=0.05,
                            base_seed=args.seed + 200)
    print(f"  Mean realised FDR = {fdr_vals.mean():.3f} ± {fdr_vals.std():.3f}")

    print(f"\nRendering figure → {args.out}")
    stats = make_figure(z_scores, fpr_data, fdr_vals, out_path=args.out)

    print("\n--- Summary statistics (for manuscript text) ---")
    print(f"Panel A KS p-value        : {stats['ks_stat']:.4f} (stat), {stats['ks_pval']:.4f} (p-val)")
    print(f"Panel B FPR at α=0.05     : {stats['fpr_mean'][0.05]:.4f} ± {stats['fpr_sd'][0.05]:.4f}")
    print(f"Panel C realised FDR mean : {stats['fdr_mean']:.3f} ± {stats['fdr_sd']:.3f}")


if __name__ == "__main__":
    main()
