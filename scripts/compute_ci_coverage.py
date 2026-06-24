"""
compute_ci_coverage.py
----------------------
Compute empirical 95% CI coverage for NODIS over true non-zero precision
matrix entries.

Coverage = fraction of true non-zero (i,j) pairs for which the asymptotic
95% CI [omega_hat - 1.96*SE, omega_hat + 1.96*SE] contains the true omega_ij.

Outputs a dict of {config: {topology: coverage}} and prints a summary table.
Run standalone or import compute_coverage().
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from scipy.stats import norm

from nodis.estimators.desparsified import DesparifiedGGM
from nodis.simulate.generator import generate

CONFIGS = [
    dict(n=100,  p=50,  label="n100/p50",  n_over_p=2.0),
    dict(n=237,  p=78,  label="n237/p78",  n_over_p=3.04),
    dict(n=513,  p=164, label="n513/p164", n_over_p=3.13),
]
TOPOLOGIES = ["hub", "scale-free", "cluster", "random"]
N_REPS = 20
Z_CRIT = norm.ppf(0.975)   # 1.96 for 95% CI


def compute_coverage(n: int, p: int, topology: str, n_reps: int = N_REPS) -> float:
    """
    Return empirical 95% CI coverage rate over true non-zero precision entries.
    Averages across n_reps independent replicates.
    """
    covered_all, total_all = 0, 0

    for rep in range(n_reps):
        data = generate(n, p, topology=topology, seed=rep)
        true_omega = data.Omega          # (p, p) true precision matrix

        model = DesparifiedGGM(n_jobs=1)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(data.X)

        omega_hat = model.result_.precision   # (p, p)
        variance  = model.result_.variance    # (p, p) tau2_i * tau2_j

        se = np.sqrt(np.maximum(variance, 0.0) / n)
        lower = omega_hat - Z_CRIT * se
        upper = omega_hat + Z_CRIT * se

        # True adjacency (upper triangle, no diagonal)
        tri = np.triu(np.ones((p, p), dtype=bool), k=1)
        true_adj = (np.abs(true_omega) > 1e-8) & tri

        idx = np.where(true_adj)
        if idx[0].size == 0:
            continue

        true_vals = true_omega[idx]
        lo        = lower[idx]
        hi        = upper[idx]

        covered = np.sum((true_vals >= lo) & (true_vals <= hi))
        covered_all += int(covered)
        total_all   += int(idx[0].size)

    if total_all == 0:
        return float("nan")
    return covered_all / total_all


def main():
    print(f"{'Config':<14} {'Topology':<12} {'Coverage':>10}  ({N_REPS} reps each)")
    print("-" * 42)
    results = {}
    for cfg in CONFIGS:
        n, p, label = cfg["n"], cfg["p"], cfg["label"]
        results[label] = {}
        for topo in TOPOLOGIES:
            cov = compute_coverage(n, p, topo, n_reps=N_REPS)
            results[label][topo] = cov
            print(f"{label:<14} {topo:<12} {cov:>10.3f}")
        # Grand mean across topologies
        vals = [v for v in results[label].values() if not np.isnan(v)]
        gm = np.mean(vals) if vals else float("nan")
        print(f"{label:<14} {'MEAN':<12} {gm:>10.3f}")
        print()
    return results


if __name__ == "__main__":
    main()
