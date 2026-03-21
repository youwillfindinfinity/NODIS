#!/usr/bin/env python3
"""
Finite-sample coverage simulation for ensemble vs asymptotic CIs.

Usage:
    python scripts/run_ensemble_ci_coverage.py --p 30 --n-trials 100 --out finalisation/

Outputs:
    finalisation/ensemble_ci_coverage.csv
"""
import argparse
import csv
from pathlib import Path

import numpy as np
from nodis.estimators.desparsified import DesparifiedGGM
from nodis.inference.confidence import ensemble_ci


def _hub_theta(p: int, rng: np.random.Generator) -> np.ndarray:
    Theta = np.eye(p) * 3.0
    n_hubs = max(1, p // 5)
    for h in range(n_hubs):
        for j in range(h + 1, p):
            Theta[h, j] = Theta[j, h] = 0.3
    eigs = np.linalg.eigvalsh(Theta)
    if eigs.min() <= 0:
        Theta += (abs(eigs.min()) + 0.1) * np.eye(p)
    return Theta


def run_coverage(p: int, n: int, n_splits: int, n_trials: int, seed: int, alpha: float):
    rng = np.random.default_rng(seed)
    Theta = _hub_theta(p, rng)
    Sigma = np.linalg.inv(Theta)
    uidx = np.triu_indices(p, k=1)
    true_omega = Theta[uidx]

    asymp_covered = []
    ens_covered = []

    for t in range(n_trials):
        X = rng.multivariate_normal(np.zeros(p), Sigma, size=n)
        # Asymptotic CI
        est = DesparifiedGGM().fit(X)
        lo_a, hi_a = est.confidence_intervals(alpha=alpha)
        covered_a = (lo_a[uidx] <= true_omega) & (true_omega <= hi_a[uidx])
        asymp_covered.append(covered_a.mean())
        # Ensemble CI
        _, lo_e, hi_e = ensemble_ci(X, n_splits=n_splits, alpha=alpha, seed=t)
        covered_e = (lo_e[uidx] <= true_omega) & (true_omega <= hi_e[uidx])
        ens_covered.append(covered_e.mean())

    return float(np.mean(asymp_covered)), float(np.mean(ens_covered))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--p", type=int, default=30)   # rebuttal specifies p=30
    parser.add_argument("--n-splits", type=int, default=25)
    parser.add_argument("--n-trials", type=int, default=100)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default="finalisation")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "ensemble_ci_coverage.csv"

    n_over_p_ratios = [1, 2, 5, 10]
    nominal = 1.0 - args.alpha

    print(f"p={args.p}, K={args.n_splits}, alpha={args.alpha}, trials={args.n_trials}")
    print(f"{'n/p':>5} {'n':>6} {'asymp_cov':>10} {'ens_cov':>10}")

    rows = []
    for ratio in n_over_p_ratios:
        n = ratio * args.p
        ac, ec = run_coverage(
            p=args.p, n=n, n_splits=args.n_splits,
            n_trials=args.n_trials, seed=args.seed, alpha=args.alpha,
        )
        print(f"{ratio:>5} {n:>6} {ac:>10.3f} {ec:>10.3f}  (nominal={nominal:.3f})")
        rows.append({"n_over_p": ratio, "n": n, "p": args.p, "alpha": args.alpha,
                     "n_splits": args.n_splits, "asymptotic_coverage": ac,
                     "ensemble_coverage": ec, "nominal_coverage": nominal})

    with open(out_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nResults saved to {out_file}")


if __name__ == "__main__":
    main()
