#!/usr/bin/env python3
"""
Finite-sample coverage simulation for ensemble vs asymptotic CIs — limit tests.

Sweeps six independent axes to expose where each CI method breaks down:
  1. topology   — null / random / hub / scale-free / cluster / dense
  2. n/p ratio  — 0.25 → 50  (spanning underdetermined → well-determined)
  3. p          — 2, 5, 10, 20, 50  (dimension sensitivity)
  4. alpha      — 0.01 / 0.05 / 0.10  (nominal level)
  5. n_splits K — 2, 5, 10, 25, 50, 100  (ensemble size)
  6. subsample_frac — 0.1 / 0.3 / 0.5 / 0.7 / 0.9  (sub-sample size)

Coverage is reported separately for:
  - ALL  edges (omega_ij)
  - NULL edges (omega_ij = 0, i.e. absent from the graph)
  - ACTIVE edges (omega_ij != 0, i.e. present in the graph)

Usage:
    python scripts/run_ensemble_ci_coverage.py --n-trials 100 --out finalisation/
"""
import argparse
import csv
import traceback
import warnings
from pathlib import Path

import networkx as nx
import numpy as np
from scipy.linalg import block_diag
from scipy.stats import norm

from nodis.estimators.desparsified import DesparifiedGGM
from nodis.inference.confidence import ensemble_ci

# ---------------------------------------------------------------------------
# Precision-matrix builders
# ---------------------------------------------------------------------------

def _make_pd(A: np.ndarray, weight: float = 0.3, min_eig: float = 0.1) -> np.ndarray:
    """Turn a binary adjacency A into a PD precision matrix."""
    p = A.shape[0]
    Theta = A.astype(float) * weight
    np.fill_diagonal(Theta, 0.0)
    eigs = np.linalg.eigvalsh(Theta)
    if eigs.min() <= min_eig:
        Theta += np.eye(p) * (abs(eigs.min()) + min_eig + 0.1)
    return Theta


def _build_theta(topology: str, p: int, rng: np.random.Generator) -> np.ndarray:
    """Return a PD precision matrix for the requested topology."""
    if topology == "null":
        # Identity — every off-diagonal entry is exactly 0
        return np.eye(p)

    elif topology == "random":
        # Erdős–Rényi, target ~5% density (at least 1 expected edge for tiny p)
        prob = min(0.4, max(0.05, 3.0 / max(p - 1, 1)))
        upper = rng.random((p, p)) < prob
        A = np.triu(upper.astype(int), k=1)
        A = A + A.T
        np.fill_diagonal(A, 0)

    elif topology == "hub":
        A = np.zeros((p, p), dtype=int)
        n_hubs = max(1, p // 5)
        hub_spacing = max(1, p // n_hubs)
        for h in range(n_hubs):
            hub = min(h * hub_spacing, p - 1)
            candidates = [k for k in range(p) if k != hub]
            n_spokes = max(1, min(p // 5, len(candidates)))
            spokes = rng.choice(candidates, size=n_spokes, replace=False)
            for s in spokes:
                A[hub, s] = A[s, hub] = 1
        np.fill_diagonal(A, 0)

    elif topology == "scale-free":
        m = max(1, int(round(0.05 * p)))
        seed_int = int(rng.integers(0, 2**31))
        G = nx.barabasi_albert_graph(p, m=m, seed=seed_int)
        A = nx.to_numpy_array(G, dtype=int)
        np.fill_diagonal(A, 0)

    elif topology == "cluster":
        cluster_size = max(3, p // 4)
        n_clusters = max(1, p // cluster_size)
        blocks = []
        for _ in range(n_clusters):
            cs = cluster_size
            B = rng.random((cs, cs)) < 0.4
            B = np.triu(B.astype(int), k=1)
            B = B + B.T
            blocks.append(B)
        remainder = p - n_clusters * cluster_size
        if remainder > 0:
            blocks.append(np.zeros((remainder, remainder), dtype=int))
        A = block_diag(*blocks).astype(int)
        np.fill_diagonal(A, 0)

    elif topology == "dense":
        # Fully connected: every pair has an edge
        A = np.ones((p, p), dtype=int)
        np.fill_diagonal(A, 0)

    else:
        raise ValueError(f"Unknown topology '{topology}'")

    return _make_pd(A)


# ---------------------------------------------------------------------------
# Core coverage runner
# ---------------------------------------------------------------------------

def run_coverage(
    topology: str,
    p: int,
    n: int,
    n_splits: int,
    subsample_frac: float,
    n_trials: int,
    alpha: float,
    seed: int,
) -> dict:
    """
    Run n_trials trials and return mean coverage metrics.

    Returns a dict with keys:
        n_active, edge_density,
        asymp_cov_all, asymp_cov_null, asymp_cov_active,
        ens_cov_all,  ens_cov_null,  ens_cov_active,
        error  (None on success, error message on failure)
    """
    result = {
        "n_active": np.nan,
        "edge_density": np.nan,
        "asymp_cov_all": np.nan,
        "asymp_cov_null": np.nan,
        "asymp_cov_active": np.nan,
        "ens_cov_all": np.nan,
        "ens_cov_null": np.nan,
        "ens_cov_active": np.nan,
        "error": None,
    }
    try:
        rng = np.random.default_rng(seed)
        Theta = _build_theta(topology, p, rng)
        Sigma = np.linalg.inv(Theta)

        uidx = np.triu_indices(p, k=1)
        true_omega = Theta[uidx]
        is_active = np.abs(true_omega) > 1e-8

        n_active = int(is_active.sum())
        n_pairs = len(true_omega)
        result["n_active"] = n_active
        result["edge_density"] = n_active / n_pairs if n_pairs > 0 else 0.0

        asymp_all, asymp_null, asymp_active = [], [], []
        ens_all, ens_null, ens_active = [], [], []

        for t in range(n_trials):
            X = rng.multivariate_normal(np.zeros(p), Sigma, size=n)

            # --- Asymptotic CI ---
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                est = DesparifiedGGM().fit(X)
            lo_a, hi_a = est.confidence_intervals(alpha=alpha)
            covered_a = (lo_a[uidx] <= true_omega) & (true_omega <= hi_a[uidx])
            asymp_all.append(covered_a.mean())
            if is_active.any():
                asymp_active.append(covered_a[is_active].mean())
            if (~is_active).any():
                asymp_null.append(covered_a[~is_active].mean())

            # --- Ensemble CI ---
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                _, lo_e, hi_e = ensemble_ci(
                    X,
                    n_splits=n_splits,
                    subsample_frac=subsample_frac,
                    alpha=alpha,
                    seed=t,
                )
            covered_e = (lo_e[uidx] <= true_omega) & (true_omega <= hi_e[uidx])
            ens_all.append(covered_e.mean())
            if is_active.any():
                ens_active.append(covered_e[is_active].mean())
            if (~is_active).any():
                ens_null.append(covered_e[~is_active].mean())

        result.update({
            "asymp_cov_all":    float(np.mean(asymp_all)),
            "asymp_cov_null":   float(np.mean(asymp_null))   if asymp_null   else np.nan,
            "asymp_cov_active": float(np.mean(asymp_active)) if asymp_active else np.nan,
            "ens_cov_all":      float(np.mean(ens_all)),
            "ens_cov_null":     float(np.mean(ens_null))     if ens_null     else np.nan,
            "ens_cov_active":   float(np.mean(ens_active))   if ens_active   else np.nan,
        })
    except Exception as e:  # noqa: BLE001
        result["error"] = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"

    return result


# ---------------------------------------------------------------------------
# Scenario definitions
# ---------------------------------------------------------------------------

def build_scenarios() -> list[dict]:
    """
    Return the full list of scenarios.  Each dict is one row in the output CSV.
    Axes swept:
      1. topology   — fixed p=20, n/p=5, alpha=0.05, K=25, f=0.5
      2. n/p ratio  — fixed topology=hub, p=20, alpha=0.05, K=25, f=0.5
      3. p          — fixed topology=hub, n/p=5, alpha=0.05, K=25, f=0.5
      4. alpha      — fixed topology=hub, p=20, n/p=5, K=25, f=0.5
      5. K (n_splits)      — fixed topology=hub, p=20, n/p=5, alpha=0.05, f=0.5
      6. subsample_frac    — fixed topology=hub, p=20, n/p=5, alpha=0.05, K=25
    """
    BASE = dict(topology="hub", p=20, n_over_p=5, alpha=0.05, n_splits=25, subsample_frac=0.5)

    def s(sweep_axis: str, **overrides) -> dict:
        d = {**BASE, "sweep_axis": sweep_axis}
        d.update(overrides)
        return d

    scenarios = [
        # ── 1. Topology sweep ──────────────────────────────────────────────
        s("topology", topology="null"),
        s("topology", topology="random"),
        s("topology", topology="hub"),         # baseline
        s("topology", topology="scale-free"),
        s("topology", topology="cluster"),
        s("topology", topology="dense"),

        # ── 2. n/p ratio sweep ────────────────────────────────────────────
        s("n_over_p", n_over_p=0.25),          # extreme underdetermined
        s("n_over_p", n_over_p=0.5),           # n < p
        s("n_over_p", n_over_p=1),
        s("n_over_p", n_over_p=2),
        # n/p=5 already covered by topology baseline above
        s("n_over_p", n_over_p=10),
        s("n_over_p", n_over_p=20),
        s("n_over_p", n_over_p=50),

        # ── 3. p sweep ────────────────────────────────────────────────────
        s("p", p=2),
        s("p", p=3),
        s("p", p=5),
        s("p", p=10),
        # p=20 already covered above
        s("p", p=50),

        # ── 4. alpha sweep ────────────────────────────────────────────────
        s("alpha", alpha=0.01),
        # alpha=0.05 already covered above
        s("alpha", alpha=0.10),
        s("alpha", alpha=0.20),

        # ── 5. K (n_splits) sweep ─────────────────────────────────────────
        s("n_splits", n_splits=2),             # minimum allowed
        s("n_splits", n_splits=5),
        s("n_splits", n_splits=10),
        # K=25 already covered above
        s("n_splits", n_splits=50),
        s("n_splits", n_splits=100),

        # ── 6. subsample_frac sweep ───────────────────────────────────────
        s("subsample_frac", subsample_frac=0.1),   # tiny subsamples (may hit n<p inside splits)
        s("subsample_frac", subsample_frac=0.3),
        # f=0.5 already covered above
        s("subsample_frac", subsample_frac=0.7),
        s("subsample_frac", subsample_frac=0.9),
    ]

    # Deduplicate: hub/p=20/n_over_p=5/alpha=0.05/K=25/f=0.5 appears several times;
    # keep only the first occurrence (topology="hub" row in sweep 1).
    seen = set()
    deduped = []
    for sc in scenarios:
        key = (sc["topology"], sc["p"], sc["n_over_p"], sc["alpha"],
               sc["n_splits"], sc["subsample_frac"])
        if key not in seen:
            seen.add(key)
            deduped.append(sc)
    return deduped


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=100,
                        help="Monte-Carlo trials per scenario (default 100)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default="finalisation")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "ensemble_ci_coverage.csv"

    scenarios = build_scenarios()
    print(f"Running {len(scenarios)} scenarios × {args.n_trials} trials each")
    print(f"{'#':>3} {'sweep':>14} {'topo':>10} {'p':>4} {'n/p':>5} "
          f"{'alpha':>6} {'K':>4} {'f':>4}  "
          f"{'asymp_all':>10} {'ens_all':>10} {'nominal':>8}")

    fieldnames = [
        "sweep_axis", "topology", "p", "n", "n_over_p",
        "alpha", "n_splits", "subsample_frac",
        "n_active", "edge_density",
        "asymp_cov_all", "asymp_cov_null", "asymp_cov_active",
        "ens_cov_all",   "ens_cov_null",   "ens_cov_active",
        "nominal_coverage", "error",
    ]

    rows = []
    for i, sc in enumerate(scenarios):
        p = sc["p"]
        n_over_p = sc["n_over_p"]
        n = max(2, int(round(n_over_p * p)))  # always at least 2 samples

        metrics = run_coverage(
            topology=sc["topology"],
            p=p,
            n=n,
            n_splits=sc["n_splits"],
            subsample_frac=sc["subsample_frac"],
            n_trials=args.n_trials,
            alpha=sc["alpha"],
            seed=args.seed,
        )

        nominal = 1.0 - sc["alpha"]
        row = {
            "sweep_axis":       sc["sweep_axis"],
            "topology":         sc["topology"],
            "p":                p,
            "n":                n,
            "n_over_p":         n_over_p,
            "alpha":            sc["alpha"],
            "n_splits":         sc["n_splits"],
            "subsample_frac":   sc["subsample_frac"],
            "nominal_coverage": nominal,
            **metrics,
        }
        rows.append(row)

        status = f"  ERROR: {metrics['error'][:60]}" if metrics["error"] else ""
        print(f"{i+1:>3} {sc['sweep_axis']:>14} {sc['topology']:>10} "
              f"{p:>4} {n_over_p:>5} {sc['alpha']:>6.2f} "
              f"{sc['n_splits']:>4} {sc['subsample_frac']:>4.1f}  "
              f"{metrics['asymp_cov_all']:>10.3f} {metrics['ens_cov_all']:>10.3f} "
              f"{nominal:>8.2f}{status}")

    with open(out_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    n_errors = sum(1 for r in rows if r["error"])
    print(f"\nDone. {len(rows)} scenarios, {n_errors} errors.")
    print(f"Results saved to {out_file}")


if __name__ == "__main__":
    main()
