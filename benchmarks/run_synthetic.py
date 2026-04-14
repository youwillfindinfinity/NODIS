"""
Synthetic benchmark runner — single task (designed for SLURM array dispatch).

Each invocation runs one (topology, config, method, rep) combination and
writes a single results CSV.  The SLURM array job decomposes the full
factorial grid (4 topologies × 3 n-configs × 4 methods × 50 reps = 2,400 runs).

Usage
-----
    python benchmarks/run_synthetic.py \\
        --topology hub --config n100p50 --method desparsified --rep 0 \\
        --out results/raw/

See jobs/synthetic_array.job for the SLURM wrapper.
"""

import argparse
import pathlib
import pickle
import time

import numpy as np
import pandas as pd

# n/p configurations matching SILGGM simulation study (Zhang et al. 2018)
CONFIGS = {
    "n100p50":   (100, 50),
    "n237p78":   (237, 78),
    "n513p164":  (513, 164),
    "n1026p328": (1026, 328),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--topology", required=True,
                        choices=["hub", "scale-free", "cluster", "random"])
    parser.add_argument("--config", required=True, choices=list(CONFIGS))
    parser.add_argument("--method", required=True,
                        choices=["desparsified", "glasso", "gglasso", "silggm_r",
                                 "piglasso", "piglasso_adaptive", "piglasso_corr",
                                 "piglasso_oracle",
                                 "piglasso_oracle_n01", "piglasso_oracle_n02",
                                 "piglasso_oracle_n03", "piglasso_oracle_n00"])
    parser.add_argument("--rep", type=int, required=True)
    parser.add_argument("--out", default="results/raw/")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--n-jobs", type=int, default=1,
                        help="Parallel workers for PIGLASSO subsampling loop.")
    parser.add_argument("--prior-noise", type=float, default=0.2,
                        help="Fraction of edge labels to flip for oracle prior (0–0.5).")
    parser.add_argument("--prior-weight", type=float, default=0.5,
                        help="Alpha in lambda_mask = 1 - alpha * prior.")
    args = parser.parse_args()

    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    result_file = out_dir / (
        f"results_{args.topology}_{args.config}"
        f"_method_{args.method}_rep{args.rep:03d}.csv"
    )
    adj_file = out_dir / (
        f"adj_{args.topology}_{args.config}"
        f"_method_{args.method}_rep{args.rep:03d}.pkl"
    )

    # For piglasso, also require the adj pickle before skipping — it is needed
    # by run_diffusion.py to avoid re-fitting during the diffusion benchmark.
    if result_file.exists() and (args.method != "piglasso" or adj_file.exists()):
        print(f"Already exists — skipping: {result_file}")
        return

    # ----------------------------------------------------------------
    # Generate data
    # ----------------------------------------------------------------
    from nodis.simulate.generator import generate
    from nodis.benchmark.evaluate import evaluate_predictions

    n, p = CONFIGS[args.config]
    seed = args.rep * 1000 + hash(args.topology) % 1000
    data = generate(n=n, p=p, topology=args.topology, prob=0.05, seed=seed)

    # ----------------------------------------------------------------
    # Fit estimator
    # ----------------------------------------------------------------
    t0 = time.perf_counter()

    if args.method == "desparsified":
        from nodis.estimators.desparsified import DesparifiedGGM
        est = DesparifiedGGM()
        est.fit(data.X)
        adj = est.get_adjacency(alpha=args.alpha)
        scores = np.abs(est.result_.z_scores)

    elif args.method == "glasso":
        from nodis.estimators.glasso import SklearnGLasso
        est = SklearnGLasso()
        est.fit(data.X)
        adj = est.get_adjacency()
        scores = np.abs(est.precision_)

    elif args.method == "gglasso":
        from nodis.estimators.glasso import GGLassoEstimator
        est = GGLassoEstimator()
        est.fit(data.X)
        adj = est.get_adjacency()
        scores = np.abs(est.precision_)

    elif args.method == "silggm_r":
        from nodis.estimators._silggm_bridge import run_silggm_r
        r_res = run_silggm_r(data.X, method="B_NW_SL", alpha=args.alpha)
        adj = r_res["adj"]
        scores = np.abs(r_res["z_score"])

    elif args.method == "piglasso":
        from nodis.estimators.piglasso import PIGLassoEstimator
        est = PIGLassoEstimator(n_jobs=args.n_jobs)
        est.fit(data.X)
        adj = est.get_adjacency()
        scores = est.precision_

    elif args.method == "piglasso_adaptive":
        from nodis.estimators.piglasso import PIGLassoEstimator
        est = PIGLassoEstimator(n_jobs=args.n_jobs, pi_thr="adaptive")
        est.fit(data.X)
        adj = est.get_adjacency()
        scores = est.precision_

    elif args.method == "piglasso_corr":
        from nodis.estimators.piglasso import PIGLassoEstimator
        from nodis.estimators.prior_utils import build_corr_prior
        prior = build_corr_prior(data.X, gamma=2.0)
        est = PIGLassoEstimator(n_jobs=args.n_jobs, prior_weight=args.prior_weight)
        est.fit(data.X, prior=prior)
        adj = est.get_adjacency()
        scores = est.precision_

    elif args.method in ("piglasso_oracle",
                         "piglasso_oracle_n00", "piglasso_oracle_n01",
                         "piglasso_oracle_n02", "piglasso_oracle_n03"):
        from nodis.estimators.piglasso import PIGLassoEstimator
        from nodis.estimators.prior_utils import build_noisy_oracle_prior
        # Noise level encoded in method name (n00=0.0, n01=0.1, n02=0.2, n03=0.3)
        # Falls back to --prior-noise arg when method == "piglasso_oracle"
        noise_map = {"piglasso_oracle_n00": 0.0, "piglasso_oracle_n01": 0.1,
                     "piglasso_oracle_n02": 0.2, "piglasso_oracle_n03": 0.3}
        noise = noise_map.get(args.method, args.prior_noise)
        adj_true_bin = (data.Omega != 0).astype(float)
        np.fill_diagonal(adj_true_bin, 0)
        prior = build_noisy_oracle_prior(adj_true_bin, noise=noise, seed=args.rep)
        est = PIGLassoEstimator(n_jobs=args.n_jobs, prior_weight=args.prior_weight)
        est.fit(data.X, prior=prior)
        adj = est.get_adjacency()
        scores = est.precision_

    wall = time.perf_counter() - t0

    # ----------------------------------------------------------------
    # Evaluate and save
    # ----------------------------------------------------------------
    metrics = evaluate_predictions(adj, data.Omega, scores=scores)
    metrics.update({
        "topology": args.topology,
        "config": args.config,
        "method": args.method,
        "rep": args.rep,
        "n": n,
        "p": p,
        "wall_seconds": wall,
    })

    if not result_file.exists():
        pd.DataFrame([metrics]).to_csv(result_file, index=False)
        print(f"Saved: {result_file}  (AUPR={metrics['aupr']:.3f}, t={wall:.1f}s)")

    if args.method == "piglasso" and not adj_file.exists():
        with open(adj_file, "wb") as fh:
            pickle.dump(adj, fh, protocol=4)
        print(f"Saved adj: {adj_file}")


if __name__ == "__main__":
    main()
