"""
Diffusion/knockout benchmark runner — single task (designed for SLURM array dispatch).

Each invocation runs one (topology, config, method, rep) combination.
With --delta all, three CSVs are written (one per delta mode).
PIGLASSO is always evaluated with delta=random regardless of --delta.

Usage
-----
    python benchmarks/run_diffusion.py \\
        --topology hub --config n100p50 --method desparsified --rep 0 \\
        --out results/diffusion/

    python benchmarks/run_diffusion.py \\
        --topology cluster --config n513p164 --method piglasso --rep 12 \\
        --n-jobs 4 --out results/diffusion/

See jobs/diffusion_array.job and jobs/piglasso_diffusion.job for SLURM wrappers.
"""

import argparse
import math
import pathlib
import time

import numpy as np
import pandas as pd

CONFIGS = {
    "n100p50":   (100, 50),
    "n237p78":   (237, 78),
    "n513p164":  (513, 164),
    "n1026p328": (1026, 328),
}

# Uniform CSV schema — inactive mode columns are written as NaN
_COLUMNS = [
    "topology", "config", "method", "rep", "n", "p", "delta_mode",
    "diffusion_spearman_mean", "diffusion_spearman_min", "diffusion_mae_mean",
    "knockout_spearman", "knockout_top10_recall", "knockout_topk_recall",
    "n_components_pred",
    # Null baseline (mean over N_NULL random graphs, same edge count as adj_pred)
    "diffusion_spearman_mean_null", "knockout_spearman_null",
    # Normalised: (inferred - null) / (1.0 - null) for Spearman [0=random, 1=perfect]
    "diffusion_spearman_norm", "knockout_spearman_norm",
    "wall_seconds",
]

N_NULL = 5  # random graphs averaged for null baseline


def _fit_estimator(method: str, X: np.ndarray, alpha: float, n_jobs: int):
    """Fit estimator and return (adj, scores)."""
    if method == "desparsified":
        from nodis.estimators.desparsified import DesparifiedGGM
        est = DesparifiedGGM(n_jobs=n_jobs)
        est.fit(X)
        return est.get_adjacency(alpha=alpha), np.abs(est.result_.z_scores)

    if method == "glasso":
        from nodis.estimators.glasso import SklearnGLasso
        est = SklearnGLasso()
        est.fit(X)
        return est.get_adjacency(), np.abs(est.precision_)

    if method == "gglasso":
        from nodis.estimators.glasso import GGLassoEstimator
        est = GGLassoEstimator()
        est.fit(X)
        return est.get_adjacency(), np.abs(est.precision_)

    if method == "piglasso":
        from nodis.estimators.piglasso import PIGLassoEstimator
        est = PIGLassoEstimator(n_jobs=n_jobs)
        est.fit(X)
        return est.get_adjacency(), est.precision_

    raise ValueError(f"Unknown method: {method!r}")


def _run_one_delta(
    method: str,
    adj_pred: np.ndarray,
    adj_true: np.ndarray,
    delta_mode: str,
    seed: int,
    t_grid: np.ndarray,
    mode: str,
    reduction: float,
    topk: int,
    normalised: bool,
) -> dict:
    """Run diffusion and/or knockout for one delta mode. Returns metric dict."""
    from nodis.benchmark.diffusion_eval import (
        make_laplacian, make_delta,
        evaluate_diffusion, evaluate_knockouts,
        erdos_renyi_matching,
    )

    L_true = make_laplacian(adj_true, normalised=False)   # always unnormalised for delta
    delta  = make_delta(adj_true, L_true, delta_mode, seed)

    metrics: dict = {}

    if mode in {"diffusion", "both"}:
        metrics.update(evaluate_diffusion(
            adj_pred, adj_true, delta, t_grid, normalised=normalised,
        ))
    else:
        metrics.update({
            "diffusion_spearman_mean": float("nan"),
            "diffusion_spearman_min":  float("nan"),
            "diffusion_mae_mean":      float("nan"),
            "n_components_pred":       float("nan"),
        })

    if mode in {"knockout", "both"}:
        metrics.update(evaluate_knockouts(
            adj_pred, adj_true, delta, t_grid,
            reduction=reduction, topk=topk,
        ))
    else:
        metrics.update({
            "knockout_spearman":     float("nan"),
            "knockout_top10_recall": float("nan"),
            "knockout_topk_recall":  float("nan"),
        })

    # n_components_pred from evaluate_diffusion; fill for knockout-only mode
    if mode == "knockout" and "n_components_pred" not in metrics:
        metrics["n_components_pred"] = float("nan")

    # ------------------------------------------------------------------
    # Null baseline: average metrics over N_NULL random graphs (L7)
    # Random graphs match adj_pred edge count — same density, random topology.
    # ------------------------------------------------------------------
    null_diff_spearman = []
    null_ko_spearman   = []

    for k in range(N_NULL):
        adj_null = erdos_renyi_matching(adj_pred, seed=seed + k + 1)

        if mode in {"diffusion", "both"}:
            nd = evaluate_diffusion(adj_null, adj_true, delta, t_grid,
                                    normalised=normalised)
            null_diff_spearman.append(nd["diffusion_spearman_mean"])

        if mode in {"knockout", "both"}:
            nk = evaluate_knockouts(adj_null, adj_true, delta, t_grid,
                                    reduction=reduction, topk=topk)
            null_ko_spearman.append(nk["knockout_spearman"])

    diff_null = float(np.mean(null_diff_spearman)) if null_diff_spearman else float("nan")
    ko_null   = float(np.mean(null_ko_spearman))   if null_ko_spearman   else float("nan")

    metrics["diffusion_spearman_mean_null"] = diff_null
    metrics["knockout_spearman_null"]        = ko_null

    # Normalised Spearman: (inferred - null) / (1.0 - null)
    # 0 = indistinguishable from random; 1 = perfect recovery
    def _norm(score, null):
        denom = 1.0 - null
        if not math.isfinite(score) or not math.isfinite(null) or abs(denom) < 1e-9:
            return float("nan")
        return (score - null) / denom

    metrics["diffusion_spearman_norm"] = _norm(
        metrics.get("diffusion_spearman_mean", float("nan")), diff_null
    )
    metrics["knockout_spearman_norm"] = _norm(
        metrics.get("knockout_spearman", float("nan")), ko_null
    )

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diffusion/knockout benchmark — one (topology, config, method, rep)."
    )
    parser.add_argument("--topology",  required=True,
                        choices=["hub", "scale-free", "cluster", "random"])
    parser.add_argument("--config",    required=True, choices=list(CONFIGS))
    parser.add_argument("--method",    required=True,
                        choices=["desparsified", "glasso", "gglasso", "piglasso"])
    parser.add_argument("--rep",       type=int, required=True)
    parser.add_argument("--mode",      default="both",
                        choices=["diffusion", "knockout", "both"])
    parser.add_argument("--delta",     default="all",
                        choices=["random", "hub", "fiedler", "all"],
                        help="Delta mode(s). 'all' writes one CSV per mode. "
                             "piglasso is always forced to 'random'.")
    parser.add_argument("--out",       default="results/diffusion/")
    parser.add_argument("--tmin",      type=float, default=1e-4)
    parser.add_argument("--tmax",      type=float, default=3.0)
    parser.add_argument("--nt",        type=int,   default=80)
    parser.add_argument("--reduction", type=float, default=0.3)
    parser.add_argument("--topk",      type=int,   default=10)
    parser.add_argument("--n-jobs",    type=int,   default=1,
                        help="Parallel workers for piglasso subsampling.")
    parser.add_argument("--alpha",     type=float, default=0.05)
    parser.add_argument("--adj-dir",   default=None,
                        help="Directory containing pre-saved adj pickles from "
                             "run_synthetic.py. Defaults to ../raw/ relative to "
                             "--out. If the pickle exists the estimator is not "
                             "re-fitted.")
    parser.add_argument("--normalised-laplacian", action="store_true",
                        help="Use normalised Laplacian for diffusion kernel.")
    args = parser.parse_args()

    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    n, p = CONFIGS[args.config]

    # Determine which delta modes to evaluate
    if args.method == "piglasso":
        delta_modes = ["random"]
    elif args.delta == "all":
        delta_modes = ["random", "hub", "fiedler"]
    else:
        delta_modes = [args.delta]

    # Check whether all output files already exist
    result_files = {
        dm: out_dir / (
            f"diffusion_{args.topology}_{args.config}"
            f"_method_{args.method}_delta_{dm}_rep{args.rep:03d}.csv"
        )
        for dm in delta_modes
    }
    if all(f.exists() for f in result_files.values()):
        print(f"All outputs exist — skipping: {list(result_files.values())}")
        return

    # ----------------------------------------------------------------
    # Generate data (same seed convention as run_synthetic.py)
    # ----------------------------------------------------------------
    import pickle
    from nodis.simulate.generator import generate

    seed = args.rep * 1000 + hash(args.topology) % 1000
    data = generate(n=n, p=p, topology=args.topology, prob=0.05, seed=seed)

    # ----------------------------------------------------------------
    # Obtain adjacency — load pickle if available, otherwise fit
    # ----------------------------------------------------------------
    t0 = time.perf_counter()

    adj_dir = pathlib.Path(args.adj_dir) if args.adj_dir else out_dir.parent / "raw"
    adj_pickle = adj_dir / (
        f"adj_{args.topology}_{args.config}"
        f"_method_{args.method}_rep{args.rep:03d}.pkl"
    )

    if adj_pickle.exists():
        with open(adj_pickle, "rb") as fh:
            adj_pred = pickle.load(fh)
        print(f"Loaded adj from pickle: {adj_pickle}")
    else:
        adj_pred, _ = _fit_estimator(args.method, data.X, args.alpha, args.n_jobs)

    fit_wall = time.perf_counter() - t0

    t_grid = np.linspace(args.tmin, args.tmax, args.nt)

    # ----------------------------------------------------------------
    # Evaluate each delta mode
    # ----------------------------------------------------------------
    for dm in delta_modes:
        out_file = result_files[dm]
        if out_file.exists():
            print(f"Already exists — skipping: {out_file}")
            continue

        t_eval = time.perf_counter()
        metrics = _run_one_delta(
            method=args.method,
            adj_pred=adj_pred,
            adj_true=data.Omega,
            delta_mode=dm,
            seed=seed,
            t_grid=t_grid,
            mode=args.mode,
            reduction=args.reduction,
            topk=args.topk,
            normalised=args.normalised_laplacian,
        )
        eval_wall = time.perf_counter() - t_eval

        row = {
            "topology":  args.topology,
            "config":    args.config,
            "method":    args.method,
            "rep":       args.rep,
            "n":         n,
            "p":         p,
            "delta_mode": dm,
            **metrics,
            "wall_seconds": fit_wall + eval_wall,
        }

        tmp_file = out_file.with_suffix(".tmp")
        pd.DataFrame([row])[_COLUMNS].to_csv(tmp_file, index=False)
        tmp_file.rename(out_file)

        ds = row.get("diffusion_spearman_mean", float("nan"))
        ks = row.get("knockout_spearman", float("nan"))
        print(
            f"Saved: {out_file}  "
            f"(diff_rho={ds:.3f}, ko_rho={ks:.3f}, t={row['wall_seconds']:.1f}s)"
        )


if __name__ == "__main__":
    main()
