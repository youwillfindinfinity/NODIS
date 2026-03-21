"""
StARS vs BH-FDR comparison benchmark — single task (designed for SLURM array dispatch).

Runs StARS stability selection on the same (topology, config, rep) grid as
run_synthetic.py and writes a results CSV.  The companion aggregate script
merges these results with the synthetic benchmark CSVs to produce the
quantitative NODIS-BH vs StARS comparison described in L13.

Usage
-----
    python benchmarks/run_stars_comparison.py \\
        --topology hub --config n100p50 --rep 0 \\
        --out results/stars/

See jobs/stars_comparison_array.job for the SLURM wrapper.
"""

import argparse
import pathlib
import time

import numpy as np
import pandas as pd

# Must match run_synthetic.py exactly
CONFIGS = {
    "n100p50":  (100, 50),
    "n237p78":  (237, 78),
    "n513p164": (513, 164),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--topology", required=True,
                        choices=["hub", "scale-free", "cluster", "random"])
    parser.add_argument("--config", required=True, choices=list(CONFIGS))
    parser.add_argument("--rep", type=int, required=True)
    parser.add_argument("--out", default="results/stars/")
    parser.add_argument("--beta", type=float, default=0.05,
                        help="StARS instability threshold (Liu et al. 2010).")
    parser.add_argument("--n-subsamples", type=int, default=20,
                        help="Number of subsampled replicates per threshold.")
    parser.add_argument("--n-jobs", type=int, default=1,
                        help="Parallel workers for subsample fits.")
    args = parser.parse_args()

    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    result_file = out_dir / (
        f"stars_{args.topology}_{args.config}_rep{args.rep:03d}.csv"
    )
    if result_file.exists():
        print(f"Already exists — skipping: {result_file}")
        return

    from nodis.simulate.generator import generate
    from nodis.benchmark.evaluate import evaluate_predictions
    from nodis.estimators.desparsified import DesparifiedGGM
    from nodis.inference.stars import stars_select

    n, p = CONFIGS[args.config]
    seed = args.rep * 1000 + hash(args.topology) % 1000
    data = generate(n=n, p=p, topology=args.topology, prob=0.05, seed=seed)

    t0 = time.perf_counter()
    adj, selected_threshold = stars_select(
        data.X,
        DesparifiedGGM(),
        beta=args.beta,
        n_subsamples=args.n_subsamples,
        n_jobs=args.n_jobs,
        seed=seed,
    )
    wall = time.perf_counter() - t0

    # Use absolute z-scores as continuous edge scores for AUPR/AUROC
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        est_full = DesparifiedGGM().fit(data.X)
    scores = np.abs(est_full.result_.z_scores)

    metrics = evaluate_predictions(adj, data.Omega, scores=scores)
    metrics.update({
        "topology": args.topology,
        "config": args.config,
        "method": "stars",
        "rep": args.rep,
        "n": n,
        "p": p,
        "beta": args.beta,
        "selected_threshold": selected_threshold,
        "n_subsamples": args.n_subsamples,
        "wall_seconds": wall,
    })

    pd.DataFrame([metrics]).to_csv(result_file, index=False)
    print(
        f"Saved: {result_file}  "
        f"(AUPR={metrics['aupr']:.3f}, theta={selected_threshold:.3f}, t={wall:.1f}s)"
    )


if __name__ == "__main__":
    main()
