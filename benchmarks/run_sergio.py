"""
SERGIO single-cell RNA-seq simulation benchmark.

Loads pre-built SERGIO datasets (Dibaeinia & Sinha 2020) and evaluates
NODIS with/without NPN preprocessing (RQ6).

Usage
-----
    python benchmarks/run_sergio.py --dataset-id 4 --method desparsified --npn

Reference
---------
Dibaeinia P, Sinha S (2020). SERGIO: A single-cell expression simulator
    guided by gene regulatory networks.
    Cell Syst 11(5): 452–466. doi:10.1016/j.cels.2020.08.003
"""

import argparse
import pathlib
import time

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/sergio")
    parser.add_argument("--dataset-id", type=int, default=4, choices=range(1, 12))
    parser.add_argument("--method", default="desparsified",
                        choices=["desparsified", "glasso"])
    parser.add_argument("--npn", action="store_true",
                        help="Apply log2(x+1) + NPN preprocessing.")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--out", default="results/sergio/")
    args = parser.parse_args()

    from nodis.simulate.loaders import load_sergio_dataset
    from nodis.benchmark.evaluate import evaluate_predictions

    expr, adj_true = load_sergio_dataset(args.data_dir, args.dataset_id)
    X = expr.T.astype(float)   # (n_cells, n_genes)

    if args.npn:
        from nodis.preprocess.npn import npn_shrinkage
        X = np.log2(X + 1.0)
        X = npn_shrinkage(X)
        preproc_label = "log2_npn"
    else:
        preproc_label = "none"

    t0 = time.perf_counter()
    if args.method == "desparsified":
        from nodis.estimators.desparsified import DesparifiedGGM
        est = DesparifiedGGM()
        est.fit(X)
        adj = est.get_adjacency(alpha=args.alpha)
        scores = np.abs(est.result_.z_scores)
    elif args.method == "glasso":
        from nodis.estimators.glasso import SklearnGLasso
        est = SklearnGLasso()
        est.fit(X)
        adj = est.get_adjacency()
        scores = np.abs(est.precision_)
    wall = time.perf_counter() - t0

    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = out_dir / (
        f"sergio_ds{args.dataset_id}_method_{args.method}"
        f"_preproc_{preproc_label}.csv"
    )
    metrics = evaluate_predictions(adj, adj_true, scores=scores)
    metrics.update({
        "method": args.method, "preprocessing": preproc_label,
        "dataset_id": args.dataset_id, "wall_seconds": wall,
    })
    pd.DataFrame([metrics]).to_csv(fname, index=False)
    print(f"Results → {fname}  (AUPR={metrics['aupr']:.3f}, t={wall:.1f}s)")


if __name__ == "__main__":
    main()
