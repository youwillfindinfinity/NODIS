"""
DREAM5 Network Inference Challenge benchmark.

Downloads and caches DREAM5 Network 1 (E. coli in silico) data, then runs
NODIS and baselines, writing AUPR metrics to results/dream5/.

Usage
-----
    python benchmarks/run_dream5.py --p 200 --method desparsified
    python benchmarks/run_dream5.py --p 500 --method glasso

Reference
---------
Marbach D, et al. (2012). Wisdom of crowds for robust gene network inference.
    Nat Methods 9: 796–804. doi:10.1038/nmeth.2016
"""

import argparse
import pathlib
import time

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/dream5")
    parser.add_argument("--network", type=int, default=1, choices=[1, 3])
    parser.add_argument("--p", type=int, default=200,
                        help="Number of genes to retain (by variance).")
    parser.add_argument("--method", default="desparsified",
                        choices=["desparsified", "glasso", "gglasso", "piglasso"])
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--n-jobs", type=int, default=1,
                        help="Parallel CV jobs for SklearnGLasso (default 1).")
    parser.add_argument("--out", default="results/dream5/")
    args = parser.parse_args()

    from nodis.simulate.loaders import load_dream5_insilico
    from nodis.preprocess.npn import npn_shrinkage
    from nodis.benchmark.evaluate import evaluate_predictions

    expr_df, gold = load_dream5_insilico(args.data_dir, network=args.network)

    # Select top-p genes by variance
    gene_var = expr_df.var(axis=0)
    top_genes = gene_var.nlargest(args.p).index.tolist()
    X = npn_shrinkage(expr_df[top_genes].values)

    # Build (p, p) binary ground-truth adjacency using gold standard edge list.
    # Only edges where both TF and target are in the selected gene set are counted.
    # The graph is undirected so edges are symmetrised.
    p = args.p
    gene_to_idx = {g: i for i, g in enumerate(top_genes)}
    adj_true = np.zeros((p, p), dtype=int)
    positive_edges = gold[gold["label"] == 1]
    for _, row in positive_edges.iterrows():
        i = gene_to_idx.get(row["TF"])
        j = gene_to_idx.get(row["target"])
        if i is not None and j is not None:
            adj_true[i, j] = 1
            adj_true[j, i] = 1

    n_edges_true = int(adj_true[np.triu_indices(p, k=1)].sum())
    print(f"Expression matrix: {X.shape}, gold edges in subgraph: {n_edges_true}")

    t0 = time.perf_counter()
    if args.method == "desparsified":
        from nodis.estimators.desparsified import DesparifiedGGM
        est = DesparifiedGGM()
        est.fit(X)
        adj = est.get_adjacency(alpha=args.alpha)
        scores = np.abs(est.result_.z_scores)
    elif args.method == "glasso":
        from nodis.estimators.glasso import SklearnGLasso
        est = SklearnGLasso(n_jobs=args.n_jobs)
        est.fit(X)
        adj = est.get_adjacency()
        scores = np.abs(est.precision_)
    elif args.method == "gglasso":
        from nodis.estimators.glasso import GGLassoEstimator
        est = GGLassoEstimator()
        est.fit(X)
        adj = est.get_adjacency()
        scores = np.abs(est.precision_)

    elif args.method == "piglasso":
        from nodis.estimators.piglasso import PIGLassoEstimator
        est = PIGLassoEstimator(n_jobs=args.n_jobs)
        est.fit(X)
        adj = est.get_adjacency()
        scores = est.precision_

    wall = time.perf_counter() - t0

    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = out_dir / f"dream5_net{args.network}_p{p}_method_{args.method}.csv"
    metrics = evaluate_predictions(adj, adj_true, scores=scores)
    metrics.update({"method": args.method, "p": p, "network": args.network,
                    "wall_seconds": wall})
    pd.DataFrame([metrics]).to_csv(fname, index=False)
    print(f"Results → {fname}  (t={wall:.1f}s)")


if __name__ == "__main__":
    main()
