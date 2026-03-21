"""
DREAM5 gold standard partial correlation coverage diagnostic (L8).

For each DREAM5 network, computes what fraction of directed gold standard
edges manifest as detectable partial correlations in the expression data.

A GGM edge (i,j) reflects a non-zero partial correlation after conditioning
on all other genes — a different quantity from a directed regulatory edge.
If <50% of gold standard edges have detectable partial correlation signal,
low AUPR for all methods is expected and should be reported as a benchmark
caveat rather than as poor method performance.

Partial correlations are estimated via the pseudo-inverse of the sample
correlation matrix (no regularisation), which is valid when n > p.
When n <= p (DREAM5 net1 can have p > n), a GraphicalLassoCV estimate
is used instead.

Output
------
Prints and saves: scripts/dream5_pcorr_coverage.csv

Usage (on Snellius)
-------------------
    cd $HOME/NODIS
    source .venv/bin/activate
    python scripts/dream5_partial_corr_coverage.py --data-dir data/dream5
    python scripts/dream5_partial_corr_coverage.py --data-dir data/dream5 --network 3
"""

import argparse
import pathlib
import warnings

import numpy as np
import pandas as pd


def partial_corr_matrix(X: np.ndarray) -> np.ndarray:
    """
    Estimate partial correlation matrix from data.

    Uses pseudo-inverse of correlation matrix when n > p (exact),
    falls back to GraphicalLassoCV precision when n <= p.
    """
    n, p = X.shape
    # Standardise
    mu = X.mean(axis=0)
    sd = X.std(axis=0, ddof=1)
    sd = np.where(sd > 0, sd, 1.0)
    Xs = (X - mu) / sd

    if n > p:
        C = np.corrcoef(Xs.T)           # (p, p) correlation matrix
        P = np.linalg.pinv(C)           # precision of correlation = partial corr scaffold
    else:
        from sklearn.covariance import GraphicalLassoCV
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            glcv = GraphicalLassoCV(cv=5, max_iter=500)
            glcv.fit(Xs)
        P = glcv.precision_

    # Partial correlation: pcorr_ij = -P_ij / sqrt(P_ii * P_jj)
    diag = np.diag(P)
    with np.errstate(divide="ignore", invalid="ignore"):
        denom = np.sqrt(np.outer(diag, diag))
        pcorr = np.where(denom > 0, -P / denom, 0.0)
    np.fill_diagonal(pcorr, 0.0)
    return pcorr


def main() -> None:
    parser = argparse.ArgumentParser(
        description="DREAM5 partial correlation coverage diagnostic."
    )
    parser.add_argument("--data-dir", default="data/dream5")
    parser.add_argument("--network", type=int, default=1, choices=[1, 3],
                        help="DREAM5 network (1=E.coli in silico, 3=E.coli in vivo)")
    parser.add_argument("--p", type=int, default=None,
                        help="Restrict to top-p genes by variance. "
                             "Default: use all genes.")
    parser.add_argument("--threshold", type=float, default=0.1,
                        help="Absolute partial correlation threshold for 'detectable' "
                             "(default 0.1).")
    parser.add_argument("--out", default="scripts/dream5_pcorr_coverage.csv")
    args = parser.parse_args()

    from nodis.simulate.loaders import load_dream5_insilico
    from nodis.preprocess.npn import npn_shrinkage

    print(f"Loading DREAM5 network {args.network} from {args.data_dir} ...")
    expr_df, gold = load_dream5_insilico(args.data_dir, network=args.network)

    if args.p is not None:
        gene_var = expr_df.var(axis=0)
        top_genes = gene_var.nlargest(args.p).index.tolist()
        expr_df = expr_df[top_genes]

    genes = expr_df.columns.tolist()
    gene_to_idx = {g: i for i, g in enumerate(genes)}
    p = len(genes)
    n = expr_df.shape[0]
    print(f"  n={n}, p={p}")

    # NPN-transform
    X = npn_shrinkage(expr_df.values)

    print("Estimating partial correlations ...")
    pcorr = partial_corr_matrix(X)

    # Gold standard edges (directed, symmetrised for GGM evaluation)
    positive_edges = gold[gold["label"] == 1]
    total_gold = len(positive_edges)
    in_gene_set = 0
    detectable = 0
    abs_pcorrs = []

    for _, row in positive_edges.iterrows():
        tf = row["TF"]
        tgt = row["target"]
        if tf not in gene_to_idx or tgt not in gene_to_idx:
            continue
        in_gene_set += 1
        i = gene_to_idx[tf]
        j = gene_to_idx[tgt]
        apc = abs(float(pcorr[i, j]))
        abs_pcorrs.append(apc)
        if apc >= args.threshold:
            detectable += 1

    coverage = detectable / in_gene_set if in_gene_set > 0 else float("nan")
    median_apc = float(np.median(abs_pcorrs)) if abs_pcorrs else float("nan")
    mean_apc   = float(np.mean(abs_pcorrs))   if abs_pcorrs else float("nan")

    print(f"\n=== DREAM5 Net{args.network} partial correlation coverage ===")
    print(f"  Total gold standard edges (directed):  {total_gold}")
    print(f"  Edges with both genes in gene set:     {in_gene_set}")
    print(f"  Edges with |pcorr| >= {args.threshold}: {detectable} "
          f"({100*coverage:.1f}%)")
    print(f"  Median |partial correlation|:           {median_apc:.4f}")
    print(f"  Mean   |partial correlation|:           {mean_apc:.4f}")
    print()
    if coverage < 0.5:
        print("  ⚠ Coverage < 50%: low AUPR on DREAM5 is expected for ALL methods.")
        print("    Report this fraction in the paper as a benchmark caveat.")
    else:
        print("  ✓ Coverage >= 50%: most gold standard edges have partial corr signal.")

    result = pd.DataFrame([{
        "network":           args.network,
        "n":                 n,
        "p":                 p,
        "total_gold_edges":  total_gold,
        "edges_in_gene_set": in_gene_set,
        "threshold":         args.threshold,
        "detectable":        detectable,
        "coverage_fraction": coverage,
        "median_abs_pcorr":  median_apc,
        "mean_abs_pcorr":    mean_apc,
    }])

    out_path = pathlib.Path(args.out)
    mode = "a" if out_path.exists() else "w"
    header = mode == "w"
    result.to_csv(out_path, mode=mode, header=header, index=False)
    print(f"Appended to {out_path}")


if __name__ == "__main__":
    main()
