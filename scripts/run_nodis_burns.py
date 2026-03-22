"""
NODIS de-sparsified inference on burn expression data.

Input
-----
- expression_top5000.tsv  : genes × samples, already preprocessed by the PIGLasso
                            pipeline (probe→gene collapse, control removal,
                            missingness filter, winsorisation 1st/99th pct,
                            gene-wise z-score, Acute-phase only).
- healthy_controls.genes.txt : gene list from GSE236713 (Agilent GPL17077,
                               platform-matched control cohort).

Preprocessing applied here (on top of PIGLasso pipeline)
---------------------------------------------------------
1. Intersect expression genes with GSE236713 gene list → reduces to the
   shared gene set where NODIS statistics are interpretable alongside
   the healthy baseline used for diffusion.
2. Apply NPN shrinkage transform (rank-based, matches huge::huge.npn) to
   correct remaining marginal non-Gaussianity after z-score normalisation.

Inference
---------
DesparsifiedGGM (scaled Lasso tuning, symmetrised de-biased estimator).
FDR control: Benjamini-Hochberg at alpha=0.05 and alpha=0.10.

Outputs (written to --out)
--------------------------
  burns_nodis_zscores.csv      — (p×p) symmetric z-score matrix
  burns_nodis_pvalues.csv      — (p×p) symmetric two-sided p-value matrix
  burns_nodis_adj_fdr05.csv    — binary adjacency at BH FDR 5%
  burns_nodis_adj_fdr10.csv    — binary adjacency at BH FDR 10%
  burns_nodis_genes.txt        — gene list (one per line, matches matrix rows)
  burns_nodis_summary.txt      — n, p, n/p, edges at each FDR level, warnings

Usage (local or Snellius)
-------------------------
    python scripts/run_nodis_burns.py \\
        --expr   data/burn/expression_top5000.tsv \\
        --ctrl-genes data/gse236713/healthy_controls.genes.txt \\
        --out    results/burns/

    # To run on ALL top-5000 genes (very slow, n/p=0.12 — exploratory only):
    python scripts/run_nodis_burns.py \\
        --expr   data/burn/expression_top5000.tsv \\
        --no-intersect \\
        --out    results/burns/

n/p note
--------
At the intersection gene set (~164 genes, n=584), n/p ≈ 3.6 — below the
conservative n > 5p floor but within the cautionary range where z-scores are
usable with appropriate caveats. Results are labelled CAUTIONARY in the output.
"""
from __future__ import annotations

import argparse
import pathlib
import warnings

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_gene_list(path: pathlib.Path) -> list[str]:
    return [l.strip() for l in path.read_text().splitlines() if l.strip()]


def intersect_expression(expr: pd.DataFrame, ctrl_genes: list[str]) -> pd.DataFrame:
    shared = [g for g in ctrl_genes if g in expr.index]
    missing = len(ctrl_genes) - len(shared)
    if missing:
        warnings.warn(f"{missing}/{len(ctrl_genes)} control genes not in expression matrix.")
    sub = expr.loc[shared]
    print(f"Intersection: {len(shared)} genes (dropped {len(expr) - len(shared)} "
          f"from expression, {missing} from ctrl gene list)")
    return sub


def npn_shrinkage(X: np.ndarray) -> np.ndarray:
    """
    Rank-based nonparanormal shrinkage transform.
    Matches nodis.preprocess.npn.npn_shrinkage exactly.
    Kept inline to avoid import issues when run standalone on Snellius.
    """
    from scipy.stats import rankdata, norm as sp_norm

    n, p = X.shape
    delta = 1.0 / (4.0 * n ** 0.25 * np.sqrt(np.pi * np.log(n)))
    Z = np.empty_like(X, dtype=float)
    for j in range(p):
        r = rankdata(X[:, j], method="average")
        # shrinkage: clip rank fraction to [delta, 1-delta] before normal quantile
        u = np.clip(r / (n + 1), delta, 1.0 - delta)
        Z[:, j] = sp_norm.ppf(u)
    return Z


def run_desparsified(X: np.ndarray, gene_names: list[str], lambda_scale: float = 1.0):
    """Fit DesparsifiedGGM. Returns (z_scores, p_values) as DataFrames."""
    from nodis.estimators.desparsified import DesparifiedGGM as DesparsifiedGGM
    from nodis.inference.pvalues import z_to_pvalues as compute_pvalues

    n, p = X.shape
    print(f"Fitting DesparsifiedGGM: n={n}, p={p}, n/p={n/p:.2f}")
    if n / p < 5:
        warnings.warn(
            f"n/p = {n/p:.2f} < 5. Asymptotic normality of z-scores is not guaranteed. "
            "Results are CAUTIONARY and should not be used as primary evidence.",
            UserWarning,
        )

    est = DesparsifiedGGM(lambda_scale=lambda_scale)
    est.fit(X)

    Z = pd.DataFrame(est.result_.z_scores, index=gene_names, columns=gene_names)
    P = pd.DataFrame(compute_pvalues(est.result_.z_scores), index=gene_names, columns=gene_names)
    return Z, P


def apply_fdr(P: pd.DataFrame, alpha: float) -> pd.DataFrame:
    from nodis.inference.fdr import fdr_control
    p_arr = P.values
    adj = fdr_control(p_arr, alpha=alpha, method="BH")
    return pd.DataFrame(adj, index=P.index, columns=P.columns)


def write_summary(
    out_dir: pathlib.Path,
    n: int,
    p: int,
    edges_05: int,
    edges_10: int,
    pi_adaptive: float | None,
) -> None:
    lines = [
        f"n (samples):          {n}",
        f"p (genes):            {p}",
        f"n/p ratio:            {n/p:.3f}",
        f"Calibration:          {'CAUTIONARY (n/p < 5)' if n/p < 5 else 'OK'}",
        f"",
        f"Edges at FDR 5%:      {edges_05}",
        f"Edges at FDR 10%:     {edges_10}",
        f"",
        f"Preprocessing:        PIGLasso pipeline (probe→gene, winsor, z-score) + NPN",
        f"Gene set:             intersection with GSE236713 healthy controls",
        f"FDR method:           Benjamini-Hochberg",
    ]
    (out_dir / "burns_nodis_summary.txt").write_text("\n".join(lines))
    print("\n".join(lines))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run NODIS de-sparsified inference on burn expression data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--expr", required=True, type=pathlib.Path,
                        help="genes × samples TSV (preprocessed by PIGLasso pipeline).")
    parser.add_argument("--ctrl-genes", type=pathlib.Path, default=None,
                        help="Gene symbol list for intersection. "
                             "Expression is restricted to these genes.")
    parser.add_argument("--no-intersect", action="store_true",
                        help="Skip gene intersection (uses all genes in --expr). "
                             "WARNING: n/p will be very low for top-5000 input.")
    parser.add_argument("--top-p", type=int, default=None,
                        help="After any intersection, retain only the top-p genes "
                             "by expression variance. Use to control n/p ratio "
                             "(e.g. --top-p 164 gives n/p≈3.6 at n=584).")
    parser.add_argument("--lambda-scale", type=float, default=1.0,
                        help="Scaling constant for the scaled Lasso tuning parameter.")
    parser.add_argument("--out", required=True, type=pathlib.Path,
                        help="Output directory.")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    # 1. Load preprocessed expression (genes × samples)
    print(f"Loading expression: {args.expr}")
    expr = pd.read_csv(args.expr, sep="\t", index_col=0)
    print(f"  {expr.shape[0]} genes × {expr.shape[1]} samples")

    # 2. Intersect with control gene list
    if not args.no_intersect:
        if args.ctrl_genes is None:
            raise ValueError("--ctrl-genes required unless --no-intersect is set.")
        ctrl_genes = load_gene_list(args.ctrl_genes)
        expr = intersect_expression(expr, ctrl_genes)

    # 2b. Optionally restrict to top-p genes by variance
    if args.top_p is not None and args.top_p < len(expr):
        gene_var = expr.var(axis=1)
        top_genes = gene_var.nlargest(args.top_p).index.tolist()
        expr = expr.loc[top_genes]
        print(f"Restricted to top-{args.top_p} genes by variance: {len(expr)} genes remain")

    gene_names = list(expr.index)
    p = len(gene_names)
    # samples × genes for NODIS
    X_raw = expr.values.T.astype(float)
    n = X_raw.shape[0]

    # 3. NPN shrinkage (on top of PIGLasso z-score preprocessing)
    print(f"Applying NPN shrinkage transform (n={n}, p={p}) ...")
    X = npn_shrinkage(X_raw)

    # 4. De-sparsified inference
    Z, P = run_desparsified(X, gene_names, lambda_scale=args.lambda_scale)

    # 5. FDR-controlled adjacency
    adj_05 = apply_fdr(P, alpha=0.05)
    adj_10 = apply_fdr(P, alpha=0.10)
    edges_05 = int(adj_05.values.sum()) // 2
    edges_10 = int(adj_10.values.sum()) // 2
    print(f"Edges retained: {edges_05} at FDR 5%, {edges_10} at FDR 10%")

    # 6. Save
    Z.to_csv(args.out / "burns_nodis_zscores.csv")
    P.to_csv(args.out / "burns_nodis_pvalues.csv")
    adj_05.to_csv(args.out / "burns_nodis_adj_fdr05.csv")
    adj_10.to_csv(args.out / "burns_nodis_adj_fdr10.csv")
    (args.out / "burns_nodis_genes.txt").write_text("\n".join(gene_names))
    write_summary(args.out, n, p, edges_05, edges_10, pi_adaptive=None)

    print(f"\nAll outputs written to {args.out}")


if __name__ == "__main__":
    main()
