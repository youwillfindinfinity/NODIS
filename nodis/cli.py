"""
NODIS command-line interface.

Commands
--------
simulate  — generate synthetic GGM benchmark data
run       — run GGM inference on an expression matrix CSV
evaluate  — evaluate an inferred network against a ground-truth adjacency
plot      — visualise results from a benchmark run
"""

from __future__ import annotations

import sys
import pathlib

import click
import numpy as np
import pandas as pd


@click.group()
@click.version_option()
def main() -> None:
    """NODIS: Statistical inference for Gaussian Graphical Models."""


# ---------------------------------------------------------------------------
# simulate
# ---------------------------------------------------------------------------

@main.command()
@click.option("--n", default=200, show_default=True, help="Number of samples.")
@click.option("--p", default=100, show_default=True, help="Number of genes.")
@click.option(
    "--topology",
    default="hub",
    show_default=True,
    type=click.Choice(["hub", "scale-free", "cluster", "random"]),
    help="Graph topology.",
)
@click.option("--reps", default=10, show_default=True, help="Number of replicates.")
@click.option("--prob", default=0.05, show_default=True, help="Edge density (for random topology).")
@click.option("--seed", default=42, show_default=True, help="Base random seed.")
@click.option("--out", default="results/simulated/", show_default=True,
              help="Output directory.")
def simulate(n: int, p: int, topology: str, reps: int, prob: float, seed: int, out: str) -> None:
    """Generate synthetic GGM benchmark datasets."""
    from nodis.simulate.generator import generate
    import pickle

    out_dir = pathlib.Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)

    for rep in range(reps):
        data = generate(n=n, p=p, topology=topology, prob=prob, seed=seed + rep)
        fname = out_dir / f"{topology}_n{n}_p{p}_rep{rep:03d}.pkl"
        with open(fname, "wb") as fh:
            pickle.dump(data, fh)

    click.echo(f"Generated {reps} replicates → {out_dir}")


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------

@main.command()
@click.option("--data", required=True, help="Path to expression matrix CSV (samples × genes).")
@click.option(
    "--method",
    default="desparsified",
    show_default=True,
    type=click.Choice(["desparsified", "glasso", "gglasso"]),
    help="Inference method.",
)
@click.option("--alpha", default=0.05, show_default=True, help="FDR level.")
@click.option("--fdr", default="BH", show_default=True,
              type=click.Choice(["BH", "BY"]), help="FDR procedure.")
@click.option("--npn", is_flag=True, default=False, help="Apply NPN preprocessing.")
@click.option("--out", default="results/", show_default=True, help="Output directory.")
def run(
    data: str, method: str, alpha: float, fdr: str, npn: bool, out: str
) -> None:
    """Run GGM inference on an expression matrix."""
    out_dir = pathlib.Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)

    X = pd.read_csv(data, index_col=0).values.astype(float)
    click.echo(f"Loaded expression matrix: {X.shape[0]} samples × {X.shape[1]} genes")

    if npn:
        from nodis.preprocess.npn import npn_shrinkage
        X = npn_shrinkage(X)
        click.echo("Applied NPN preprocessing.")

    if method == "desparsified":
        from nodis.estimators.desparsified import DesparifiedGGM
        est = DesparifiedGGM()
        est.fit(X)
        adj = est.get_adjacency(alpha=alpha, method=fdr)
        stem = pathlib.Path(data).stem
        pd.DataFrame(est.result_.p_values).to_csv(out_dir / f"{stem}_pvalues.csv", index=False)
        pd.DataFrame(est.result_.z_scores).to_csv(out_dir / f"{stem}_zscores.csv", index=False)
        pd.DataFrame(adj).to_csv(out_dir / f"{stem}_adjacency.csv", index=False)

    elif method == "glasso":
        from nodis.estimators.glasso import SklearnGLasso
        est = SklearnGLasso()
        est.fit(X)
        adj = est.get_adjacency()
        stem = pathlib.Path(data).stem
        pd.DataFrame(adj).to_csv(out_dir / f"{stem}_adjacency.csv", index=False)

    elif method == "gglasso":
        from nodis.estimators.glasso import GGLassoEstimator
        est = GGLassoEstimator()
        est.fit(X)
        adj = est.get_adjacency()
        stem = pathlib.Path(data).stem
        pd.DataFrame(adj).to_csv(out_dir / f"{stem}_adjacency.csv", index=False)

    click.echo(f"Results written to {out_dir}")


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------

@main.command()
@click.option("--predicted", required=True, help="Path to predicted adjacency CSV.")
@click.option("--ground-truth", required=True, help="Path to ground-truth adjacency CSV.")
@click.option("--scores", default=None, help="Path to continuous scores CSV (optional).")
@click.option("--out", default="results/metrics.csv", show_default=True, help="Output CSV.")
def evaluate(predicted: str, ground_truth: str, scores: str | None, out: str) -> None:
    """Evaluate an inferred network against a known ground truth."""
    from nodis.benchmark.evaluate import evaluate_predictions

    adj_pred = pd.read_csv(predicted, index_col=None, header=0).values.astype(int)
    adj_true = pd.read_csv(ground_truth, index_col=None, header=0).values.astype(int)
    score_mat = (
        pd.read_csv(scores, index_col=None, header=0).values.astype(float)
        if scores else None
    )

    metrics = evaluate_predictions(adj_pred, adj_true, scores=score_mat)
    pd.DataFrame([metrics]).to_csv(out, index=False)
    click.echo(f"Metrics written to {out}")
    for k, v in metrics.items():
        if isinstance(v, float):
            click.echo(f"  {k}: {v:.4f}")
        else:
            click.echo(f"  {k}: {v}")


# ---------------------------------------------------------------------------
# enrich
# ---------------------------------------------------------------------------

@main.command("enrich")
@click.option("--adj", "adj_path", required=True, type=click.Path(exists=True),
              help="Path to .npy binary adjacency matrix (FDR-controlled).")
@click.option("--genes", "genes_path", required=True, type=click.Path(exists=True),
              help="Path to a text file with one gene name per line.")
@click.option("--pvalues", "pval_path", default=None, type=click.Path(exists=True),
              help="Path to .npy edge p-value matrix (optional; required for prerank).")
@click.option("--level", default="all",
              type=click.Choice(["rna", "post_transcriptional", "protein", "all"]),
              show_default=True, help="Biological level(s) to query.")
@click.option("--method", default="ora",
              type=click.Choice(["ora", "prerank"]),
              show_default=True, help="Enrichment method.")
@click.option("--backend", default="gprofiler",
              type=click.Choice(["gprofiler", "gseapy"]),
              show_default=True, help="Enrichment backend.")
@click.option("--extraction", default="hub",
              type=click.Choice(["hub", "prerank", "community"]),
              show_default=True, help="Gene extraction strategy.")
@click.option("--organism", default="hsapiens", show_default=True,
              help="Organism code (g:Profiler format).")
@click.option("--out", "out_path", default="enrichment_results.csv",
              show_default=True, help="Output CSV path for enrichment results.")
def enrich_cmd(adj_path, genes_path, pval_path, level, method, backend,
               extraction, organism, out_path):
    """Run topology-aware gene enrichment on a GGM adjacency matrix.

    \b
    Covers three biological levels:
      rna                   GO terms, KEGG, Reactome
      post_transcriptional  miRNA targets, TF motifs
      protein               CORUM complexes, InterPro domains
      all                   All three combined (default)

    \b
    Example:
      nodis enrich --adj adj.npy --genes genes.txt --level all --out results.csv
    """
    from nodis.enrich import from_adjacency

    adj = np.load(adj_path)
    with open(genes_path) as fh:
        gene_names = [line.strip() for line in fh if line.strip()]
    p_values = np.load(pval_path) if pval_path else None

    click.echo(
        f"Running enrichment: level={level}, method={method}, "
        f"backend={backend}, extraction={extraction}, "
        f"genes={len(gene_names)}"
    )

    hits = from_adjacency(
        adj=adj,
        gene_names=gene_names,
        p_values=p_values,
        level=level,
        method=method,
        backend=backend,
        extraction=extraction,
        organism=organism,
    )

    if not hits:
        click.echo("No enrichment results returned.")
        return

    frames = []
    for h in hits:
        if h.is_empty():
            continue
        df = h.results.copy()
        df.insert(0, "gene_set_name", h.gene_set_name)
        df.insert(1, "level", h.level)
        df.insert(2, "backend", h.backend)
        df.insert(3, "method", h.method)
        frames.append(df)

    if not frames:
        click.echo("No significant enrichment found.")
        return

    out_df = pd.concat(frames, ignore_index=True)
    out_df.to_csv(out_path, index=False)
    click.echo(f"Saved {len(out_df)} enrichment terms to {out_path}")


if __name__ == "__main__":
    main()
