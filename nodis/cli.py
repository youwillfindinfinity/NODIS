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
@click.pass_context
@click.option("--data", required=True,
              help="Path to expression matrix CSV (samples × genes) or AnnData .h5ad file.")
@click.option(
    "--preset",
    default=None,
    type=click.Choice(["bulk_rnaseq", "scrna_pseudobulk", "proteomics",
                       "methylation", "differential"]),
    help="Apply a named preset (sets method/npn/fdr/alpha defaults; "
         "individual options override preset values).",
)
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
@click.option("--use-hvg", is_flag=True, default=False,
              help="Subset to highly variable genes (requires .h5ad input).")
@click.option("--layer", default=None,
              help="AnnData layer to extract (requires .h5ad input; default: .X).")
@click.option("--output-anndata", is_flag=True, default=False,
              help="Write results back into the .h5ad file (requires .h5ad input). "
                   "Populates adata.obsp['nodis_connectivities'] and adata.uns['nodis'].")
@click.option("--pseudobulk-groupby", multiple=True, default=(),
              help="obs columns for pseudobulk aggregation before fitting "
                   "(repeat for multiple, e.g. --pseudobulk-groupby sample_id "
                   "--pseudobulk-groupby cell_type). Requires .h5ad input.")
@click.option("--report", is_flag=True, default=False,
              help="Generate HTML report after inference (written to <out>/report.html).")
@click.option("--prior-string", is_flag=True, default=False,
              help="Build STRING prior and use PIGLassoEstimator. "
                   "Gene names taken from column headers of the expression matrix.")
@click.option("--prior-organism", default=9606, show_default=True,
              help="NCBI taxon ID for STRING (default 9606 = human).")
@click.option("--prior-score-threshold", default=400, show_default=True,
              help="STRING combined score threshold (default 400 = medium confidence).")
@click.option("--out", default="results/", show_default=True, help="Output directory.")
def run(
    ctx,
    data: str, preset: str | None, method: str, alpha: float, fdr: str, npn: bool,
    use_hvg: bool, layer: str | None, output_anndata: bool,
    pseudobulk_groupby: tuple,
    report: bool,
    prior_string: bool, prior_organism: int, prior_score_threshold: int,
    out: str,
) -> None:
    """Run GGM inference on an expression matrix.

    Accepts a CSV file (samples × genes) or an AnnData .h5ad file.
    With --output-anndata the FDR graph is written back into the .h5ad as
    adata.obsp['nodis_connectivities'] in the format expected by scanpy/squidpy.
    """
    # Apply preset defaults for any options the user did not supply explicitly
    if preset is not None:
        from nodis.advisor import PRESETS
        from click.core import ParameterSource
        preset_cfg = PRESETS[preset]
        if ctx.get_parameter_source("method") == ParameterSource.DEFAULT:
            method = preset_cfg["method"]
        if ctx.get_parameter_source("alpha") == ParameterSource.DEFAULT:
            alpha = preset_cfg["alpha"]
        if ctx.get_parameter_source("fdr") == ParameterSource.DEFAULT:
            fdr = preset_cfg["fdr"]
        if ctx.get_parameter_source("npn") == ParameterSource.DEFAULT:
            npn = preset_cfg["npn"]

    out_dir = pathlib.Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)

    data_path = pathlib.Path(data)
    stem = data_path.stem
    is_h5ad = data_path.suffix.lower() == ".h5ad"

    if output_anndata and not is_h5ad:
        raise click.UsageError("--output-anndata requires --data to be an .h5ad file.")
    if pseudobulk_groupby and not is_h5ad:
        raise click.UsageError("--pseudobulk-groupby requires --data to be an .h5ad file.")

    # ----------------------------------------------------------------
    # Load expression matrix
    # ----------------------------------------------------------------
    adata = None
    if is_h5ad:
        try:
            import anndata as ad
        except ImportError:
            raise click.ClickException(
                "anndata is not installed. Install it with: pip install anndata"
            )
        adata = ad.read_h5ad(data)
        if pseudobulk_groupby:
            from nodis.preprocess.pseudobulk import aggregate_pseudobulk
            n_cells = adata.n_obs
            adata = aggregate_pseudobulk(
                adata, groupby=list(pseudobulk_groupby), layer=layer,
            )
            click.echo(
                f"Pseudobulk: {n_cells} cells → {adata.n_obs} groups "
                f"(groupby: {', '.join(pseudobulk_groupby)})"
            )
        from nodis.preprocess.anndata_compat import from_anndata
        X = from_anndata(adata, layer=layer, use_hvg=use_hvg, npn=npn)
        click.echo(
            f"Loaded AnnData: {adata.n_obs} obs × {adata.n_vars} vars"
            + (f" → {X.shape[1]} HVGs" if use_hvg else "")
        )
    else:
        X = pd.read_csv(data, index_col=0).values.astype(float)
        click.echo(f"Loaded CSV: {X.shape[0]} samples × {X.shape[1]} genes")
        if npn:
            from nodis.preprocess.npn import npn_shrinkage
            X = npn_shrinkage(X)
            click.echo("Applied NPN preprocessing.")

    # ----------------------------------------------------------------
    # STRING prior (forces method=piglasso when requested)
    # ----------------------------------------------------------------
    prior_matrix = None
    if prior_string:
        if method not in ("desparsified", "glasso", "gglasso") or prior_string:
            if method != "piglasso":
                click.echo(f"--prior-string forces method=piglasso (was '{method}').")
                method = "piglasso"
        from nodis.priors.string_prior import prior_from_string
        # Gene names come from adata.var_names (h5ad) or CSV column headers
        if is_h5ad:
            gene_names = list(adata.var_names)
        else:
            gene_names = list(pd.read_csv(data, index_col=0, nrows=0).columns)
        click.echo(
            f"Querying STRING for {len(gene_names)} genes "
            f"(organism={prior_organism}, threshold={prior_score_threshold})..."
        )
        prior_matrix = prior_from_string(
            gene_names, organism=prior_organism,
            score_threshold=prior_score_threshold,
        )
        click.echo("STRING prior matrix built.")

    # ----------------------------------------------------------------
    # Fit estimator
    # ----------------------------------------------------------------
    if method == "desparsified":
        from nodis.estimators.desparsified import DesparifiedGGM
        est = DesparifiedGGM()
        est.fit(X)
        adj = est.get_adjacency(alpha=alpha, method=fdr)
        pd.DataFrame(est.result_.p_values).to_csv(out_dir / f"{stem}_pvalues.csv", index=False)
        pd.DataFrame(est.result_.z_scores).to_csv(out_dir / f"{stem}_zscores.csv", index=False)
        pd.DataFrame(adj).to_csv(out_dir / f"{stem}_adjacency.csv", index=False)

        if output_anndata:
            from nodis.preprocess.anndata_compat import to_anndata
            to_anndata(adata, est.result_)
            out_h5ad = out_dir / f"{stem}_nodis.h5ad"
            adata.write_h5ad(out_h5ad)
            n_edges = est.result_.adj_fdr.sum() // 2
            click.echo(
                f"Wrote AnnData with {n_edges} edges → {out_h5ad} "
                "(adata.obsp['nodis_connectivities'], adata.uns['nodis'])"
            )

    elif method == "glasso":
        from nodis.estimators.glasso import SklearnGLasso
        est = SklearnGLasso()
        est.fit(X)
        adj = est.get_adjacency()
        pd.DataFrame(adj).to_csv(out_dir / f"{stem}_adjacency.csv", index=False)

    elif method == "gglasso":
        from nodis.estimators.glasso import GGLassoEstimator
        est = GGLassoEstimator()
        est.fit(X)
        adj = est.get_adjacency()
        pd.DataFrame(adj).to_csv(out_dir / f"{stem}_adjacency.csv", index=False)

    click.echo(f"Results written to {out_dir}")

    if report and method == "desparsified":
        from nodis.report.generator import generate_report
        if is_h5ad and adata is not None:
            g_names = list(adata.var_names)
        else:
            g_names = [str(i) for i in range(X.shape[1])]
        report_path = out_dir / "report.html"
        generate_report(est.result_, g_names, out_path=report_path, title=stem)
        click.echo(f"Report written to {report_path}")


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


# ---------------------------------------------------------------------------
# diff
# ---------------------------------------------------------------------------

@main.command("diff")
@click.option("--cond1", required=True, help="Condition 1 data (.h5ad or CSV).")
@click.option("--cond2", required=True, help="Condition 2 data (.h5ad or CSV).")
@click.option("--method", default="desparsified_test", show_default=True,
              type=click.Choice(["desparsified_test", "fused_glasso"]),
              help="Differential analysis method.")
@click.option("--alpha", default=0.05, show_default=True, help="FDR level.")
@click.option("--fdr", default="BH", show_default=True,
              type=click.Choice(["BH", "BY"]), help="FDR procedure.")
@click.option("--out", default="results/diff/", show_default=True,
              help="Output directory.")
def diff_cmd(cond1, cond2, method, alpha, fdr, out):
    """Compare GGMs across two conditions (differential network analysis)."""
    import pathlib as _pl
    from nodis.compare.differential import DifferentialNetwork

    out_dir = _pl.Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)

    def _load(path):
        p = _pl.Path(path)
        if p.suffix.lower() == ".h5ad":
            import anndata as ad
            from nodis.preprocess.anndata_compat import from_anndata
            return from_anndata(ad.read_h5ad(p))
        return pd.read_csv(p, index_col=0).values.astype(float)

    X1 = _load(cond1)
    X2 = _load(cond2)

    click.echo(f"Condition 1: {X1.shape[0]} × {X1.shape[1]}")
    click.echo(f"Condition 2: {X2.shape[0]} × {X2.shape[1]}")

    result = DifferentialNetwork(X1, X2, method=method, alpha=alpha, fdr=fdr).fit()

    pd.DataFrame(result.adj_cond1).to_csv(out_dir / "diff_cond1_adjacency.csv", index=False)
    pd.DataFrame(result.adj_cond2).to_csv(out_dir / "diff_cond2_adjacency.csv", index=False)
    pd.DataFrame(result.adj_shared).to_csv(out_dir / "diff_shared.csv", index=False)
    pd.DataFrame(result.adj_cond1_only).to_csv(out_dir / "diff_cond1_only.csv", index=False)
    pd.DataFrame(result.adj_cond2_only).to_csv(out_dir / "diff_cond2_only.csv", index=False)

    if result.adj_diff_fdr is not None:
        pd.DataFrame(result.adj_diff_fdr).to_csv(out_dir / "diff_differential.csv", index=False)
        pd.DataFrame(result.p_values_diff).to_csv(out_dir / "diff_pvalues.csv", index=False)

    summary_lines = [
        f"Shared edges      : {result.n_shared}",
        f"Cond1-only edges  : {result.n_cond1_only}",
        f"Cond2-only edges  : {result.n_cond2_only}",
    ]
    if result.adj_diff_fdr is not None:
        n_diff = int(np.triu(result.adj_diff_fdr, k=1).sum())
        summary_lines.append(f"Differential edges: {n_diff}")

    summary_path = out_dir / "diff_summary.txt"
    summary_path.write_text("\n".join(summary_lines) + "\n")
    click.echo("\n".join(summary_lines))
    click.echo(f"\nResults written to {out_dir}")


# ---------------------------------------------------------------------------
# validate
# ---------------------------------------------------------------------------

@main.command("validate")
@click.option("--adjacency", required=True, help="Inferred adjacency CSV.")
@click.option("--genes", required=True,
              help="Gene list CSV (one column, no header) matching adjacency row/col order.")
@click.option("--organism", default=9606, show_default=True, help="NCBI taxon ID.")
@click.option("--threshold", default=400, show_default=True,
              help="STRING combined score threshold.")
@click.option("--out", default="results/validation.csv", show_default=True)
def validate_cmd(adjacency, genes, organism, threshold, out):
    """Validate inferred network against STRING interactions."""
    import csv
    from nodis.validate.string_validator import validate_against_string

    adj = pd.read_csv(adjacency, index_col=None, header=0).values.astype(int)
    with open(genes) as fh:
        gene_names = [row[0] for row in csv.reader(fh) if row]
    result = validate_against_string(adj, gene_names, organism=organism,
                                     score_threshold=threshold)
    pd.DataFrame([{
        "n_inferred_edges": result.n_inferred_edges,
        "n_string_edges": result.n_string_edges,
        "n_overlap": result.n_overlap,
        "precision": result.precision,
        "recall": result.recall,
        "f1": result.f1,
        "jaccard": result.jaccard,
    }]).to_csv(out, index=False)
    click.echo(f"Validation results written to {out}")
    click.echo(f"  Precision: {result.precision:.4f}  Recall: {result.recall:.4f}"
               f"  F1: {result.f1:.4f}  Jaccard: {result.jaccard:.4f}")


# ---------------------------------------------------------------------------
# wizard
# ---------------------------------------------------------------------------

@main.command()
@click.option("--n", required=True, type=int, help="Number of samples.")
@click.option("--p", required=True, type=int, help="Number of genes / features.")
@click.option("--goal", default="edge_pvalues", show_default=True,
              type=click.Choice(["edge_pvalues", "network_structure",
                                 "module_detection", "differential"]),
              help="Analysis goal.")
@click.option("--data-type", default="bulk_rnaseq", show_default=True,
              type=click.Choice(["bulk_rnaseq", "scrna_seq", "proteomics",
                                 "methylation", "generic"]),
              help="Input data modality.")
@click.option("--conditions", default=1, show_default=True,
              help="Number of experimental conditions (≥2 enables differential mode).")
@click.option("--has-prior", is_flag=True, default=False,
              help="Prior network knowledge available (STRING, BioGRID, etc.).")
def wizard(n, p, goal, data_type, conditions, has_prior):
    """Interactive estimator advisor — prints a recommended pipeline."""
    from nodis.advisor import recommend
    result = recommend(n=n, p=p, goal=goal, data_type=data_type,
                       n_conditions=conditions, has_prior=has_prior)

    click.echo("\n=== NODIS Advisor ===\n")
    click.echo(f"Recommended estimator : {result.estimator}")
    click.echo(f"Preset                : {result.preset}")
    click.echo("\nReasoning:")
    for line in result.reasoning:
        click.echo(f"  • {line}")
    if result.warnings:
        click.echo("\nWarnings:")
        for w in result.warnings:
            click.echo(f"  ! {w}")
    click.echo("\n--- CLI ---")
    click.echo(result.cli_snippet)
    click.echo("\n--- Python ---")
    click.echo(result.python_snippet)


# ---------------------------------------------------------------------------
# pseudobulk
# ---------------------------------------------------------------------------

@main.command()
@click.option("--data", required=True, help=".h5ad file (per-cell).")
@click.option("--groupby", required=True, multiple=True,
              help="obs columns to group by (repeat for multiple, e.g. "
                   "--groupby sample_id --groupby cell_type).")
@click.option("--layer", default=None, help="Layer to aggregate (default: .X).")
@click.option("--agg", default="sum", show_default=True,
              type=click.Choice(["sum", "mean"]), help="Aggregation function.")
@click.option("--min-cells", default=10, show_default=True,
              help="Min cells per group; groups below this are dropped.")
@click.option("--out", required=True, help="Output .h5ad path.")
def pseudobulk(data, groupby, layer, agg, min_cells, out):
    """Aggregate per-cell AnnData to pseudobulk samples for GGM inference."""
    import anndata as ad
    from nodis.preprocess.pseudobulk import aggregate_pseudobulk
    adata = ad.read_h5ad(data)
    pb = aggregate_pseudobulk(adata, groupby=list(groupby), layer=layer,
                               agg=agg, min_cells=min_cells)
    pb.write_h5ad(out)
    click.echo(f"Pseudobulk: {adata.n_obs} cells → {pb.n_obs} groups → {out}")


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------

@main.command("report")
@click.option("--pvalues", required=True, type=click.Path(exists=True),
              help="Edge p-value matrix CSV.")
@click.option("--adjacency", required=True, type=click.Path(exists=True),
              help="Binary adjacency matrix CSV.")
@click.option("--genes", required=True, type=click.Path(exists=True),
              help="Gene list (one per line, no header).")
@click.option("--title", default="NODIS Inference Report", show_default=True)
@click.option("--out", default="nodis_report.html", show_default=True,
              help="Output HTML path.")
def report_cmd(pvalues, adjacency, genes, title, out):
    """Generate a standalone HTML report from saved inference outputs."""
    import types
    from nodis.report.generator import generate_report

    with open(genes) as fh:
        gene_names = [line.strip() for line in fh if line.strip()]

    pv_mat = pd.read_csv(pvalues, index_col=None, header=0).values.astype(float)
    adj_mat = pd.read_csv(adjacency, index_col=None, header=0).values.astype(int)

    # Build a minimal result object from saved CSVs
    result = types.SimpleNamespace(
        p_values=pv_mat,
        adj_fdr=adj_mat,
        fdr_alpha=None,
        precision=np.zeros_like(pv_mat),
        variance=np.zeros_like(pv_mat),
    )
    out_path = generate_report(result, gene_names, out_path=out, title=title)
    click.echo(f"Report written to {out_path}")


# ---------------------------------------------------------------------------
# network
# ---------------------------------------------------------------------------

@main.command("network")
@click.option("--adjacency", required=True, type=click.Path(exists=True),
              help="Path to adjacency matrix (.csv or .npy). "
                   "CSV: square, no header; .npy: (p, p) binary array.")
@click.option("--genes", "genes_path", default=None, type=click.Path(exists=True),
              help="Text file with one gene name per line (optional). "
                   "If omitted, integer indices are used.")
@click.option("--weights", "weights_path", default=None, type=click.Path(exists=True),
              help="Path to edge weight matrix (.csv or .npy), e.g. absolute "
                   "precision entries |Omega_hat|. Required for disparity backbone.")
@click.option("--communities", "community_algo", default="louvain", show_default=True,
              type=click.Choice(["leiden", "louvain", "greedy", "spectral", "none"]),
              help="Community detection algorithm. 'leiden' requires leidenalg+igraph.")
@click.option("--resolution", default=1.0, show_default=True,
              help="Resolution parameter for Leiden/Louvain.")
@click.option("--n-clusters", default=5, show_default=True,
              help="Target number of clusters for spectral clustering.")
@click.option("--hubs", is_flag=True, default=False,
              help="Run hub gene identification with permutation test.")
@click.option("--hub-metric", default="degree", show_default=True,
              type=click.Choice(["degree", "betweenness", "eigenvector", "strength"]),
              help="Centrality metric for hub identification.")
@click.option("--hub-permutations", default=500, show_default=True,
              help="Number of permutations for hub significance test.")
@click.option("--hub-alpha", default=0.05, show_default=True,
              help="Significance threshold for hub p-values.")
@click.option("--backbone", "backbone_method", default="none", show_default=True,
              type=click.Choice(["disparity", "threshold", "none"]),
              help="Backbone extraction method.")
@click.option("--backbone-alpha", default=0.05, show_default=True,
              help="Alpha for disparity filter backbone.")
@click.option("--backbone-threshold", default=0.1, show_default=True,
              help="Edge weight cut-off for threshold backbone.")
@click.option("--seed", default=42, show_default=True, help="Random seed.")
@click.option("--out", default="results/network/", show_default=True,
              help="Output directory.")
def network_cmd(
    adjacency, genes_path, weights_path,
    community_algo, resolution, n_clusters,
    hubs, hub_metric, hub_permutations, hub_alpha,
    backbone_method, backbone_alpha, backbone_threshold,
    seed, out,
):
    """Network topology analysis: community detection, hub genes, backbone.

    \b
    Example (communities + hubs on a NODIS adjacency):
      nodis network --adjacency adj.csv --genes genes.txt \\
          --communities louvain --hubs --hub-metric degree \\
          --backbone disparity --weights precision.csv --out results/network/
    """
    import pathlib as _pl
    from nodis.network import NetworkTopology, write_anndata_network

    out_dir = _pl.Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load adjacency
    adj_path = _pl.Path(adjacency)
    if adj_path.suffix.lower() == ".npy":
        adj_mat = np.load(adj_path)
    else:
        adj_mat = pd.read_csv(adj_path, header=0, index_col=None).values
    adj_mat = adj_mat.astype(float)

    # Load weights (optional)
    w_mat = None
    if weights_path is not None:
        wp = _pl.Path(weights_path)
        if wp.suffix.lower() == ".npy":
            w_mat = np.load(wp).astype(float)
        else:
            w_mat = pd.read_csv(wp, header=0, index_col=None).values.astype(float)

    # Load gene names (optional)
    gene_names = None
    if genes_path is not None:
        with open(genes_path) as fh:
            gene_names = [line.strip() for line in fh if line.strip()]

    p = adj_mat.shape[0]
    click.echo(f"Loaded adjacency: {p} × {p}")
    if gene_names:
        click.echo(f"Gene names: {len(gene_names)} genes")

    nt = NetworkTopology(adj_mat, gene_names=gene_names, weights=w_mat)
    summary = nt.summary()
    click.echo(
        f"Graph: {summary['nodes']} nodes, {summary['edges']} edges, "
        f"density={summary['density']:.4f}, "
        f"components={summary['n_components']}"
    )

    # Community detection
    comm_result = None
    if community_algo != "none":
        click.echo(f"Detecting communities ({community_algo})...")
        comm_result = nt.detect_communities(
            algorithm=community_algo,
            resolution=resolution,
            seed=seed,
            n_clusters=n_clusters,
        )
        click.echo(
            f"  {comm_result.n_communities} communities found  "
            f"(modularity={comm_result.modularity:.4f}, "
            f"algorithm={comm_result.algorithm})"
        )
        comm_df = comm_result.as_dataframe()
        comm_df.to_csv(out_dir / "communities.csv", index=False)
        click.echo(f"  Communities written to {out_dir / 'communities.csv'}")

    # Hub genes
    hub_result = None
    if hubs:
        click.echo(
            f"Identifying hub genes ({hub_metric}, "
            f"{hub_permutations} permutations, α={hub_alpha})..."
        )
        hub_result = nt.hub_genes(
            metric=hub_metric,
            n_permutations=hub_permutations,
            alpha=hub_alpha,
            seed=seed,
        )
        click.echo(f"  {hub_result.n_hubs} significant hub genes found.")
        hub_result.scores.to_csv(out_dir / "hub_genes.csv", index=False)
        click.echo(f"  Hub scores written to {out_dir / 'hub_genes.csv'}")

    # Backbone
    bb_adj = None
    if backbone_method != "none":
        click.echo(f"Extracting backbone ({backbone_method})...")
        bb_adj = nt.backbone(
            method=backbone_method,
            alpha=backbone_alpha,
            threshold=backbone_threshold,
        )
        n_backbone_edges = int(np.triu(bb_adj, k=1).sum())
        click.echo(f"  Backbone: {n_backbone_edges} edges retained.")
        np.save(out_dir / "backbone_adjacency.npy", bb_adj)
        pd.DataFrame(bb_adj).to_csv(out_dir / "backbone_adjacency.csv", index=False)
        click.echo(f"  Backbone written to {out_dir / 'backbone_adjacency.npy'}")

    # Summary JSON
    import json
    summary_out = {
        "nodes": summary["nodes"],
        "edges": summary["edges"],
        "density": float(summary["density"]),
        "n_components": summary["n_components"],
    }
    if comm_result is not None:
        summary_out["communities"] = {
            "n_communities": comm_result.n_communities,
            "modularity": float(comm_result.modularity),
            "algorithm": comm_result.algorithm,
        }
    if hub_result is not None:
        summary_out["hubs"] = {
            "n_significant": hub_result.n_hubs,
            "metric": hub_result.metric,
            "alpha": hub_result.alpha,
        }
    if bb_adj is not None:
        summary_out["backbone_edges"] = int(np.triu(bb_adj, k=1).sum())

    with open(out_dir / "network_summary.json", "w") as fh:
        json.dump(summary_out, fh, indent=2)
    click.echo(f"\nSummary written to {out_dir / 'network_summary.json'}")
    click.echo(f"All outputs in {out_dir}")


# ---------------------------------------------------------------------------
# multi
# ---------------------------------------------------------------------------

@main.command("multi")
@click.option("--data", required=True, type=click.Path(exists=True),
              help=".h5ad file (AnnData) or a CSV glob pattern like "
                   "'data/cond*.csv' (one CSV per condition, filename = condition name).")
@click.option("--condition-key", default=None,
              help="adata.obs column to split conditions (.h5ad only).")
@click.option("--reg", default="GGL", show_default=True,
              type=click.Choice(["GGL", "FGL"]),
              help="Group (GGL) or Fused (FGL) Graphical Lasso.")
@click.option("--lambda1", default=None, type=float,
              help="Sparsity penalty λ₁. Omit to run eBIC model selection.")
@click.option("--lambda2", default=None, type=float,
              help="Group/fusion penalty λ₂. Omit to run eBIC model selection.")
@click.option("--ebic-gamma", default=0.1, show_default=True,
              help="eBIC γ (0=BIC; larger → sparser selection).")
@click.option("--threshold", default=0.0, show_default=True,
              help="Adjacency threshold on |precision| (default 0 = GGLasso support).")
@click.option("--npn", is_flag=True, default=False,
              help="Apply NPN shrinkage per condition before fitting.")
@click.option("--layer", default=None,
              help="AnnData layer to extract (.h5ad only).")
@click.option("--use-hvg", is_flag=True, default=False,
              help="Subset to highly variable genes (.h5ad only).")
@click.option("--out", default="results/multi/", show_default=True,
              help="Output directory.")
def multi_cmd(data, condition_key, reg, lambda1, lambda2, ebic_gamma,
              threshold, npn, layer, use_hvg, out):
    """Multi-condition Group / Fused Graphical Lasso.

    \b
    Examples:
      # GGL from AnnData split by 'treatment' column:
      nodis multi --data expr.h5ad --condition-key treatment --reg GGL \\
          --lambda1 0.1 --lambda2 0.05 --out results/multi/

      # FGL with eBIC model selection from per-condition CSVs:
      nodis multi --data 'data/cond*.csv' --reg FGL --out results/multi/
    """
    import pathlib as _pl
    import glob
    from nodis.estimators.group_glasso import MultiConditionGLasso

    out_dir = _pl.Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)

    data_path = _pl.Path(data)

    # ----------------------------------------------------------------
    # Load expression matrices
    # ----------------------------------------------------------------
    is_h5ad = data_path.suffix.lower() == ".h5ad"

    if is_h5ad:
        if condition_key is None:
            raise click.UsageError(
                "--condition-key is required when --data is an .h5ad file."
            )
        try:
            import anndata as ad
        except ImportError:
            raise click.ClickException("anndata is not installed: pip install anndata")
        adata = ad.read_h5ad(data_path)
        from nodis.preprocess.anndata_compat import from_anndata
        conditions = adata.obs[condition_key].unique()
        X_dict = {}
        for cond in conditions:
            mask = (adata.obs[condition_key] == cond).values
            X_dict[str(cond)] = from_anndata(
                adata[mask], layer=layer, use_hvg=use_hvg, npn=False,
            )
    else:
        # Glob for per-condition CSVs
        csv_files = sorted(glob.glob(str(data_path)))
        if not csv_files:
            raise click.UsageError(
                f"No files matched the pattern: {data}"
            )
        X_dict = {}
        for csv_f in csv_files:
            cond = _pl.Path(csv_f).stem
            X_dict[cond] = pd.read_csv(csv_f, index_col=0).values.astype(float)
        adata = None

    click.echo(
        f"Loaded {len(X_dict)} conditions: "
        + ", ".join(f"{c} (n={X.shape[0]})" for c, X in X_dict.items())
    )

    # ----------------------------------------------------------------
    # Fit
    # ----------------------------------------------------------------
    label = "eBIC" if (lambda1 is None or lambda2 is None) else f"λ₁={lambda1}, λ₂={lambda2}"
    click.echo(f"Fitting {reg} ({label}) ...")

    est = MultiConditionGLasso(
        reg=reg,
        lambda1=lambda1,
        lambda2=lambda2,
        ebic_gamma=ebic_gamma,
        threshold=threshold,
        npn=npn,
    )
    est.fit(X_dict)
    r = est.result_

    click.echo(
        f"Done. λ₁={r.lambda1_:.4f}, λ₂={r.lambda2_:.4f} "
        f"(eBIC={'yes' if r.ebic_selected else 'no'})"
    )
    click.echo(f"Shared edges: {r.n_shared_edges}")
    for c in r.condition_names:
        click.echo(f"  {c}: {r.n_edges(c)} edges")

    # ----------------------------------------------------------------
    # Write outputs
    # ----------------------------------------------------------------
    for c in r.condition_names:
        safe = c.replace(" ", "_")
        pd.DataFrame(r.precision_[c]).to_csv(
            out_dir / f"precision_{safe}.csv", index=False)
        pd.DataFrame(r.adjacency_[c]).to_csv(
            out_dir / f"adjacency_{safe}.csv", index=False)
        pd.DataFrame(r.unique_adjacency[c]).to_csv(
            out_dir / f"unique_{safe}.csv", index=False)

    pd.DataFrame(r.shared_adjacency).to_csv(
        out_dir / "adjacency_shared.csv", index=False)

    import json
    with open(out_dir / "multi_summary.json", "w") as fh:
        json.dump(r.summary(), fh, indent=2)

    if is_h5ad and adata is not None:
        est.to_anndata(adata, key="nodis_mgl")
        out_h5ad = out_dir / f"{data_path.stem}_mgl.h5ad"
        adata.write_h5ad(out_h5ad)
        click.echo(f"AnnData written to {out_h5ad}")

    click.echo(f"All outputs in {out_dir}")


if __name__ == "__main__":
    main()
