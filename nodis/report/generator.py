"""
HTML run report generator.

generate_report(result, gene_names, out_path, title) → pathlib.Path

Produces a single-file HTML report with embedded base64 figures.
No external CSS/JS dependencies — fully self-contained.

Sections
--------
1. Run summary       (estimator, n, p, n/p, λ, runtime, nodis version)
2. Network overview  (n edges, density, largest component, isolates)
3. P-value calibration  (histogram of all edge p-values)
4. Degree distribution
5. Top hub genes     (table: gene, degree, top 5 neighbours)
6. GO enrichment     (top 20 terms if enrichment was run)
7. QC flags          (warnings from advisor if applicable)
"""
from __future__ import annotations

import base64
import io
import pathlib
from datetime import datetime
from typing import Optional

import numpy as np


def generate_report(
    result,
    gene_names: list[str],
    out_path: str | pathlib.Path = "nodis_report.html",
    title: str = "NODIS Inference Report",
    enrichment_result=None,
    advisor_result=None,
) -> pathlib.Path:
    """Generate a self-contained HTML report.

    Parameters
    ----------
    result          : GGMInferenceResult (must have p_values and adj_fdr set)
    gene_names      : list of gene names matching result dimensions
    out_path        : output HTML file path
    title           : report title
    enrichment_result : optional EnrichmentResult (from nodis.enrich)
    advisor_result  : optional AdvisorResult (from nodis.advisor)
    """
    try:
        from jinja2 import Environment, FileSystemLoader
    except ImportError:
        raise ImportError(
            "jinja2 is required for report generation. "
            "Install it with: pip install jinja2"
        )

    import nodis

    out_path = pathlib.Path(out_path)
    template_dir = pathlib.Path(__file__).parent / "templates"
    env = Environment(loader=FileSystemLoader(str(template_dir)), autoescape=True)
    template = env.get_template("report.html.j2")

    p = len(gene_names)
    adj = _get_adj(result)
    p_values = _get_pvalues(result)

    # --- 1. Run summary ---
    n = p_values.shape[0]
    n_edges = int(np.triu(adj, k=1).sum()) if adj is not None else None
    summary = _build_summary(result, n, p, nodis.__version__)

    # --- 2. Network overview ---
    network = _network_stats(adj, p)

    # --- 3. & 4. Figures ---
    figures = {}
    if p_values is not None:
        figures["pvalue_hist"] = _pvalue_histogram(p_values)
    if adj is not None:
        figures["degree_dist"] = _degree_distribution(adj)

    # --- 5. Hub genes ---
    hub_genes = _hub_table(adj, gene_names, top_n=20)

    # --- 6. Enrichment ---
    enrichment_ctx = None
    if enrichment_result is not None:
        enrichment_ctx = _enrichment_context(enrichment_result)

    # --- 7. Warnings ---
    warnings: list[str] = []
    if advisor_result is not None:
        warnings = list(advisor_result.warnings)

    html = template.render(
        title=title,
        generated_at=datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
        nodis_version=nodis.__version__,
        summary=summary,
        network=network,
        figures=figures,
        hub_genes=hub_genes,
        enrichment=enrichment_ctx,
        warnings=warnings,
    )

    out_path.write_text(html, encoding="utf-8")
    return out_path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_adj(result) -> np.ndarray | None:
    adj = getattr(result, "adj_fdr", None)
    if adj is None:
        return None
    if hasattr(adj, "toarray"):
        adj = adj.toarray()
    return np.asarray(adj)


def _get_pvalues(result) -> np.ndarray | None:
    pv = getattr(result, "p_values", None)
    if pv is None:
        return None
    if hasattr(pv, "toarray"):
        pv = pv.toarray()
    return np.asarray(pv)


def _build_summary(result, n: int, p: int, version: str) -> dict:
    summary = {
        "NODIS version": version,
        "Dimensions (n × p)": f"{n} × {p}",
        "n/p ratio": f"{n / p:.2f}",
    }
    if hasattr(result, "fdr_alpha") and result.fdr_alpha is not None:
        summary["FDR α"] = str(result.fdr_alpha)
    return summary


class _NetworkStats:
    def __init__(self, n_edges, density, n_components, n_isolates):
        self.n_edges = n_edges
        self.density = density
        self.n_components = n_components
        self.n_isolates = n_isolates


def _network_stats(adj: np.ndarray | None, p: int) -> _NetworkStats:
    if adj is None or adj.sum() == 0:
        return _NetworkStats(0, 0.0, p, p)

    n_edges = int(np.triu(adj, k=1).sum())
    max_edges = p * (p - 1) // 2
    density = n_edges / max_edges if max_edges > 0 else 0.0

    try:
        import networkx as nx
        G = nx.from_numpy_array(adj)
        n_components = nx.number_connected_components(G)
        n_isolates = sum(1 for node in G.nodes() if G.degree(node) == 0)
    except ImportError:
        n_components = -1
        n_isolates = -1

    return _NetworkStats(n_edges, density, n_components, n_isolates)


def _pvalue_histogram(p_values: np.ndarray) -> str:
    """Return base64-encoded PNG of upper-triangle p-value histogram."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    uidx = np.triu_indices(p_values.shape[0], k=1)
    pv = p_values[uidx]

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.hist(pv, bins=50, color="#2c7bb6", edgecolor="white", linewidth=0.4)
    ax.set_xlabel("p-value")
    ax.set_ylabel("Count")
    ax.set_title("Edge p-value distribution")
    ax.set_xlim(0, 1)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def _degree_distribution(adj: np.ndarray) -> str:
    """Return base64-encoded PNG of node degree distribution."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    degrees = adj.sum(axis=1).astype(int)

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.hist(degrees, bins=max(10, int(degrees.max()) + 1),
            color="#d7191c", edgecolor="white", linewidth=0.4)
    ax.set_xlabel("Degree")
    ax.set_ylabel("Count")
    ax.set_title("Node degree distribution")
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def _hub_table(adj: np.ndarray | None, gene_names: list[str], top_n: int = 20):
    if adj is None:
        return []

    degrees = adj.sum(axis=1).astype(int)
    order = np.argsort(degrees)[::-1][:top_n]

    rows = []
    for i in (int(idx) for idx in order):
        degree = int(degrees[i])
        if degree == 0:
            break
        neighbours = np.where(adj[i] > 0)[0]
        top5 = [gene_names[j] for j in neighbours[:5]]
        rows.append({
            "gene": gene_names[i],
            "degree": degree,
            "top_neighbours": ", ".join(top5),
        })
    return rows


def _enrichment_context(enrichment_result):
    """Convert EnrichmentResult list to template-friendly dict."""
    try:
        rows = []
        for hit in enrichment_result:
            if hasattr(hit, "results") and hit.results is not None:
                df = hit.results
                for _, row in df.head(20).iterrows():
                    rows.append({
                        "term": row.get("name", row.get("term_name", "?")),
                        "source": row.get("source", hit.gene_set_name),
                        "pvalue": float(row.get("p_value", row.get("pval", 1.0))),
                    })
        return {
            "summary": f"{len(rows)} significant terms found.",
            "table": rows[:20],
        }
    except Exception:
        return {"summary": "Enrichment results available (could not format).", "table": []}
