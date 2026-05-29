"""
Gene extraction utilities — convert GGM inference output into enrichment-ready
gene lists and ranked gene series.

Three extraction strategies
---------------------------
hub_genes       : Top-N genes by network centrality (degree, betweenness,
                  eigenvector) from an FDR-controlled adjacency matrix.
ranked_genes    : All genes ranked by an aggregated edge-evidence score;
                  suitable for pre-ranked GSEA input.
community_gene_sets : Dictionary of community label → gene list using
                  networkx community detection on the inferred network.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import networkx as nx

_VALID_CENTRALITY = ("degree", "betweenness", "eigenvector")
_VALID_RANK_METHODS = ("min_pvalue", "degree", "fisher_combined")


def hub_genes(
    adj: np.ndarray,
    gene_names: list[str],
    centrality: str = "degree",
    quantile: float = 0.9,
) -> list[str]:
    """Return hub genes above the given centrality quantile.

    Parameters
    ----------
    adj : (p, p) ndarray
        Binary symmetric adjacency matrix (e.g. ``GGMInferenceResult.adj_fdr``).
    gene_names : list of str
        Gene identifiers, length p.
    centrality : str, default ``"degree"``
        Node centrality measure: ``"degree"``, ``"betweenness"``, or
        ``"eigenvector"``.
    quantile : float, default 0.9
        Genes at or above this quantile of centrality scores are returned.

    Returns
    -------
    list of str
        Gene names of hub nodes.

    Raises
    ------
    ValueError
        If ``centrality`` is not a recognised measure.
    """
    if centrality not in _VALID_CENTRALITY:
        raise ValueError(
            f"centrality must be one of {_VALID_CENTRALITY}; got '{centrality}'."
        )

    G = nx.from_numpy_array(adj)
    mapping = {i: name for i, name in enumerate(gene_names)}
    G = nx.relabel_nodes(G, mapping)

    if centrality == "degree":
        scores = {k: float(v) for k, v in dict(G.degree()).items()}
    elif centrality == "betweenness":
        scores = nx.betweenness_centrality(G, normalized=True)
    else:  # eigenvector
        try:
            scores = nx.eigenvector_centrality(G, max_iter=1000, tol=1e-6)
        except nx.PowerIterationFailedConvergence:
            # Fall back to degree when eigenvector fails (disconnected graph)
            scores = {k: float(v) for k, v in dict(G.degree()).items()}

    score_series = pd.Series(scores)
    threshold = score_series.quantile(quantile)
    return sorted(score_series[score_series >= threshold].index.tolist())


def ranked_genes(
    adj: np.ndarray,
    gene_names: list[str],
    p_values: np.ndarray,
    method: str = "min_pvalue",
) -> pd.Series:
    """Return a ranked gene series for pre-ranked GSEA.

    Each gene receives a score aggregated from its incident edge p-values.
    Higher score = stronger evidence of network involvement.

    Parameters
    ----------
    adj : (p, p) ndarray
        Binary symmetric adjacency matrix (significant edges only).
    gene_names : list of str
        Gene identifiers, length p.
    p_values : (p, p) ndarray
        Edge-level two-sided p-values.
    method : str, default ``"min_pvalue"``
        Scoring method:

        ``"min_pvalue"``
            Score = −log10(min significant edge p-value) per gene.
            Genes with no significant edges get score 0.
        ``"degree"``
            Score = node degree (number of significant edges).
        ``"fisher_combined"``
            Score = Fisher's combined −log10(p) across all significant
            incident edges per gene.

    Returns
    -------
    pd.Series
        Index = gene_names, values = scores (descending = more significant).

    Raises
    ------
    ValueError
        If ``method`` is not recognised.
    """
    if method not in _VALID_RANK_METHODS:
        raise ValueError(
            f"method must be one of {_VALID_RANK_METHODS}; got '{method}'."
        )

    p = len(gene_names)
    scores = np.zeros(p, dtype=float)

    if method == "degree":
        scores = adj.sum(axis=1).astype(float)

    elif method == "min_pvalue":
        for i in range(p):
            sig_mask = adj[i, :].astype(bool)
            if sig_mask.any():
                min_p = p_values[i, sig_mask].min()
                min_p = max(min_p, 1e-300)  # guard against log(0)
                scores[i] = -np.log10(min_p)

    elif method == "fisher_combined":
        # Fisher's combined test: -2 * sum(log(p_i))
        for i in range(p):
            sig_mask = adj[i, :].astype(bool)
            if sig_mask.any():
                pv = np.clip(p_values[i, sig_mask], 1e-300, 1.0)
                scores[i] = float(-2.0 * np.sum(np.log(pv)))

    return pd.Series(scores, index=gene_names)
