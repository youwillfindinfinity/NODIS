"""
nodis.enrich — topology-aware enrichment from GGM inference results.

Biological levels
-----------------
``"rna"``               GO terms, KEGG, Reactome, WikiPathways
``"post_transcriptional"``  miRNA targets, TF binding motifs, RNA-binding proteins
``"protein"``           CORUM complexes, InterPro domains, Human Protein Atlas
``"all"``               All three levels combined (default)

Extraction strategies
---------------------
``"hub"``               Hub genes above centrality quantile
``"prerank"``           All genes ranked by edge-evidence score
``"community"``         One gene set per detected network community

Quick start
-----------
>>> from nodis import DesparifiedGGM
>>> from nodis.enrich import from_result
>>> result = DesparifiedGGM().fit(X, fdr_alpha=0.05)
>>> hits = from_result(result, gene_names=gene_list, level="all")
>>> for h in hits:
...     print(h.gene_set_name, len(h.significant()))
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from nodis.enrich.result import EnrichmentResult
from nodis.enrich.databases import get_databases, VALID_LEVELS
from nodis.enrich.extract import hub_genes, ranked_genes, community_gene_sets

__all__ = ["EnrichmentResult", "from_result", "from_adjacency"]

_VALID_BACKENDS = ("gprofiler", "gseapy")
_VALID_EXTRACTIONS = ("hub", "prerank", "community")


def from_result(
    result,
    gene_names: list[str],
    level: str = "all",
    method: str = "ora",
    backend: str = "gprofiler",
    extraction: str = "hub",
    centrality: str = "degree",
    hub_quantile: float = 0.9,
    rank_method: str = "min_pvalue",
    community_algorithm: str = "greedy_modularity",
    organism: str = "hsapiens",
    background: Optional[list[str]] = None,
) -> list[EnrichmentResult]:
    """Run topology-aware enrichment from a ``GGMInferenceResult``.

    Parameters
    ----------
    result : GGMInferenceResult
        Output of ``DesparifiedGGM.fit()``.  Must have ``adj_fdr`` set
        (call ``fit(fdr_alpha=0.05)`` or assign ``result.adj_fdr`` manually).
    gene_names : list of str
        Gene identifiers corresponding to columns of the expression matrix.
        Length must equal ``result.adj_fdr.shape[0]``.
    level : str, default ``"all"``
        Biological level(s) to query. One of ``"rna"``,
        ``"post_transcriptional"``, ``"protein"``, ``"all"``.
    method : str, default ``"ora"``
        Enrichment method: ``"ora"`` or ``"prerank"``.
    backend : str, default ``"gprofiler"``
        Enrichment backend: ``"gprofiler"`` or ``"gseapy"``.
    extraction : str, default ``"hub"``
        Gene extraction strategy: ``"hub"``, ``"prerank"``, or ``"community"``.
    centrality : str, default ``"degree"``
        Centrality metric for ``extraction="hub"``.
    hub_quantile : float, default 0.9
        Quantile cutoff for ``extraction="hub"``.
    rank_method : str, default ``"min_pvalue"``
        Scoring method for ``extraction="prerank"``.
    community_algorithm : str, default ``"greedy_modularity"``
        Community detection algorithm for ``extraction="community"``.
    organism : str, default ``"hsapiens"``
        Organism code (g:Profiler format; ignored by gseapy).
    background : list of str or None
        Custom ORA background. ``None`` → backend default.

    Returns
    -------
    list of EnrichmentResult
        One ``EnrichmentResult`` per gene set submitted (one for
        ``"hub"`` / ``"prerank"``; one per community for ``"community"``).

    Raises
    ------
    ValueError
        If ``level``, ``backend``, or ``extraction`` is invalid, or if
        ``result.adj_fdr`` is ``None``.
    """
    if backend not in _VALID_BACKENDS:
        raise ValueError(
            f"backend must be one of {_VALID_BACKENDS}; got '{backend}'."
        )
    if extraction not in _VALID_EXTRACTIONS:
        raise ValueError(
            f"extraction must be one of {_VALID_EXTRACTIONS}; got '{extraction}'."
        )
    if result.adj_fdr is None:
        raise ValueError(
            "adj_fdr is None. Run DesparifiedGGM.fit(fdr_alpha=0.05) or set "
            "result.adj_fdr manually before calling from_result()."
        )

    return from_adjacency(
        adj=result.adj_fdr,
        gene_names=gene_names,
        p_values=result.p_values,
        level=level,
        method=method,
        backend=backend,
        extraction=extraction,
        centrality=centrality,
        hub_quantile=hub_quantile,
        rank_method=rank_method,
        community_algorithm=community_algorithm,
        organism=organism,
        background=background,
    )


def from_adjacency(
    adj: np.ndarray,
    gene_names: list[str],
    p_values: Optional[np.ndarray] = None,
    level: str = "all",
    method: str = "ora",
    backend: str = "gprofiler",
    extraction: str = "hub",
    centrality: str = "degree",
    hub_quantile: float = 0.9,
    rank_method: str = "min_pvalue",
    community_algorithm: str = "greedy_modularity",
    organism: str = "hsapiens",
    background: Optional[list[str]] = None,
) -> list[EnrichmentResult]:
    """Run topology-aware enrichment from a raw adjacency matrix.

    Same parameters and behaviour as ``from_result()``, but accepts a
    plain adjacency matrix instead of a ``GGMInferenceResult``.

    Parameters
    ----------
    adj : (p, p) ndarray
        Binary symmetric adjacency matrix.
    gene_names : list of str
        Gene identifiers, length p.
    p_values : (p, p) ndarray or None
        Edge p-value matrix. Required when ``extraction="prerank"`` with
        ``rank_method="min_pvalue"`` or ``"fisher_combined"``. Defaults to
        ones matrix (uniform scores) if ``None`` and needed.

    (All other parameters same as ``from_result()``.)

    Returns
    -------
    list of EnrichmentResult
    """
    if backend not in _VALID_BACKENDS:
        raise ValueError(
            f"backend must be one of {_VALID_BACKENDS}; got '{backend}'."
        )
    if extraction not in _VALID_EXTRACTIONS:
        raise ValueError(
            f"extraction must be one of {_VALID_EXTRACTIONS}; got '{extraction}'."
        )

    p = len(gene_names)

    # Fallback p-value matrix for prerank when none provided
    if p_values is None:
        _pv: np.ndarray = np.ones((p, p), dtype=float)
    else:
        _pv = p_values

    # ── Build gene sets to enrich ──────────────────────────────────────────
    gene_sets_map: dict[str, list[str]] = {}

    if extraction == "hub":
        hubs = hub_genes(adj, gene_names, centrality=centrality,
                         quantile=hub_quantile)
        gene_sets_map["hub_genes"] = hubs

    elif extraction == "prerank":
        # prerank is handled differently — entire ranked series is submitted
        gene_sets_map["_prerank"] = gene_names  # sentinel: use ranked_genes()

    elif extraction == "community":
        gene_sets_map = community_gene_sets(adj, gene_names,
                                            algorithm=community_algorithm)

    # ── Run enrichment ─────────────────────────────────────────────────────
    databases = get_databases(level=level, backend=backend)
    results: list[EnrichmentResult] = []

    if backend == "gprofiler":
        from nodis.enrich.backends.gprofiler_backend import run_ora as gp_ora

        for set_name, genes in gene_sets_map.items():
            if set_name == "_prerank":
                # gprofiler doesn't support pre-ranked mode; fall back to ORA
                # using hub genes ranked by degree
                genes = hub_genes(adj, gene_names, centrality="degree",
                                  quantile=0.5)
                set_name = "hub_genes_fallback"

            df = gp_ora(
                gene_list=genes,
                sources=databases,
                organism=organism,
                background=background,
            )
            results.append(EnrichmentResult(
                gene_set_name=set_name,
                level=level,
                backend="gprofiler",
                method="ora",
                results=df,
                gene_list=genes,
                background=background,
                metadata={
                    "centrality": centrality,
                    "hub_quantile": hub_quantile,
                    "extraction": extraction,
                    "community_algorithm": community_algorithm,
                },
            ))

    elif backend == "gseapy":
        from nodis.enrich.backends.gseapy_backend import (
            run_prerank as gs_prerank,
            run_ora as gs_ora,
        )

        if extraction == "prerank":
            rnk = ranked_genes(adj, gene_names, _pv, method=rank_method)
            df = gs_prerank(ranked_series=rnk, gene_sets=databases)
            results.append(EnrichmentResult(
                gene_set_name="ranked_genes",
                level=level,
                backend="gseapy",
                method="prerank",
                results=df,
                gene_list=gene_names,
                background=None,
                metadata={"rank_method": rank_method, "extraction": "prerank"},
            ))
        else:
            for set_name, genes in gene_sets_map.items():
                df = gs_ora(
                    gene_list=genes,
                    gene_sets=databases,
                    background=background,
                )
                results.append(EnrichmentResult(
                    gene_set_name=set_name,
                    level=level,
                    backend="gseapy",
                    method="ora",
                    results=df,
                    gene_list=genes,
                    background=background,
                    metadata={
                        "centrality": centrality,
                        "hub_quantile": hub_quantile,
                        "extraction": extraction,
                        "community_algorithm": community_algorithm,
                    },
                ))

    return results
