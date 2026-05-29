"""
EnrichmentResult — container for a single enrichment analysis output.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd


@dataclass
class EnrichmentResult:
    """Container for one enrichment analysis run.

    Attributes
    ----------
    gene_set_name : str
        Label for the input gene set (e.g. ``"hub_genes"``, ``"community_0"``).
    level : str
        Biological level queried: ``"rna"``, ``"post_transcriptional"``,
        ``"protein"``, or ``"all"``.
    backend : str
        Enrichment backend used: ``"gprofiler"`` or ``"gseapy"``.
    method : str
        Analysis method: ``"ora"`` (over-representation) or ``"prerank"``
        (pre-ranked GSEA).
    results : pd.DataFrame
        Enrichment table. Columns depend on backend but always include
        ``term_id``, ``term_name``, ``p_value``, ``adjusted_p_value``.
    gene_list : list of str
        Input gene identifiers submitted for enrichment.
    background : list of str or None
        Background gene set used for ORA. ``None`` → backend default.
    metadata : dict
        Extra context (e.g. centrality scores, community algorithm, hub
        quantile threshold).
    """

    gene_set_name: str
    level: str
    backend: str
    method: str
    results: pd.DataFrame
    gene_list: list[str]
    background: Optional[list[str]] = None
    metadata: dict = field(default_factory=dict)

    def is_empty(self) -> bool:
        """Return True if the results table has no rows."""
        return len(self.results) == 0

    def significant(self, alpha: float = 0.05) -> pd.DataFrame:
        """Return rows where ``adjusted_p_value <= alpha``.

        If ``adjusted_p_value`` is not a column in ``self.results``, returns
        the full table and emits a ``UserWarning``.
        """
        if "adjusted_p_value" not in self.results.columns:
            warnings.warn(
                "EnrichmentResult.significant(): 'adjusted_p_value' column not found; "
                "returning all rows. Ensure the backend normalises column names.",
                UserWarning,
                stacklevel=2,
            )
            return self.results
        return self.results[self.results["adjusted_p_value"] <= alpha]
