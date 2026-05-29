"""
nodis.enrich — topology-aware enrichment analysis from GGM inference results.
"""
from nodis.enrich.result import EnrichmentResult

__all__ = ["EnrichmentResult", "from_result", "from_adjacency"]


def from_result(result, gene_names, **kwargs):
    raise NotImplementedError("from_result not yet implemented")


def from_adjacency(adj, gene_names, **kwargs):
    raise NotImplementedError("from_adjacency not yet implemented")
