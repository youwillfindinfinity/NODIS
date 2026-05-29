"""
Integration tests for nodis.enrich public API.
Network calls are mocked — no internet required.
"""
import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock

from nodis.estimators.desparsified import GGMInferenceResult
from nodis.enrich import from_result, from_adjacency
from nodis.enrich.result import EnrichmentResult


def _make_inference_result(p: int = 10, seed: int = 0):
    rng = np.random.default_rng(seed)
    A = (rng.random((p, p)) < 0.25).astype(int)
    A = np.triu(A, k=1); A = A + A.T; np.fill_diagonal(A, 0)
    pvals = rng.random((p, p))
    pvals = (pvals + pvals.T) / 2; np.fill_diagonal(pvals, 1.0)
    result = GGMInferenceResult(
        z_scores=np.zeros((p, p)), p_values=pvals,
        precision=np.eye(p), variance=np.ones((p, p)),
        adj_fdr=A, fdr_alpha=0.05,
    )
    return result, [f"GENE{i}" for i in range(p)]


def _mock_gprofiler_df():
    return pd.DataFrame({
        "term_id": ["GO:0006915"], "term_name": ["apoptotic process"],
        "p_value": [0.001], "adjusted_p_value": [0.02],
        "source": ["GO:BP"], "intersection_size": [2],
        "query_size": [5], "term_size": [200], "intersections": [["GENE0"]],
    })


def _mock_gp_class(df):
    mock_gp_instance = MagicMock()
    mock_gp_instance.profile.return_value = df
    mock_class = MagicMock(return_value=mock_gp_instance)
    return mock_class


def test_from_result_returns_list_of_enrichment_results():
    result, genes = _make_inference_result()
    with patch("nodis.enrich.backends.gprofiler_backend._get_gprofiler",
               return_value=_mock_gp_class(_mock_gprofiler_df())):
        hits = from_result(result, gene_names=genes, level="rna",
                           method="ora", backend="gprofiler", extraction="hub")
    assert isinstance(hits, list)
    assert len(hits) >= 1
    assert all(isinstance(h, EnrichmentResult) for h in hits)


def test_from_result_community_extraction():
    result, genes = _make_inference_result()
    with patch("nodis.enrich.backends.gprofiler_backend._get_gprofiler",
               return_value=_mock_gp_class(_mock_gprofiler_df())):
        hits = from_result(result, gene_names=genes, level="rna",
                           method="ora", backend="gprofiler", extraction="community")
    assert isinstance(hits, list)
    assert len(hits) >= 1
    for h in hits:
        assert h.gene_set_name.startswith("community_")


def test_from_adjacency_accepts_plain_adj():
    rng = np.random.default_rng(42)
    p = 8
    A = (rng.random((p, p)) < 0.3).astype(int)
    A = np.triu(A, k=1); A = A + A.T; np.fill_diagonal(A, 0)
    pvals = rng.random((p, p)); pvals = (pvals + pvals.T)/2; np.fill_diagonal(pvals, 1.0)
    with patch("nodis.enrich.backends.gprofiler_backend._get_gprofiler",
               return_value=_mock_gp_class(_mock_gprofiler_df())):
        hits = from_adjacency(A, [f"G{i}" for i in range(p)],
                              p_values=pvals, level="protein",
                              method="ora", backend="gprofiler")
    assert isinstance(hits, list)


def test_from_result_invalid_backend():
    result, genes = _make_inference_result()
    with pytest.raises(ValueError, match="backend must be"):
        from_result(result, gene_names=genes, backend="r_clusterprofiler")


def test_from_result_invalid_extraction():
    result, genes = _make_inference_result()
    with pytest.raises(ValueError, match="extraction must be"):
        from_result(result, gene_names=genes, extraction="random_walk")


def test_from_result_adj_fdr_none_raises():
    result, genes = _make_inference_result()
    result.adj_fdr = None
    with pytest.raises(ValueError, match="adj_fdr is None"):
        from_result(result, gene_names=genes)
