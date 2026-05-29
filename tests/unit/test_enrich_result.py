import pandas as pd
import pytest
from nodis.enrich.result import EnrichmentResult


def test_enrichment_result_fields():
    df = pd.DataFrame({
        "term_id": ["GO:0006915"],
        "term_name": ["apoptotic process"],
        "p_value": [0.001],
        "adjusted_p_value": [0.04],
        "intersection_size": [5],
    })
    er = EnrichmentResult(
        gene_set_name="hub_genes",
        level="rna",
        backend="gprofiler",
        method="ora",
        results=df,
        gene_list=["TP53", "BCL2", "CASP3"],
        background=None,
        metadata={},
    )
    assert er.gene_set_name == "hub_genes"
    assert er.level == "rna"
    assert isinstance(er.results, pd.DataFrame)
    assert len(er.results) == 1


def test_enrichment_result_is_empty():
    er = EnrichmentResult(
        gene_set_name="community_0",
        level="protein",
        backend="gseapy",
        method="prerank",
        results=pd.DataFrame(),
        gene_list=[],
        background=None,
        metadata={},
    )
    assert er.is_empty()


def test_enrichment_result_not_empty():
    df = pd.DataFrame({"term_id": ["GO:0001"]})
    er = EnrichmentResult(
        gene_set_name="hub_genes",
        level="all",
        backend="gprofiler",
        method="ora",
        results=df,
        gene_list=["A"],
        background=None,
        metadata={},
    )
    assert not er.is_empty()


def test_significant_filters_by_alpha():
    df = pd.DataFrame({
        "term_id": ["GO:0001", "GO:0002", "GO:0003"],
        "term_name": ["term A", "term B", "term C"],
        "adjusted_p_value": [0.01, 0.06, 0.04],
    })
    er = EnrichmentResult(
        gene_set_name="hub_genes",
        level="rna",
        backend="gprofiler",
        method="ora",
        results=df,
        gene_list=["A"],
    )
    sig = er.significant(alpha=0.05)
    assert len(sig) == 2
    assert set(sig["term_id"]) == {"GO:0001", "GO:0003"}


def test_significant_custom_alpha():
    df = pd.DataFrame({
        "term_id": ["GO:0001", "GO:0002"],
        "adjusted_p_value": [0.01, 0.04],
    })
    er = EnrichmentResult(
        gene_set_name="hub_genes", level="rna", backend="gprofiler",
        method="ora", results=df, gene_list=["A"],
    )
    assert len(er.significant(alpha=0.02)) == 1
    assert len(er.significant(alpha=0.05)) == 2


def test_significant_missing_column_warns():
    df = pd.DataFrame({"term_id": ["GO:0001"], "p_value": [0.001]})
    er = EnrichmentResult(
        gene_set_name="hub_genes", level="rna", backend="gprofiler",
        method="ora", results=df, gene_list=["A"],
    )
    with pytest.warns(UserWarning, match="adjusted_p_value"):
        result = er.significant(alpha=0.05)
    assert len(result) == 1  # full df returned
