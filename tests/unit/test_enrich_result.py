import pandas as pd
import pytest
from nodis.enrich.databases import get_databases, VALID_LEVELS
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


def test_valid_levels():
    assert set(VALID_LEVELS) == {"rna", "post_transcriptional", "protein", "all"}


def test_get_databases_rna_gprofiler():
    dbs = get_databases(level="rna", backend="gprofiler")
    assert "GO:BP" in dbs
    assert "KEGG" in dbs
    assert "REAC" in dbs


def test_get_databases_protein_gprofiler():
    dbs = get_databases(level="protein", backend="gprofiler")
    assert "CORUM" in dbs
    assert "HPA" in dbs


def test_get_databases_post_transcriptional_gprofiler():
    dbs = get_databases(level="post_transcriptional", backend="gprofiler")
    assert "MIRNA" in dbs
    assert "TF" in dbs


def test_get_databases_all_gprofiler():
    dbs = get_databases(level="all", backend="gprofiler")
    assert "GO:BP" in dbs
    assert "CORUM" in dbs
    assert "MIRNA" in dbs


def test_get_databases_rna_gseapy():
    dbs = get_databases(level="rna", backend="gseapy")
    assert "GO_Biological_Process_2023" in dbs
    assert "KEGG_2021_Human" in dbs
    assert "Reactome_2022" in dbs


def test_get_databases_invalid_level():
    with pytest.raises(ValueError, match="level must be one of"):
        get_databases(level="lipids", backend="gprofiler")


def test_nodis_enrich_importable():
    """nodis.enrich must be importable from the top-level package."""
    import nodis.enrich
    assert hasattr(nodis.enrich, "from_result")
    assert hasattr(nodis.enrich, "from_adjacency")
    assert hasattr(nodis.enrich, "EnrichmentResult")
