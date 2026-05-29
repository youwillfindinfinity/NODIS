"""
Backends are tested with mocked API calls — no network required.
"""
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock


def test_gprofiler_run_ora_returns_dataframe():
    from nodis.enrich.backends.gprofiler_backend import run_ora

    mock_gp_instance = MagicMock()
    mock_gp_instance.profile.return_value = {"results": [
        {
            "source": "GO:BP",
            "native": "GO:0006915",
            "name": "apoptotic process",
            "p_value": 0.001,
            "significant": True,
            "intersection_size": 3,
            "query_size": 10,
            "term_size": 200,
            "intersections": ["TP53", "BCL2", "CASP3"],
        }
    ], "meta": {}}

    mock_gprofiler_class = MagicMock(return_value=mock_gp_instance)
    with patch("nodis.enrich.backends.gprofiler_backend._get_gprofiler",
               return_value=mock_gprofiler_class):
        df = run_ora(
            gene_list=["TP53", "BCL2", "CASP3", "EGFR"],
            sources=["GO:BP", "CORUM"],
            organism="hsapiens",
            background=None,
        )

    assert isinstance(df, pd.DataFrame)
    assert "term_id" in df.columns
    assert "term_name" in df.columns
    assert "p_value" in df.columns
    assert "adjusted_p_value" in df.columns


def test_gprofiler_run_ora_empty_gene_list_returns_empty():
    from nodis.enrich.backends.gprofiler_backend import run_ora

    df = run_ora(gene_list=[], sources=["GO:BP"], organism="hsapiens")
    assert isinstance(df, pd.DataFrame)
    assert df.empty


def test_gprofiler_run_ora_dataframe_response():
    """Test the DataFrame response path (primary production path with return_dataframe=True)."""
    from nodis.enrich.backends.gprofiler_backend import run_ora

    mock_df = pd.DataFrame({
        "native": ["GO:0006915"],
        "name": ["apoptotic process"],
        "p_value": [0.001],
        "source": ["GO:BP"],
        "intersection_size": [3],
        "query_size": [10],
        "term_size": [200],
        "intersections": [["TP53", "BCL2"]],
    })
    mock_gp_instance = MagicMock()
    mock_gp_instance.profile.return_value = mock_df
    mock_gprofiler_class = MagicMock(return_value=mock_gp_instance)

    with patch("nodis.enrich.backends.gprofiler_backend._get_gprofiler",
               return_value=mock_gprofiler_class):
        df = run_ora(gene_list=["TP53", "BCL2"], sources=["GO:BP"])

    assert "term_id" in df.columns
    assert "term_name" in df.columns
    assert df.iloc[0]["term_id"] == "GO:0006915"


def test_gprofiler_run_ora_none_response_returns_empty():
    """Test that None response from profile() returns empty DataFrame."""
    from nodis.enrich.backends.gprofiler_backend import run_ora

    mock_gp_instance = MagicMock()
    mock_gp_instance.profile.return_value = None
    mock_gprofiler_class = MagicMock(return_value=mock_gp_instance)

    with patch("nodis.enrich.backends.gprofiler_backend._get_gprofiler",
               return_value=mock_gprofiler_class):
        df = run_ora(gene_list=["TP53"], sources=["GO:BP"])

    assert isinstance(df, pd.DataFrame)
    assert df.empty


def test_gprofiler_run_ora_empty_results_dict_returns_empty():
    """Test that dict response with empty results list returns empty DataFrame."""
    from nodis.enrich.backends.gprofiler_backend import run_ora

    mock_gp_instance = MagicMock()
    mock_gp_instance.profile.return_value = {"results": [], "meta": {}}
    mock_gprofiler_class = MagicMock(return_value=mock_gp_instance)

    with patch("nodis.enrich.backends.gprofiler_backend._get_gprofiler",
               return_value=mock_gprofiler_class):
        df = run_ora(gene_list=["TP53"], sources=["GO:BP"])

    assert isinstance(df, pd.DataFrame)
    assert df.empty


# ── gseapy backend ────────────────────────────────────────────────────────────

def test_gseapy_prerank_returns_dataframe():
    from nodis.enrich.backends.gseapy_backend import run_prerank
    import pandas as pd

    mock_result = MagicMock()
    mock_result.res2d = pd.DataFrame({
        "Term": ["GO:0006915 apoptotic process"],
        "ES": [0.62],
        "NES": [1.85],
        "NOM p-val": [0.012],
        "FDR q-val": [0.04],
        "FGSEA p-val": [0.012],
        "Tag %": ["30%"],
        "Lead_genes": ["TP53;BCL2"],
    })

    with patch("nodis.enrich.backends.gseapy_backend.gseapy") as mock_gs:
        mock_gs.prerank.return_value = mock_result
        df = run_prerank(
            ranked_series=pd.Series({"TP53": 3.5, "BCL2": 2.1, "EGFR": 0.5}),
            gene_sets=["GO_Biological_Process_2023"],
            threads=1,
        )

    assert isinstance(df, pd.DataFrame)
    assert "term_name" in df.columns
    assert "adjusted_p_value" in df.columns
    assert "NES" in df.columns


def test_gseapy_prerank_empty_series_returns_empty():
    from nodis.enrich.backends.gseapy_backend import run_prerank
    import pandas as pd

    df = run_prerank(
        ranked_series=pd.Series(dtype=float),
        gene_sets=["GO_Biological_Process_2023"],
    )
    assert isinstance(df, pd.DataFrame)
    assert df.empty


def test_gseapy_ora_returns_dataframe():
    from nodis.enrich.backends.gseapy_backend import run_ora

    mock_result = MagicMock()
    mock_result.results = pd.DataFrame({
        "Gene_set": ["GO_Biological_Process_2023"],
        "Term": ["apoptotic process"],
        "Overlap": ["5/200"],
        "P-value": [0.001],
        "Adjusted P-value": [0.04],
        "Old P-value": [0.001],
        "Old Adjusted P-value": [0.04],
        "Odds Ratio": [3.2],
        "Combined Score": [22.1],
        "Genes": ["TP53;BCL2;CASP3"],
    })

    with patch("nodis.enrich.backends.gseapy_backend.gseapy") as mock_gs:
        mock_gs.enrichr.return_value = mock_result
        df = run_ora(
            gene_list=["TP53", "BCL2", "CASP3"],
            gene_sets=["GO_Biological_Process_2023", "KEGG_2021_Human"],
        )

    assert isinstance(df, pd.DataFrame)
    assert "term_name" in df.columns
    assert "p_value" in df.columns
    assert "adjusted_p_value" in df.columns
