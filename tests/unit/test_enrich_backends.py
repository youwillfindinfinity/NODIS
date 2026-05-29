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

    with patch("nodis.enrich.backends.gprofiler_backend.GProfiler",
               return_value=mock_gp_instance):
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
