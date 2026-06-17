"""
Integration test: AnnData roundtrip via from_anndata → fit → to_anndata.

Verifies that NODIS results are correctly written back into the AnnData
object and that the connectivities matrix is populated in the format
expected by scanpy/squidpy.

Uses a minimal duck-typed AnnData stub — no anndata package required.
"""

import numpy as np
import pytest
from scipy.sparse import issparse

from nodis.estimators.desparsified import DesparifiedGGM
from nodis.preprocess.anndata_compat import from_anndata, to_anndata


class _AnnDataStub:
    """Minimal duck-typed AnnData-like object for testing."""

    def __init__(self, X, var_names):
        self.X = X
        self.obsp = {}
        self.uns = {}

        class _Var:
            pass
        self.var = _Var()
        self.var_names = list(var_names)
        self.var.highly_variable = None


def _make_stub(n=60, p=20, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p)).astype(np.float32)
    gene_names = [f"gene_{i:03d}" for i in range(p)]
    return _AnnDataStub(X, gene_names)


def test_from_anndata_returns_ndarray():
    stub = _make_stub()
    X = from_anndata(stub)
    assert isinstance(X, np.ndarray)
    assert X.dtype == np.float64
    assert X.shape == (60, 20)


def test_to_anndata_requires_adj_fdr():
    stub = _make_stub()
    X = from_anndata(stub)
    est = DesparifiedGGM()
    est.fit(X)
    # adj_fdr is None before get_adjacency() — must raise
    with pytest.raises(ValueError, match="adj_fdr is None"):
        to_anndata(stub, est.result_)


def test_to_anndata_populates_obsp_and_uns():
    """Full roundtrip: AnnData → fit → get_adjacency → to_anndata."""
    import warnings
    stub = _make_stub(n=80, p=20, seed=7)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        X = from_anndata(stub)
        est = DesparifiedGGM()
        est.fit(X)
        est.get_adjacency(alpha=0.05)
        to_anndata(stub, est.result_, key="nodis")

    assert "nodis_connectivities" in stub.obsp
    conn = stub.obsp["nodis_connectivities"]
    assert issparse(conn), "connectivities must be a scipy sparse matrix"
    assert conn.shape == (20, 20), "shape must be (p, p)"

    assert "nodis" in stub.uns
    meta = stub.uns["nodis"]
    assert meta["method"] == "B_NW_SL"
    assert meta["fdr_alpha"] == 0.05
    assert isinstance(meta["n_edges"], int)
    assert meta["doi"] == "10.5281/zenodo.20452188"


def test_to_anndata_connectivity_matches_adj_fdr():
    """Connectivities sparse matrix encodes exactly the FDR adjacency."""
    import warnings
    stub = _make_stub(n=100, p=20, seed=3)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        X = from_anndata(stub)
        est = DesparifiedGGM()
        est.fit(X)
        adj = est.get_adjacency(alpha=0.05)
        to_anndata(stub, est.result_)

    conn_dense = stub.obsp["nodis_connectivities"].toarray()
    expected = adj.astype(np.float32)
    np.testing.assert_array_equal(conn_dense, expected)


def test_to_anndata_custom_key():
    """Custom key prefix is written correctly."""
    import warnings
    stub = _make_stub(n=80, p=15, seed=11)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        X = from_anndata(stub)
        est = DesparifiedGGM()
        est.fit(X)
        est.get_adjacency(alpha=0.10)
        to_anndata(stub, est.result_, key="coexp")

    assert "coexp_connectivities" in stub.obsp
    assert "coexp" in stub.uns
    assert stub.uns["coexp"]["fdr_alpha"] == 0.10
