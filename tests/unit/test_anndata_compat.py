"""
Unit tests for AnnData input adapter.

AnnData is an optional dependency; tests mock the AnnData object so the
test suite can run without it installed.
"""

import numpy as np
import pytest
import scipy.sparse


class _MockVar:
    """Minimal mock for adata.var."""
    def __init__(self, index, hvg_flags=None):
        self._index = list(index)
        self._data = {}
        if hvg_flags is not None:
            self._data['highly_variable'] = np.array(hvg_flags, dtype=bool)

    def __contains__(self, key):
        return key in self._data

    def __getitem__(self, key):
        return self._data[key]

    @property
    def index(self):
        return self._index


class _MockAnnData:
    """Minimal mock of an AnnData object."""
    def __init__(self, X, layers=None, var_names=None, hvg_flags=None):
        self.X = X
        self.layers = layers or {}
        n_genes = X.shape[1]
        names = var_names or [f"Gene{i}" for i in range(n_genes)]
        self.var = _MockVar(names, hvg_flags)
        self.var_names = names


# ---------------------------------------------------------------------------
# Basic extraction
# ---------------------------------------------------------------------------

def test_dense_array_passthrough():
    from nodis.preprocess.anndata_compat import from_anndata
    X = np.ones((10, 5), dtype=np.float32)
    adata = _MockAnnData(X)
    result = from_anndata(adata)
    assert result.shape == (10, 5)
    assert result.dtype == np.float64


def test_sparse_converted_to_dense():
    from nodis.preprocess.anndata_compat import from_anndata
    X = scipy.sparse.csr_matrix(np.eye(8, 4))
    adata = _MockAnnData(X)
    result = from_anndata(adata)
    assert isinstance(result, np.ndarray)
    assert result.shape == (8, 4)


def test_csc_sparse_converted_to_dense():
    from nodis.preprocess.anndata_compat import from_anndata
    X = scipy.sparse.csc_matrix(np.eye(6, 4))
    adata = _MockAnnData(X)
    result = from_anndata(adata)
    assert isinstance(result, np.ndarray)
    assert result.shape == (6, 4)


def test_layer_selection():
    from nodis.preprocess.anndata_compat import from_anndata
    X = np.zeros((10, 5))
    log_layer = np.ones((10, 5)) * 2.0
    adata = _MockAnnData(X, layers={"log1p_norm": log_layer})
    result = from_anndata(adata, layer="log1p_norm")
    np.testing.assert_array_equal(result, log_layer)


def test_missing_layer_raises():
    from nodis.preprocess.anndata_compat import from_anndata
    X = np.zeros((10, 5))
    adata = _MockAnnData(X)
    with pytest.raises(KeyError):
        from_anndata(adata, layer="nonexistent")


# ---------------------------------------------------------------------------
# Gene subsetting
# ---------------------------------------------------------------------------

def test_hvg_subset():
    from nodis.preprocess.anndata_compat import from_anndata
    X = np.arange(40, dtype=float).reshape(8, 5)
    hvg = [True, False, True, False, True]
    adata = _MockAnnData(X, hvg_flags=hvg)
    result = from_anndata(adata, use_hvg=True)
    assert result.shape == (8, 3)
    np.testing.assert_array_equal(result, X[:, hvg])


def test_hvg_flag_absent_no_subset():
    from nodis.preprocess.anndata_compat import from_anndata
    X = np.ones((6, 4))
    adata = _MockAnnData(X)   # no hvg_flags
    result = from_anndata(adata, use_hvg=True)
    assert result.shape == (6, 4)


def test_genes_by_name():
    from nodis.preprocess.anndata_compat import from_anndata
    X = np.eye(5)
    names = ["A", "B", "C", "D", "E"]
    adata = _MockAnnData(X, var_names=names)
    result = from_anndata(adata, genes=["A", "C", "E"])
    assert result.shape == (5, 3)


def test_genes_unknown_name_raises():
    from nodis.preprocess.anndata_compat import from_anndata
    X = np.eye(4)
    adata = _MockAnnData(X, var_names=["A", "B", "C", "D"])
    with pytest.raises(ValueError, match="not found"):
        from_anndata(adata, genes=["A", "Z"])


# ---------------------------------------------------------------------------
# NPN integration
# ---------------------------------------------------------------------------

def test_npn_flag_transforms():
    from nodis.preprocess.anndata_compat import from_anndata
    rng = np.random.default_rng(0)
    X = rng.lognormal(size=(50, 8))
    adata = _MockAnnData(X)
    result_raw = from_anndata(adata, npn=False)
    result_npn = from_anndata(adata, npn=True)
    assert not np.allclose(result_raw, result_npn)
    # NPN shrinkage maps marginals to approximately N(0,1)
    assert np.allclose(result_npn.mean(axis=0), 0.0, atol=0.3)
    assert np.allclose(result_npn.std(axis=0), 1.0, atol=0.3)


# ---------------------------------------------------------------------------
# Top-level re-export
# ---------------------------------------------------------------------------

def test_top_level_import():
    import nodis
    assert hasattr(nodis, "from_anndata")
