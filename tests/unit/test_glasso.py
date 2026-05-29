"""
Unit tests for nodis/estimators/glasso.py.
"""
import importlib.util

import numpy as np
import pytest

from nodis.estimators.glasso import SklearnGLasso

GGLASSO_AVAILABLE = importlib.util.find_spec("gglasso") is not None

RNG = np.random.default_rng(0)
X = RNG.standard_normal((50, 8))
N, P = X.shape


# ---------------------------------------------------------------------------
# SklearnGLasso
# ---------------------------------------------------------------------------

def test_sklearn_fit_returns_self():
    est = SklearnGLasso(cv=3, n_jobs=1)
    result = est.fit(X)
    assert result is est


def test_sklearn_adjacency_shape():
    est = SklearnGLasso(cv=3, n_jobs=1).fit(X)
    adj = est.get_adjacency()
    assert adj.shape == (P, P)


def test_sklearn_adjacency_binary():
    est = SklearnGLasso(cv=3, n_jobs=1).fit(X)
    adj = est.get_adjacency()
    assert set(np.unique(adj)).issubset({0, 1})


def test_sklearn_adjacency_no_self_loops():
    est = SklearnGLasso(cv=3, n_jobs=1).fit(X)
    adj = est.get_adjacency()
    assert np.diag(adj).sum() == 0


def test_sklearn_adjacency_symmetric():
    est = SklearnGLasso(cv=3, n_jobs=1).fit(X)
    adj = est.get_adjacency()
    np.testing.assert_array_equal(adj, adj.T)


def test_sklearn_precision_shape():
    est = SklearnGLasso(cv=3, n_jobs=1).fit(X)
    assert est.precision_.shape == (P, P)


def test_sklearn_get_adjacency_before_fit_raises():
    est = SklearnGLasso()
    with pytest.raises(RuntimeError):
        est.get_adjacency()


def test_sklearn_precision_before_fit_raises():
    est = SklearnGLasso()
    with pytest.raises(RuntimeError):
        _ = est.precision_


def test_sklearn_threshold_effect():
    """threshold=1e6 → all-zero adjacency (no entry can exceed it)."""
    est = SklearnGLasso(cv=3, n_jobs=1).fit(X)
    adj = est.get_adjacency(threshold=1e6)
    assert adj.sum() == 0


# ---------------------------------------------------------------------------
# GGLassoEstimator (skipped when gglasso not installed)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not GGLASSO_AVAILABLE, reason="gglasso not installed")
def test_gglasso_fit_returns_self():
    from nodis.estimators.glasso import GGLassoEstimator
    est = GGLassoEstimator()
    assert est.fit(X) is est


@pytest.mark.skipif(not GGLASSO_AVAILABLE, reason="gglasso not installed")
def test_gglasso_adjacency_shape():
    from nodis.estimators.glasso import GGLassoEstimator
    adj = GGLassoEstimator().fit(X).get_adjacency()
    assert adj.shape == (P, P)


@pytest.mark.skipif(not GGLASSO_AVAILABLE, reason="gglasso not installed")
def test_gglasso_adjacency_no_self_loops():
    from nodis.estimators.glasso import GGLassoEstimator
    adj = GGLassoEstimator().fit(X).get_adjacency()
    assert np.diag(adj).sum() == 0


@pytest.mark.skipif(not GGLASSO_AVAILABLE, reason="gglasso not installed")
def test_gglasso_get_adjacency_before_fit_raises():
    from nodis.estimators.glasso import GGLassoEstimator
    est = GGLassoEstimator()
    with pytest.raises(RuntimeError):
        est.get_adjacency()
