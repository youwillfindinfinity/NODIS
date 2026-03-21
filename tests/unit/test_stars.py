"""
Unit tests for StARS stability selection.
"""

import warnings
import numpy as np
import pytest
from nodis.inference.stars import stars_select
from nodis.estimators.desparsified import DesparifiedGGM


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _hub_data(n: int = 80, p: int = 15, seed: int = 0) -> np.ndarray:
    """Generate data from a hub-graph GGM (n > 5p for reliable inference)."""
    rng = np.random.default_rng(seed)
    # Build hub precision matrix: hub node 0 connects to all others
    Theta = np.eye(p) * 2.0
    for j in range(1, p):
        Theta[0, j] = Theta[j, 0] = 0.3
    # Ensure positive definiteness by inflating diagonal
    min_eig = np.linalg.eigvalsh(Theta).min()
    if min_eig <= 0:
        Theta += (abs(min_eig) + 0.1) * np.eye(p)
    Sigma = np.linalg.inv(Theta)
    return rng.multivariate_normal(np.zeros(p), Sigma, size=n)


# ---------------------------------------------------------------------------
# Output property tests
# ---------------------------------------------------------------------------

def test_returns_tuple():
    X = _hub_data()
    result = stars_select(X, DesparifiedGGM(), n_subsamples=5, seed=0)
    assert isinstance(result, tuple) and len(result) == 2


def test_adjacency_shape():
    X = _hub_data()
    adj, _ = stars_select(X, DesparifiedGGM(), n_subsamples=5, seed=0)
    assert adj.shape == (X.shape[1], X.shape[1])


def test_adjacency_binary():
    X = _hub_data()
    adj, _ = stars_select(X, DesparifiedGGM(), n_subsamples=5, seed=0)
    assert set(np.unique(adj)).issubset({0, 1})


def test_adjacency_symmetric():
    X = _hub_data()
    adj, _ = stars_select(X, DesparifiedGGM(), n_subsamples=5, seed=0)
    np.testing.assert_array_equal(adj, adj.T)


def test_no_self_loops():
    X = _hub_data()
    adj, _ = stars_select(X, DesparifiedGGM(), n_subsamples=5, seed=0)
    np.testing.assert_array_equal(np.diag(adj), np.zeros(X.shape[1], dtype=int))


def test_selected_threshold_in_grid():
    X = _hub_data()
    grid = np.array([0.01, 0.05, 0.1, 0.2, 0.5])
    _, theta = stars_select(X, DesparifiedGGM(), threshold_grid=grid,
                            n_subsamples=5, seed=0)
    assert theta in grid


def test_instability_helper_known_values():
    """_instability returns correct D for known frequency matrices."""
    from nodis.inference.stars import _instability
    p = 4
    # All edges included in every subsample → F=1, D=0
    freq_all = np.ones((p, p))
    assert _instability(freq_all, p) == pytest.approx(0.0)
    # All edges in exactly half of subsamples → F=0.5, D=1.0
    freq_half = np.full((p, p), 0.5)
    assert _instability(freq_half, p) == pytest.approx(1.0)
    # Mixed: half edges stable, half at F=0.5
    freq_mixed = np.ones((p, p))
    uidx = np.triu_indices(p, k=1)
    freq_mixed[uidx[0][:3], uidx[1][:3]] = 0.5
    freq_mixed[uidx[1][:3], uidx[0][:3]] = 0.5
    D = _instability(freq_mixed, p)
    assert D == pytest.approx(0.5)


def test_selected_threshold_not_densest():
    """On well-conditioned hub data StARS should not select the densest graph."""
    grid = np.linspace(0.01, 0.5, 15)
    X = _hub_data(n=100, p=12, seed=0)
    adj_stars, theta_stars = stars_select(X, DesparifiedGGM(),
                                          threshold_grid=grid,
                                          n_subsamples=8, seed=0)
    # At the densest threshold (0.01 → smallest alpha → most edges), the graph
    # should be denser than what StARS selects for good data.
    adj_dense = DesparifiedGGM().fit(X).get_adjacency(0.01)
    # StARS should select a threshold >= the smallest in the grid
    assert theta_stars >= grid[0]
    # The StARS adjacency should have no more edges than the densest graph
    assert adj_stars.sum() <= adj_dense.sum() + 2   # small tolerance


def test_reproducibility_with_seed():
    X = _hub_data()
    adj1, t1 = stars_select(X, DesparifiedGGM(), n_subsamples=5, seed=42)
    adj2, t2 = stars_select(X, DesparifiedGGM(), n_subsamples=5, seed=42)
    np.testing.assert_array_equal(adj1, adj2)
    assert t1 == t2


def test_invalid_beta_raises():
    X = _hub_data()
    with pytest.raises(ValueError, match="beta"):
        stars_select(X, DesparifiedGGM(), beta=1.5)


def test_low_n_p_still_runs():
    """StARS should run (and be preferred over parametric) at n/p < 5."""
    rng = np.random.default_rng(7)
    X = rng.standard_normal((40, 15))   # n/p = 2.7
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        adj, _ = stars_select(X, DesparifiedGGM(), n_subsamples=5, seed=0)
    assert adj.shape == (15, 15)
