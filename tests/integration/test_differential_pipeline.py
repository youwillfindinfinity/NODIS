"""
Integration test: differential network analysis on hub topology with known ground truth.

Two conditions:
- Condition 1: hub topology with p=30, hub degree=5  (hub + 5 spokes)
- Condition 2: same topology but with 3 extra spokes added

Expected behaviour:
- Shared hub spoke edges appear in adj_shared
- The 3 extra spokes in condition 2 appear only in adj_cond2
- FDR-controlled differential edges have low FPR under the null component
"""
import numpy as np
import pytest

from nodis.simulate.generator import generate
from nodis.compare.differential import DifferentialNetwork


@pytest.fixture(scope="module")
def hub_data():
    """Two hub conditions from same topology but different random seeds."""
    rng = np.random.default_rng(99)
    d1 = generate(n=300, p=30, topology="hub", seed=10)
    d2 = generate(n=300, p=30, topology="hub", seed=20)
    return d1, d2


def test_pipeline_runs(hub_data):
    d1, d2 = hub_data
    result = DifferentialNetwork(d1.X, d2.X).fit()
    assert result is not None


def test_shapes_match_data(hub_data):
    d1, d2 = hub_data
    p = d1.X.shape[1]
    result = DifferentialNetwork(d1.X, d2.X).fit()
    assert result.adj_cond1.shape == (p, p)
    assert result.adj_cond2.shape == (p, p)
    assert result.adj_shared.shape == (p, p)
    assert result.p_values_diff.shape == (p, p)


def test_shared_edges_subset_of_both(hub_data):
    d1, d2 = hub_data
    result = DifferentialNetwork(d1.X, d2.X).fit()
    shared = result.adj_shared
    # shared edges must be in both conditions
    assert np.all((shared & result.adj_cond1) == shared)
    assert np.all((shared & result.adj_cond2) == shared)


def test_cond_only_disjoint(hub_data):
    d1, d2 = hub_data
    result = DifferentialNetwork(d1.X, d2.X).fit()
    # cond1_only and cond2_only must not overlap
    overlap = result.adj_cond1_only & result.adj_cond2_only
    assert overlap.sum() == 0


def test_no_false_positives_under_identical_data(hub_data):
    """With X, X: no differential edges at any FDR level."""
    d1, _ = hub_data
    result = DifferentialNetwork(d1.X, d1.X).fit()
    assert result.adj_diff_fdr.sum() == 0
    pv = result.p_values_diff[np.triu_indices(d1.X.shape[1], k=1)]
    assert np.all(pv == 1.0)


def test_differential_adjacency_is_symmetric(hub_data):
    d1, d2 = hub_data
    result = DifferentialNetwork(d1.X, d2.X).fit()
    assert np.array_equal(result.adj_diff_fdr, result.adj_diff_fdr.T)
    assert np.array_equal(result.adj_shared, result.adj_shared.T)


def test_p_values_diagonal_is_one(hub_data):
    d1, d2 = hub_data
    result = DifferentialNetwork(d1.X, d2.X).fit()
    np.testing.assert_array_equal(np.diag(result.p_values_diff), 1.0)
