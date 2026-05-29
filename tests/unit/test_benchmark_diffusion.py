"""
Unit tests for nodis.benchmark.diffusion_eval.

Covers:
  erdos_renyi_matching   — shape, edge count, symmetry, reproducibility
  make_laplacian         — row-sum-zero, PSD, symmetry, zero input, normalised, isolated node
  make_delta             — random/hub/fiedler modes, disconnected fallback, invalid mode
  diffuse                — output shape, t=0 identity, energy non-increase
  evaluate_diffusion     — perfect-graph spearman, return keys, n_components for empty graph
  evaluate_knockouts     — perfect-graph spearman, return keys, k_eff clipping
"""

import warnings

import numpy as np
import pytest

from nodis.benchmark.diffusion_eval import (
    diffuse,
    erdos_renyi_matching,
    evaluate_diffusion,
    evaluate_knockouts,
    make_delta,
    make_laplacian,
)

# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------

T_GRID = np.linspace(0.01, 1.0, 5)


def _path_graph(p: int) -> np.ndarray:
    """Connected path graph on p nodes (1-2-3-...-p)."""
    A = np.zeros((p, p), dtype=int)
    for i in range(p - 1):
        A[i, i + 1] = 1
        A[i + 1, i] = 1
    return A


def _complete_graph(p: int) -> np.ndarray:
    """Complete graph on p nodes."""
    A = np.ones((p, p), dtype=int) - np.eye(p, dtype=int)
    return A


def _block_disconnected(p1: int, p2: int) -> np.ndarray:
    """Two disconnected complete cliques of sizes p1 and p2."""
    p = p1 + p2
    A = np.zeros((p, p), dtype=int)
    A[:p1, :p1] = _complete_graph(p1)
    A[p1:, p1:] = _complete_graph(p2)
    return A


# ---------------------------------------------------------------------------
# erdos_renyi_matching
# ---------------------------------------------------------------------------

class TestErdosRenyiMatching:

    def test_output_shape_matches_input(self):
        p = 10
        adj = _path_graph(p)
        out = erdos_renyi_matching(adj, seed=0)
        assert out.shape == (p, p)

    def test_edge_count_matches(self):
        p = 12
        adj = _path_graph(p)           # p-1 = 11 edges
        expected_edges = int(adj.sum()) // 2
        out = erdos_renyi_matching(adj, seed=42)
        actual_edges = int(out.sum()) // 2
        assert actual_edges == expected_edges

    def test_symmetric_zero_diagonal(self):
        p = 10
        adj = _complete_graph(p)
        out = erdos_renyi_matching(adj, seed=7)
        np.testing.assert_array_equal(out, out.T)
        assert np.trace(out) == 0

    def test_reproducible_same_seed(self):
        adj = _path_graph(15)
        out1 = erdos_renyi_matching(adj, seed=99)
        out2 = erdos_renyi_matching(adj, seed=99)
        np.testing.assert_array_equal(out1, out2)

    def test_different_seed_gives_different_result(self):
        # Use a sparse graph so many upper-triangle positions exist but only few
        # are chosen — different seeds will pick different subsets.
        p = 20
        adj = _path_graph(p)   # 19 edges out of C(20,2)=190 possible
        out1 = erdos_renyi_matching(adj, seed=1)
        out2 = erdos_renyi_matching(adj, seed=2)
        assert not np.array_equal(out1, out2)


# ---------------------------------------------------------------------------
# make_laplacian (unnormalised)
# ---------------------------------------------------------------------------

class TestMakeLaplacianUnnormalised:

    def test_row_sums_zero(self):
        """L·1 = 0 for any adjacency matrix."""
        adj = _path_graph(10)
        L = make_laplacian(adj, normalised=False)
        row_sums = L.sum(axis=1)
        np.testing.assert_allclose(row_sums, 0.0, atol=1e-12)

    def test_psd(self):
        adj = _complete_graph(8)
        L = make_laplacian(adj, normalised=False)
        eigenvalues = np.linalg.eigvalsh(L)
        assert np.all(eigenvalues >= -1e-10)

    def test_symmetric(self):
        adj = _path_graph(10)
        L = make_laplacian(adj, normalised=False)
        np.testing.assert_allclose(L, L.T, atol=1e-14)

    def test_all_zeros_adj_gives_zero_laplacian(self):
        p = 6
        adj = np.zeros((p, p), dtype=int)
        L = make_laplacian(adj, normalised=False)
        np.testing.assert_array_equal(L, np.zeros((p, p)))


# ---------------------------------------------------------------------------
# make_laplacian (normalised)
# ---------------------------------------------------------------------------

class TestMakeLaplacianNormalised:

    def test_symmetric_normalised(self):
        adj = _path_graph(10)
        L = make_laplacian(adj, normalised=True)
        np.testing.assert_allclose(L, L.T, atol=1e-14)

    def test_psd_normalised(self):
        adj = _complete_graph(8)
        L = make_laplacian(adj, normalised=True)
        eigenvalues = np.linalg.eigvalsh(L)
        assert np.all(eigenvalues >= -1e-10)

    def test_isolated_node_no_nan_inf(self):
        """Node 0 is isolated (degree 0); d_inv_sqrt should be 0, not NaN."""
        p = 6
        # Build a path graph but leave node 0 isolated
        adj = np.zeros((p, p), dtype=int)
        for i in range(1, p - 1):
            adj[i, i + 1] = 1
            adj[i + 1, i] = 1
        L = make_laplacian(adj, normalised=True)
        assert not np.any(np.isnan(L))
        assert not np.any(np.isinf(L))


# ---------------------------------------------------------------------------
# make_delta
# ---------------------------------------------------------------------------

class TestMakeDelta:

    def _laplacian_for(self, adj):
        return make_laplacian(adj, normalised=False)

    def test_random_mode_unit_norm_and_shape(self):
        p = 10
        adj = _path_graph(p)
        L = self._laplacian_for(adj)
        v = make_delta(adj, L, mode='random', seed=0)
        assert v.shape == (p,)
        assert pytest.approx(np.linalg.norm(v), abs=1e-12) == 1.0

    def test_hub_mode_one_hot_max_degree(self):
        p = 10
        adj = _path_graph(p)
        L = self._laplacian_for(adj)
        v = make_delta(adj, L, mode='hub', seed=0)
        assert v.shape == (p,)
        # Must be one-hot
        assert int(v.sum()) == 1
        assert np.count_nonzero(v) == 1
        # Must point to max-degree node
        hub_idx = int(np.argmax(adj.sum(axis=1)))
        assert v[hub_idx] == 1.0

    def test_fiedler_mode_unit_norm_connected_graph(self):
        p = 10
        adj = _path_graph(p)
        L = self._laplacian_for(adj)
        v = make_delta(adj, L, mode='fiedler', seed=0)
        assert v.shape == (p,)
        assert pytest.approx(np.linalg.norm(v), abs=1e-10) == 1.0

    def test_fiedler_disconnected_warns_and_returns_unit_norm(self):
        p = 10
        adj = _block_disconnected(5, 5)
        L = self._laplacian_for(adj)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            v = make_delta(adj, L, mode='fiedler', seed=42)
        # At least one UserWarning about disconnected graph
        assert any(issubclass(warning.category, UserWarning) for warning in w)
        assert v.shape == (p,)
        assert pytest.approx(np.linalg.norm(v), abs=1e-10) == 1.0

    def test_invalid_mode_raises_value_error(self):
        p = 10
        adj = _path_graph(p)
        L = self._laplacian_for(adj)
        with pytest.raises(ValueError, match="Unknown delta mode"):
            make_delta(adj, L, mode='nonsense', seed=0)


# ---------------------------------------------------------------------------
# diffuse
# ---------------------------------------------------------------------------

class TestDiffuse:

    def test_output_shape(self):
        p, T = 10, 5
        adj = _path_graph(p)
        L = make_laplacian(adj)
        delta = make_delta(adj, L, mode='random', seed=0)
        S = diffuse(L, delta, T_GRID)
        assert S.shape == (p, T)

    def test_t0_gives_delta(self):
        """exp(-0·L)·delta = I·delta = delta."""
        p = 10
        adj = _path_graph(p)
        L = make_laplacian(adj)
        delta = make_delta(adj, L, mode='random', seed=0)
        t_with_zero = np.array([0.0, 0.5, 1.0])
        S = diffuse(L, delta, t_with_zero)
        np.testing.assert_allclose(S[:, 0], delta, atol=1e-10)

    def test_energy_non_increasing(self):
        """||S(t)||_2 is non-increasing in t for t > 0."""
        p = 10
        adj = _complete_graph(p)
        L = make_laplacian(adj)
        delta = make_delta(adj, L, mode='random', seed=0)
        t_grid = np.linspace(0.01, 2.0, 10)
        S = diffuse(L, delta, t_grid)
        norms = np.linalg.norm(S, axis=0)  # (T,)
        diffs = np.diff(norms)
        assert np.all(diffs <= 1e-12), f"Energy increased: {diffs}"


# ---------------------------------------------------------------------------
# evaluate_diffusion
# ---------------------------------------------------------------------------

class TestEvaluateDiffusion:

    def test_perfect_graph_spearman_near_one(self):
        p = 12
        adj = _path_graph(p)
        L = make_laplacian(adj)
        delta = make_delta(adj, L, mode='random', seed=0)
        result = evaluate_diffusion(adj, adj, delta, T_GRID)
        assert result["diffusion_spearman_mean"] >= 0.99

    def test_returns_all_expected_keys(self):
        p = 10
        adj = _path_graph(p)
        L = make_laplacian(adj)
        delta = make_delta(adj, L, mode='random', seed=0)
        result = evaluate_diffusion(adj, adj, delta, T_GRID)
        expected_keys = {
            "diffusion_spearman_mean",
            "diffusion_spearman_min",
            "diffusion_mae_mean",
            "n_components_pred",
        }
        assert set(result.keys()) == expected_keys

    def test_n_components_all_zeros_adj(self):
        """All-zero adj_pred → every node is its own component → n_components = p."""
        p = 10
        adj_true = _path_graph(p)
        adj_pred = np.zeros((p, p), dtype=int)
        L = make_laplacian(adj_true)
        delta = make_delta(adj_true, L, mode='random', seed=0)
        result = evaluate_diffusion(adj_pred, adj_true, delta, T_GRID)
        assert result["n_components_pred"] == p


# ---------------------------------------------------------------------------
# evaluate_knockouts
# ---------------------------------------------------------------------------

class TestEvaluateKnockouts:

    def test_perfect_graph_spearman_near_one(self):
        p = 12
        adj = _path_graph(p)
        L = make_laplacian(adj)
        delta = make_delta(adj, L, mode='random', seed=0)
        result = evaluate_knockouts(adj, adj, delta, T_GRID)
        assert result["knockout_spearman"] >= 0.99

    def test_returns_all_expected_keys(self):
        p = 10
        adj = _path_graph(p)
        L = make_laplacian(adj)
        delta = make_delta(adj, L, mode='random', seed=0)
        result = evaluate_knockouts(adj, adj, delta, T_GRID)
        expected_keys = {
            "knockout_spearman",
            "knockout_top10_recall",
            "knockout_topk_recall",
        }
        assert set(result.keys()) == expected_keys

    def test_k_eff_clipping_p50(self):
        """p=50 → k_eff = min(10, 50//5) = 10."""
        p = 50
        adj = _path_graph(p)
        L = make_laplacian(adj)
        delta = make_delta(adj, L, mode='random', seed=0)
        # topk=10, p//5=10 → k_eff=10; knockout_topk_recall denominator is 10
        result = evaluate_knockouts(adj, adj, delta, T_GRID, topk=10)
        # k_eff=10; recall numerator ≤ k_eff so value in [0,1]
        assert 0.0 <= result["knockout_topk_recall"] <= 1.0

    def test_k_eff_clipping_p15(self):
        """p=15 → k_eff = min(10, 15//5) = 3."""
        p = 15
        adj = _path_graph(p)
        L = make_laplacian(adj)
        delta = make_delta(adj, L, mode='random', seed=0)
        # topk=10, p//5=3 → k_eff=3
        result = evaluate_knockouts(adj, adj, delta, T_GRID, topk=10)
        # With perfect graph topk_recall should equal 1.0
        assert result["knockout_topk_recall"] == pytest.approx(1.0, abs=1e-12)
