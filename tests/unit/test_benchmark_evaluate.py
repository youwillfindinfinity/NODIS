"""
Unit tests for nodis.benchmark.evaluate.

Covers: perfect prediction, all-wrong prediction, SHD correctness, MCC formula,
no-positive ground truth, all-positive ground truth, scores matrix, return keys,
asymmetric input behaviour, and 2x2 matrix sanity.
"""

import math

import numpy as np
import pytest

from nodis.benchmark.evaluate import evaluate_predictions, _upper_triangle


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sym(A: np.ndarray) -> np.ndarray:
    """Return a symmetric version of A (upper tri mirrored)."""
    return np.triu(A, k=1) + np.triu(A, k=1).T


def _make_adj(p: int, edges: list[tuple[int, int]]) -> np.ndarray:
    """Build a symmetric binary adjacency matrix with the given edges."""
    A = np.zeros((p, p), dtype=int)
    for i, j in edges:
        A[i, j] = 1
        A[j, i] = 1
    return A


EXPECTED_KEYS = {"aupr", "auroc", "f1", "f1_opt", "mcc", "shd",
                 "precision", "recall", "tp", "fp", "tn", "fn"}


# ---------------------------------------------------------------------------
# Test 1 — Perfect prediction
# ---------------------------------------------------------------------------

class TestPerfectPrediction:
    def setup_method(self):
        p = 5
        self.adj_true = _make_adj(p, [(0, 1), (0, 2), (1, 3)])
        self.adj_pred = self.adj_true.copy()
        self.result = evaluate_predictions(self.adj_pred, self.adj_true)

    def test_tp_equals_num_edges(self):
        assert self.result["tp"] == 3

    def test_fp_zero(self):
        assert self.result["fp"] == 0

    def test_fn_zero(self):
        assert self.result["fn"] == 0

    def test_mcc_one(self):
        assert self.result["mcc"] == pytest.approx(1.0, abs=1e-6)

    def test_f1_one(self):
        assert self.result["f1"] == pytest.approx(1.0, abs=1e-6)

    def test_shd_zero(self):
        assert self.result["shd"] == 0

    def test_aupr_one(self):
        assert self.result["aupr"] == pytest.approx(1.0, abs=1e-6)

    def test_auroc_one(self):
        assert self.result["auroc"] == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Test 2 — All-wrong prediction (fully flipped)
# ---------------------------------------------------------------------------

class TestAllWrongPrediction:
    def setup_method(self):
        # 4-node graph; upper triangle has 6 pairs
        p = 4
        # true: edges (0,1), (0,2) → 2 true positives possible
        self.adj_true = _make_adj(p, [(0, 1), (0, 2)])
        # pred: complement — predict every non-edge as edge and vice versa
        upper = np.triu(np.ones((p, p), dtype=int), k=1)
        self.adj_pred = _sym(upper - np.triu(self.adj_true, k=1))
        self.result = evaluate_predictions(self.adj_pred, self.adj_true)

    def test_tp_zero(self):
        assert self.result["tp"] == 0

    def test_fn_equals_true_edges(self):
        n_true = int(_upper_triangle(self.adj_true).sum())
        assert self.result["fn"] == n_true

    def test_shd_equals_twice_true_edges(self):
        # All true edges are missed (fn) + all non-edges predicted (fp); total mismatches = all pairs
        n_pairs = self.adj_true.shape[0] * (self.adj_true.shape[0] - 1) // 2
        assert self.result["shd"] == n_pairs

    def test_mcc_not_positive(self):
        # MCC must be <= 0 for a fully flipped prediction
        assert self.result["mcc"] <= 0.0


# ---------------------------------------------------------------------------
# Test 3 — SHD correctness
# ---------------------------------------------------------------------------

class TestSHDCorrectness:
    def test_known_mismatch_count(self):
        p = 5
        adj_true = _make_adj(p, [(0, 1), (0, 2), (1, 3), (2, 4)])
        # pred misses (1,3) and adds a false edge (3,4)
        adj_pred = _make_adj(p, [(0, 1), (0, 2), (2, 4), (3, 4)])
        result = evaluate_predictions(adj_pred, adj_true)
        # 1 false negative ((1,3) missed) + 1 false positive ((3,4) added) = 2
        assert result["shd"] == 2

    def test_identical_matrices_shd_zero(self):
        adj = _make_adj(4, [(0, 2), (1, 3)])
        assert evaluate_predictions(adj, adj)["shd"] == 0

    def test_empty_pred_shd_equals_true_edges(self):
        p = 4
        adj_true = _make_adj(p, [(0, 1), (1, 2)])
        adj_pred = np.zeros((p, p), dtype=int)
        result = evaluate_predictions(adj_pred, adj_true)
        assert result["shd"] == 2


# ---------------------------------------------------------------------------
# Test 4 — MCC formula verification
# ---------------------------------------------------------------------------

class TestMCCFormula:
    def test_mcc_matches_manual_calculation(self):
        # Build a 5-node graph so we control TP/FP/TN/FN exactly
        p = 5
        # upper triangle has 10 pairs
        # true edges: (0,1), (0,2), (0,3)   → 3 positives, 7 negatives
        adj_true = _make_adj(p, [(0, 1), (0, 2), (0, 3)])
        # pred edges: (0,1), (0,2), (1,2)   → hits 2 true, misses 1 true, adds 1 false
        adj_pred = _make_adj(p, [(0, 1), (0, 2), (1, 2)])
        result = evaluate_predictions(adj_pred, adj_true)

        tp, fp, tn, fn = result["tp"], result["fp"], result["tn"], result["fn"]
        assert tp == 2
        assert fp == 1
        assert fn == 1
        assert tn == 6

        denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        expected_mcc = (tp * tn - fp * fn) / denom
        assert result["mcc"] == pytest.approx(expected_mcc, abs=1e-6)

    def test_mcc_zero_when_denominator_zero(self):
        # all-zero pred, all-zero true → denom = 0 → mcc = 0.0
        p = 3
        adj_zero = np.zeros((p, p), dtype=int)
        result = evaluate_predictions(adj_zero, adj_zero)
        assert result["mcc"] == 0.0


# ---------------------------------------------------------------------------
# Test 5 — No-positive ground truth
# ---------------------------------------------------------------------------

class TestNoPositiveGroundTruth:
    def setup_method(self):
        p = 4
        adj_true = np.zeros((p, p), dtype=int)
        adj_pred = _make_adj(p, [(0, 1)])
        self.result = evaluate_predictions(adj_pred, adj_true)

    def test_aupr_nan(self):
        assert math.isnan(self.result["aupr"])

    def test_auroc_nan(self):
        assert math.isnan(self.result["auroc"])

    def test_f1_opt_nan(self):
        assert math.isnan(self.result["f1_opt"])


# ---------------------------------------------------------------------------
# Test 6 — All-positive ground truth → auroc nan
# ---------------------------------------------------------------------------

class TestAllPositiveGroundTruth:
    def test_auroc_nan_when_all_true(self):
        p = 3
        # All upper-triangle pairs are true edges: (0,1), (0,2), (1,2)
        adj_true = _sym(np.triu(np.ones((p, p), dtype=int), k=1))
        adj_pred = adj_true.copy()
        result = evaluate_predictions(adj_pred, adj_true)
        assert math.isnan(result["auroc"])

    def test_aupr_not_nan_when_all_true(self):
        p = 3
        adj_true = _sym(np.triu(np.ones((p, p), dtype=int), k=1))
        adj_pred = adj_true.copy()
        result = evaluate_predictions(adj_pred, adj_true)
        assert not math.isnan(result["aupr"])


# ---------------------------------------------------------------------------
# Test 7 — With separate scores matrix
# ---------------------------------------------------------------------------

class TestWithScoresMatrix:
    def test_scores_affect_aupr_auroc(self):
        p = 5
        rng = np.random.default_rng(42)
        adj_true = _make_adj(p, [(0, 1), (0, 2), (1, 3)])

        # Binary prediction that is mediocre
        adj_pred = _make_adj(p, [(0, 1), (2, 3), (3, 4)])

        # Continuous scores: higher values on true edges
        scores = rng.uniform(0, 1, (p, p))
        scores = (scores + scores.T) / 2  # symmetrise
        # Boost true-edge scores to give strong ranking signal
        for i, j in [(0, 1), (0, 2), (1, 3)]:
            scores[i, j] = scores[j, i] = 0.95

        result_binary = evaluate_predictions(adj_pred, adj_true)
        result_scored = evaluate_predictions(adj_pred, adj_true, scores=scores)

        # aupr with informative scores should differ from binary aupr
        assert result_binary["aupr"] != pytest.approx(result_scored["aupr"], abs=1e-6)

    def test_scores_do_not_affect_shd_f1_mcc(self):
        p = 5
        rng = np.random.default_rng(7)
        adj_true = _make_adj(p, [(0, 1), (0, 3)])
        adj_pred = _make_adj(p, [(0, 1), (1, 2)])
        scores = rng.uniform(0, 1, (p, p))
        scores = (scores + scores.T) / 2

        r1 = evaluate_predictions(adj_pred, adj_true)
        r2 = evaluate_predictions(adj_pred, adj_true, scores=scores)

        assert r1["shd"] == r2["shd"]
        assert r1["f1"] == pytest.approx(r2["f1"], abs=1e-6)
        assert r1["mcc"] == pytest.approx(r2["mcc"], abs=1e-6)


# ---------------------------------------------------------------------------
# Test 8 — Return keys
# ---------------------------------------------------------------------------

class TestReturnKeys:
    def test_all_expected_keys_present(self):
        p = 4
        adj = _make_adj(p, [(0, 1)])
        result = evaluate_predictions(adj, adj)
        assert EXPECTED_KEYS == set(result.keys())

    def test_no_extra_keys(self):
        p = 4
        adj = _make_adj(p, [(0, 1)])
        result = evaluate_predictions(adj, adj)
        assert set(result.keys()) == EXPECTED_KEYS


# ---------------------------------------------------------------------------
# Test 9 — Asymmetric input: only upper triangle is used
# ---------------------------------------------------------------------------

class TestAsymmetricInputGuard:
    def test_upper_triangle_only_used(self):
        p = 4
        adj_sym = _make_adj(p, [(0, 1), (1, 2)])

        # Corrupt lower triangle; upper triangle unchanged
        adj_corrupt = adj_sym.copy()
        adj_corrupt[1, 0] = 0   # remove lower-tri entry for (0,1)
        adj_corrupt[3, 0] = 1   # add spurious lower-tri entry

        result_sym = evaluate_predictions(adj_sym, adj_sym)
        result_corrupt = evaluate_predictions(adj_corrupt, adj_sym)

        # Results should be identical since only upper triangle is read from adj_pred
        assert result_sym["tp"] == result_corrupt["tp"]
        assert result_sym["fp"] == result_corrupt["fp"]
        assert result_sym["shd"] == result_corrupt["shd"]


# ---------------------------------------------------------------------------
# Test 10 — 2×2 matrix sanity
# ---------------------------------------------------------------------------

class TestTwoByTwoMatrix:
    """A 2×2 matrix has exactly one upper-triangle entry: position (0,1)."""

    def test_edge_present_correctly_predicted(self):
        adj = np.array([[0, 1], [1, 0]], dtype=int)
        result = evaluate_predictions(adj, adj)
        assert result["tp"] == 1
        assert result["fp"] == 0
        assert result["tn"] == 0
        assert result["fn"] == 0
        assert result["shd"] == 0

    def test_edge_present_not_predicted(self):
        adj_true = np.array([[0, 1], [1, 0]], dtype=int)
        adj_pred = np.zeros((2, 2), dtype=int)
        result = evaluate_predictions(adj_pred, adj_true)
        assert result["tp"] == 0
        assert result["fn"] == 1
        assert result["shd"] == 1

    def test_no_edge_correctly_predicted(self):
        adj = np.zeros((2, 2), dtype=int)
        result = evaluate_predictions(adj, adj)
        assert result["tn"] == 1
        assert result["tp"] == 0
        assert result["shd"] == 0
        # No positives → nan for rank-based metrics
        assert math.isnan(result["aupr"])
        assert math.isnan(result["auroc"])

    def test_no_edge_falsely_predicted(self):
        adj_true = np.zeros((2, 2), dtype=int)
        adj_pred = np.array([[0, 1], [1, 0]], dtype=int)
        result = evaluate_predictions(adj_pred, adj_true)
        assert result["fp"] == 1
        assert result["tn"] == 0
        assert result["shd"] == 1
