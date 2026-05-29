"""
Unit tests for nodis.benchmark.runner.

Covers: desparsified path, glasso path, multiple estimators, error handling,
parallel execution, method name preservation, no-attribute error capture,
and wall_seconds typing.
"""

from __future__ import annotations

import numpy as np
import pytest

from nodis.benchmark.runner import run_benchmark


# ---------------------------------------------------------------------------
# Fixtures / shared data
# ---------------------------------------------------------------------------

N, P = 50, 10

@pytest.fixture
def X():
    rng = np.random.default_rng(0)
    return rng.standard_normal((N, P))


@pytest.fixture
def adj_true():
    return np.zeros((P, P), dtype=int)


# Expected metric keys returned by evaluate_predictions
METRIC_KEYS = {"aupr", "auroc", "f1", "f1_opt", "mcc", "shd",
               "precision", "recall", "tp", "fp", "tn", "fn"}


# ---------------------------------------------------------------------------
# Mock estimators
# ---------------------------------------------------------------------------

class MockDesparifiedEstimator:
    """Simulates DesparifiedGGM: sets result_.z_scores and get_adjacency(alpha)."""

    def fit(self, X):
        p = X.shape[1]
        self.result_ = type("R", (), {"z_scores": np.zeros((p, p))})()
        self._p = p
        return self

    def get_adjacency(self, alpha=0.05):
        return np.zeros((self._p, self._p), dtype=int)


class MockGlassoEstimator:
    """Simulates GLasso: sets precision_ and get_adjacency(threshold)."""

    def fit(self, X):
        p = X.shape[1]
        self.precision_ = np.eye(p)
        return self

    def get_adjacency(self, threshold=0.0):
        p = self.precision_.shape[0]
        return np.zeros((p, p), dtype=int)


class MockErrorEstimator:
    """Always raises RuntimeError in fit()."""

    def fit(self, X):
        raise RuntimeError("intentional test failure")


class MockNoAttributeEstimator:
    """Has neither result_ nor precision_ after fit()."""

    def fit(self, X):
        return self

    def get_adjacency(self):
        return np.zeros((P, P), dtype=int)


# ---------------------------------------------------------------------------
# Test 1 — Single estimator, desparsified path
# ---------------------------------------------------------------------------

class TestDesparifiedPath:
    def test_result_has_method_key(self, X, adj_true):
        results = run_benchmark(
            [(MockDesparifiedEstimator, {}, "desparsified")],
            X, adj_true,
        )
        assert results[0]["method"] == "desparsified"

    def test_error_is_none(self, X, adj_true):
        results = run_benchmark(
            [(MockDesparifiedEstimator, {}, "desparsified")],
            X, adj_true,
        )
        assert results[0]["error"] is None

    def test_wall_seconds_present(self, X, adj_true):
        results = run_benchmark(
            [(MockDesparifiedEstimator, {}, "desparsified")],
            X, adj_true,
        )
        assert "wall_seconds" in results[0]

    def test_all_metric_keys_present(self, X, adj_true):
        results = run_benchmark(
            [(MockDesparifiedEstimator, {}, "desparsified")],
            X, adj_true,
        )
        assert METRIC_KEYS.issubset(results[0].keys())


# ---------------------------------------------------------------------------
# Test 2 — Single estimator, glasso path
# ---------------------------------------------------------------------------

class TestGlassoPath:
    def test_result_has_method_key(self, X, adj_true):
        results = run_benchmark(
            [(MockGlassoEstimator, {}, "glasso")],
            X, adj_true,
        )
        assert results[0]["method"] == "glasso"

    def test_error_is_none(self, X, adj_true):
        results = run_benchmark(
            [(MockGlassoEstimator, {}, "glasso")],
            X, adj_true,
        )
        assert results[0]["error"] is None

    def test_wall_seconds_present(self, X, adj_true):
        results = run_benchmark(
            [(MockGlassoEstimator, {}, "glasso")],
            X, adj_true,
        )
        assert "wall_seconds" in results[0]

    def test_all_metric_keys_present(self, X, adj_true):
        results = run_benchmark(
            [(MockGlassoEstimator, {}, "glasso")],
            X, adj_true,
        )
        assert METRIC_KEYS.issubset(results[0].keys())


# ---------------------------------------------------------------------------
# Test 3 — Multiple estimators
# ---------------------------------------------------------------------------

class TestMultipleEstimators:
    def test_two_results_returned(self, X, adj_true):
        estimators = [
            (MockDesparifiedEstimator, {}, "desparsified"),
            (MockGlassoEstimator, {}, "glasso"),
        ]
        results = run_benchmark(estimators, X, adj_true)
        assert len(results) == 2

    def test_method_names_match(self, X, adj_true):
        estimators = [
            (MockDesparifiedEstimator, {}, "desparsified"),
            (MockGlassoEstimator, {}, "glasso"),
        ]
        results = run_benchmark(estimators, X, adj_true)
        names = {r["method"] for r in results}
        assert names == {"desparsified", "glasso"}

    def test_each_result_has_metric_keys(self, X, adj_true):
        estimators = [
            (MockDesparifiedEstimator, {}, "desparsified"),
            (MockGlassoEstimator, {}, "glasso"),
        ]
        results = run_benchmark(estimators, X, adj_true)
        for r in results:
            assert METRIC_KEYS.issubset(r.keys())


# ---------------------------------------------------------------------------
# Test 4 — Error handling
# ---------------------------------------------------------------------------

class TestErrorHandling:
    def test_error_is_not_none_on_failure(self, X, adj_true):
        results = run_benchmark(
            [(MockErrorEstimator, {}, "broken")],
            X, adj_true,
        )
        assert results[0]["error"] is not None

    def test_error_contains_traceback_text(self, X, adj_true):
        results = run_benchmark(
            [(MockErrorEstimator, {}, "broken")],
            X, adj_true,
        )
        assert "RuntimeError" in results[0]["error"]

    def test_wall_seconds_still_present_on_error(self, X, adj_true):
        results = run_benchmark(
            [(MockErrorEstimator, {}, "broken")],
            X, adj_true,
        )
        assert "wall_seconds" in results[0]

    def test_does_not_propagate_exception(self, X, adj_true):
        # Should not raise; error captured in result dict
        results = run_benchmark(
            [(MockErrorEstimator, {}, "broken")],
            X, adj_true,
        )
        assert len(results) == 1


# ---------------------------------------------------------------------------
# Test 5 — n_jobs=-1 (parallel)
# ---------------------------------------------------------------------------

class TestParallelExecution:
    def test_results_correct_with_n_jobs_minus_one(self, X, adj_true):
        estimators = [
            (MockDesparifiedEstimator, {}, "desparsified"),
            (MockGlassoEstimator, {}, "glasso"),
        ]
        results = run_benchmark(estimators, X, adj_true, n_jobs=-1)
        assert len(results) == 2
        names = {r["method"] for r in results}
        assert names == {"desparsified", "glasso"}

    def test_errors_captured_in_parallel(self, X, adj_true):
        estimators = [
            (MockErrorEstimator, {}, "broken"),
            (MockDesparifiedEstimator, {}, "ok"),
        ]
        results = run_benchmark(estimators, X, adj_true, n_jobs=-1)
        by_name = {r["method"]: r for r in results}
        assert by_name["broken"]["error"] is not None
        assert by_name["ok"]["error"] is None


# ---------------------------------------------------------------------------
# Test 6 — Method names preserved
# ---------------------------------------------------------------------------

class TestMethodNamesPreserved:
    def test_single_method_name(self, X, adj_true):
        results = run_benchmark(
            [(MockGlassoEstimator, {}, "my_glasso_v2")],
            X, adj_true,
        )
        assert results[0]["method"] == "my_glasso_v2"

    def test_names_with_spaces_and_special_chars(self, X, adj_true):
        name = "method A (test)"
        results = run_benchmark(
            [(MockDesparifiedEstimator, {}, name)],
            X, adj_true,
        )
        assert results[0]["method"] == name


# ---------------------------------------------------------------------------
# Test 7 — No result_ or precision_ attribute raises captured error
# ---------------------------------------------------------------------------

class TestNoAttributeError:
    def test_error_captured_not_propagated(self, X, adj_true):
        results = run_benchmark(
            [(MockNoAttributeEstimator, {}, "no_attr")],
            X, adj_true,
        )
        assert results[0]["error"] is not None

    def test_error_mentions_attribute(self, X, adj_true):
        results = run_benchmark(
            [(MockNoAttributeEstimator, {}, "no_attr")],
            X, adj_true,
        )
        assert "AttributeError" in results[0]["error"]

    def test_wall_seconds_still_present(self, X, adj_true):
        results = run_benchmark(
            [(MockNoAttributeEstimator, {}, "no_attr")],
            X, adj_true,
        )
        assert "wall_seconds" in results[0]


# ---------------------------------------------------------------------------
# Test 8 — wall_seconds is a non-negative float
# ---------------------------------------------------------------------------

class TestWallSeconds:
    def test_wall_seconds_is_float(self, X, adj_true):
        results = run_benchmark(
            [(MockDesparifiedEstimator, {}, "desparsified")],
            X, adj_true,
        )
        assert isinstance(results[0]["wall_seconds"], float)

    def test_wall_seconds_non_negative(self, X, adj_true):
        results = run_benchmark(
            [(MockDesparifiedEstimator, {}, "desparsified")],
            X, adj_true,
        )
        assert results[0]["wall_seconds"] >= 0.0

    def test_wall_seconds_non_negative_on_error(self, X, adj_true):
        results = run_benchmark(
            [(MockErrorEstimator, {}, "broken")],
            X, adj_true,
        )
        assert results[0]["wall_seconds"] >= 0.0
