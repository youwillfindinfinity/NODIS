"""
Unit tests for confidence interval functions.
"""

import numpy as np
import pytest
from nodis.inference.confidence import asymptotic_ci, ensemble_ci
from nodis.estimators.desparsified import DesparifiedGGM


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

def _hub_data(n: int = 120, p: int = 12, seed: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Return (X, true_Theta) for a hub GGM."""
    rng = np.random.default_rng(seed)
    Theta = np.eye(p) * 2.0
    for j in range(1, p):
        Theta[0, j] = Theta[j, 0] = 0.25
    min_eig = np.linalg.eigvalsh(Theta).min()
    if min_eig <= 0:
        Theta += (abs(min_eig) + 0.1) * np.eye(p)
    Sigma = np.linalg.inv(Theta)
    X = rng.multivariate_normal(np.zeros(p), Sigma, size=n)
    return X, Theta


# ---------------------------------------------------------------------------
# asymptotic_ci (smoke tests — main logic already tested implicitly elsewhere)
# ---------------------------------------------------------------------------

def test_asymptotic_ci_shape():
    X, _ = _hub_data()
    est = DesparifiedGGM().fit(X)
    lower, upper = est.confidence_intervals(alpha=0.05)
    p = X.shape[1]
    assert lower.shape == (p, p)
    assert upper.shape == (p, p)


def test_asymptotic_ci_lower_le_upper():
    X, _ = _hub_data()
    est = DesparifiedGGM().fit(X)
    lower, upper = est.confidence_intervals(alpha=0.05)
    assert np.all(lower <= upper + 1e-12)


def test_asymptotic_ci_diagonal_zero():
    X, _ = _hub_data()
    est = DesparifiedGGM().fit(X)
    lower, upper = est.confidence_intervals()
    np.testing.assert_array_equal(np.diag(lower), 0.0)
    np.testing.assert_array_equal(np.diag(upper), 0.0)


# ---------------------------------------------------------------------------
# ensemble_ci — output properties
# ---------------------------------------------------------------------------

def test_ensemble_ci_shape():
    X, _ = _hub_data()
    omega, lower, upper = ensemble_ci(X, n_splits=5, seed=0)
    p = X.shape[1]
    assert omega.shape == (p, p)
    assert lower.shape == (p, p)
    assert upper.shape == (p, p)


def test_ensemble_ci_lower_le_upper():
    X, _ = _hub_data()
    _, lower, upper = ensemble_ci(X, n_splits=5, seed=0)
    assert np.all(lower <= upper + 1e-12)


def test_ensemble_ci_symmetric():
    X, _ = _hub_data()
    omega, lower, upper = ensemble_ci(X, n_splits=5, seed=0)
    np.testing.assert_allclose(omega, omega.T, atol=1e-10)
    np.testing.assert_allclose(lower, lower.T, atol=1e-10)
    np.testing.assert_allclose(upper, upper.T, atol=1e-10)


def test_ensemble_ci_diagonal_zero():
    X, _ = _hub_data()
    omega, lower, upper = ensemble_ci(X, n_splits=5, seed=0)
    np.testing.assert_array_equal(np.diag(omega), 0.0)
    np.testing.assert_array_equal(np.diag(lower), 0.0)
    np.testing.assert_array_equal(np.diag(upper), 0.0)


@pytest.mark.slow
def test_ensemble_ci_wider_than_asymptotic_for_most_edges():
    """Ensemble CIs should be wider than asymptotic for >= 70% of off-diagonal entries.

    Marked slow: uses n_splits=25 for stable SE estimates.
    Ensemble CIs use a mean-of-subsamples estimand and empirical SE, so they
    are expected to be wider than single-fit asymptotic CIs for most edges
    at moderate n/p, but the threshold is conservative (70%) to avoid flakiness.
    """
    X, _ = _hub_data()
    est = DesparifiedGGM().fit(X)
    a_lower, a_upper = est.confidence_intervals(alpha=0.05)
    _, e_lower, e_upper = ensemble_ci(X, n_splits=25, seed=0, alpha=0.05)

    a_width = a_upper - a_lower
    e_width = e_upper - e_lower

    p = X.shape[1]
    uidx = np.triu_indices(p, k=1)
    frac_wider = np.mean(e_width[uidx] >= a_width[uidx] - 1e-10)
    assert frac_wider >= 0.70, (
        f"Expected ensemble CIs to be wider for >= 70% of edges; got {frac_wider:.1%}"
    )


def test_ensemble_ci_reproducible():
    X, _ = _hub_data()
    omega1, _, _ = ensemble_ci(X, n_splits=5, seed=99)
    omega2, _, _ = ensemble_ci(X, n_splits=5, seed=99)
    np.testing.assert_allclose(omega1, omega2, atol=1e-12)


def test_ensemble_ci_invalid_n_splits_raises():
    X, _ = _hub_data()
    with pytest.raises(ValueError, match="n_splits"):
        ensemble_ci(X, n_splits=1, seed=0)


def test_ensemble_ci_invalid_subsample_frac_raises():
    X, _ = _hub_data()
    with pytest.raises(ValueError, match="subsample_frac"):
        ensemble_ci(X, subsample_frac=1.5, seed=0)


# ---------------------------------------------------------------------------
# DesparifiedGGM.confidence_intervals() — method parameter
# ---------------------------------------------------------------------------

def test_desparsified_ensemble_method():
    """DesparifiedGGM.confidence_intervals(method='ensemble') must accept X."""
    X, _ = _hub_data()
    est = DesparifiedGGM().fit(X)
    omega, lower, upper = est.confidence_intervals(alpha=0.05,
                                                    method='ensemble',
                                                    X=X, n_splits=5)
    assert omega.shape == (X.shape[1], X.shape[1])


def test_desparsified_ensemble_without_X_raises():
    """method='ensemble' without X should raise ValueError."""
    X, _ = _hub_data()
    est = DesparifiedGGM().fit(X)
    with pytest.raises(ValueError, match="X must be provided"):
        est.confidence_intervals(method='ensemble')


def test_desparsified_invalid_method_raises():
    X, _ = _hub_data()
    est = DesparifiedGGM().fit(X)
    with pytest.raises(ValueError, match="method"):
        est.confidence_intervals(method='bootstrap')
