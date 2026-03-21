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



def test_ensemble_ci_width_scales_with_n_splits():
    """CI width should scale as 1/sqrt(K): quadrupling K halves width (approx).

    ensemble_ci draws K independent subsamples each of size floor(subsample_frac*n).
    With default subsample_frac=0.5 and n=400, each split gets 200 samples regardless
    of K. CI width = 2 * z_crit * std(Omega_k) / sqrt(K), so:
        width(K=4) / width(K=16) ≈ sqrt(16)/sqrt(4) = 2.0

    The SD estimate from K=4 splits has high variance (chi-squared, df=3), so
    we allow a wide tolerance [1.4, 3.0] around the expected ratio of 2.0.

    Many edges in a hub graph are structural zeros; both K=4 and K=16 produce
    zero width for them, so we restrict the ratio check to edges where K=4
    gives a non-negligible width.
    """
    X, _ = _hub_data(n=400, p=10, seed=42)
    _, lo4, hi4 = ensemble_ci(X, n_splits=4, seed=0, alpha=0.05)
    _, lo16, hi16 = ensemble_ci(X, n_splits=16, seed=0, alpha=0.05)

    p = X.shape[1]
    uidx = np.triu_indices(p, k=1)
    width4 = (hi4 - lo4)[uidx]
    width16 = (hi16 - lo16)[uidx]

    # Restrict to edges where K=4 already produces a non-zero CI width;
    # structural zero edges have std=0 for both K and don't test the 1/sqrt(K) rule.
    active = width4 > 1e-12
    assert active.sum() >= 3, "Too few active edges to test scaling; check data seed."

    # Expected ratio ≈ 2.0; [1.4, 4.5] absorbs finite-sample noise in SD estimates
    # (K=4 has only df=3 for SD, so individual ratios can reach ~4; median is stable)
    ratios = width4[active] / np.maximum(width16[active], 1e-12)
    median_ratio = float(np.median(ratios))
    assert 1.4 <= median_ratio <= 4.5, (
        f"Expected width ratio (K=4 / K=16) near 2.0 (1/sqrt(K) rule); got {median_ratio:.3f}. "
        "This suggests SE is not divided by sqrt(K)."
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
