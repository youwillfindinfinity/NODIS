"""
Unit tests for nodis/estimators/prior_utils.py.
"""
import numpy as np
import pytest

from nodis.estimators.prior_utils import build_corr_prior, build_noisy_oracle_prior

RNG = np.random.default_rng(42)
X = RNG.standard_normal((50, 10))  # (n, p)

P = 8
ADJ_TRUE = np.zeros((P, P), dtype=int)
for i, j in [(0, 1), (1, 2), (3, 4), (5, 6)]:
    ADJ_TRUE[i, j] = 1
    ADJ_TRUE[j, i] = 1


# ---------------------------------------------------------------------------
# build_corr_prior
# ---------------------------------------------------------------------------

def test_build_corr_prior_shape():
    result = build_corr_prior(X)
    assert result.shape == (X.shape[1], X.shape[1])


def test_build_corr_prior_diagonal_zero():
    result = build_corr_prior(X)
    assert np.all(np.diag(result) == 0.0)


def test_build_corr_prior_range():
    result = build_corr_prior(X)
    assert result.min() >= 0.0
    assert result.max() <= 1.0


def test_build_corr_prior_symmetric():
    result = build_corr_prior(X)
    np.testing.assert_allclose(result, result.T, atol=1e-12)


def test_build_corr_prior_gamma_effect():
    """Higher gamma → sparser prior (smaller sum of off-diagonal entries)."""
    p1 = build_corr_prior(X, gamma=1.0)
    p4 = build_corr_prior(X, gamma=4.0)
    assert p4.sum() < p1.sum()


# ---------------------------------------------------------------------------
# build_noisy_oracle_prior
# ---------------------------------------------------------------------------

def test_noisy_oracle_shape():
    result = build_noisy_oracle_prior(ADJ_TRUE, noise=0.2, seed=0)
    assert result.shape == (P, P)


def test_noisy_oracle_diagonal_zero():
    result = build_noisy_oracle_prior(ADJ_TRUE, noise=0.2, seed=0)
    assert np.all(np.diag(result) == 0.0)


def test_noisy_oracle_symmetric():
    result = build_noisy_oracle_prior(ADJ_TRUE, noise=0.2, seed=0)
    np.testing.assert_array_equal(result, result.T)


def test_noisy_oracle_no_noise():
    """noise=0.0 → output identical to adj_true."""
    result = build_noisy_oracle_prior(ADJ_TRUE, noise=0.0, seed=0)
    np.testing.assert_array_equal(result, ADJ_TRUE.astype(float))


def test_noisy_oracle_half_noise_differs():
    """noise=0.4 with seed=0 → output differs from adj_true."""
    result = build_noisy_oracle_prior(ADJ_TRUE, noise=0.4, seed=0)
    assert not np.array_equal(result, ADJ_TRUE.astype(float))


def test_noisy_oracle_reproducible():
    """Same seed → same result."""
    r1 = build_noisy_oracle_prior(ADJ_TRUE, noise=0.2, seed=7)
    r2 = build_noisy_oracle_prior(ADJ_TRUE, noise=0.2, seed=7)
    np.testing.assert_array_equal(r1, r2)


def test_noisy_oracle_different_seeds():
    """Different seeds → different results."""
    r1 = build_noisy_oracle_prior(ADJ_TRUE, noise=0.3, seed=1)
    r2 = build_noisy_oracle_prior(ADJ_TRUE, noise=0.3, seed=99)
    assert not np.array_equal(r1, r2)
