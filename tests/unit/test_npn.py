"""
Unit tests for the NPN (nonparanormal) preprocessing transform.
Covers both the shrinkage variant and the SKEPTIC variant (Liu et al. 2012).
"""

import numpy as np
import pytest
from scipy.stats import normaltest

from nodis.preprocess.npn import npn_shrinkage, npn_skeptic, npn_transform


@pytest.fixture(scope="module")
def lognormal_data():
    """Log-normal data — strongly non-Gaussian before NPN."""
    rng = np.random.default_rng(42)
    return np.exp(rng.standard_normal((500, 40)))


def test_output_shape(lognormal_data):
    out = npn_shrinkage(lognormal_data)
    assert out.shape == lognormal_data.shape


def test_output_dtype(lognormal_data):
    out = npn_shrinkage(lognormal_data)
    assert out.dtype == float


def test_no_nan_or_inf(lognormal_data):
    out = npn_shrinkage(lognormal_data)
    assert np.isfinite(out).all(), "NPN output contains NaN or Inf values"


def test_marginals_approx_normal(lognormal_data):
    """After NPN, each marginal should not be significantly non-normal.

    We use D'Agostino–Pearson test at alpha=0.001 and require ≥ 90% of
    columns to pass (finite-sample deviations are expected).
    """
    out = npn_shrinkage(lognormal_data)
    n_pass = 0
    for j in range(out.shape[1]):
        _, p = normaltest(out[:, j])
        if p > 0.001:
            n_pass += 1
    pass_rate = n_pass / out.shape[1]
    assert pass_rate >= 0.90, (
        f"Only {pass_rate:.0%} of columns passed normality test after NPN. "
        "Expected ≥ 90%."
    )


def test_standard_normal_data_unchanged_in_shape():
    """Standard normal data should pass through without structural changes."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((100, 10))
    out = npn_shrinkage(X)
    assert out.shape == X.shape


def test_custom_truncation():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((200, 20))
    out = npn_shrinkage(X, truncation=0.01)
    assert np.isfinite(out).all()


def test_invalid_input_raises():
    with pytest.raises(ValueError, match="2-D"):
        npn_shrinkage(np.ones(10))


def test_tie_breaking_average_rank():
    """Tied values must receive average ranks, matching R's rank(ties.method='average').

    For vector [1, 1, 2, 3, 3, 4]:
      - Values 1 at positions 0,1 → rank 1.5 (average of ranks 1,2)
      - Value  2 at position  2  → rank 3
      - Values 3 at positions 3,4 → rank 4.5 (average of ranks 4,5)
      - Value  4 at position  5  → rank 6

    Verify that both tied positions receive identical NPN-transformed values.
    """
    X = np.array([[1, 1, 2, 3, 3, 4]], dtype=float).T  # (6, 1)
    out = npn_shrinkage(X)

    # Tied pairs must be identical after transform
    assert out[0, 0] == out[1, 0], (
        f"Tied values at rank 1.5 produced different NPN outputs: "
        f"{out[0, 0]:.8f} vs {out[1, 0]:.8f}. "
        "Check that rankdata(method='average') is used."
    )
    assert out[3, 0] == out[4, 0], (
        f"Tied values at rank 4.5 produced different NPN outputs: "
        f"{out[3, 0]:.8f} vs {out[4, 0]:.8f}. "
        "Check that rankdata(method='average') is used."
    )

    # Strict ordering must hold for non-tied values
    assert out[1, 0] < out[2, 0] < out[3, 0] < out[5, 0], (
        "Non-tied values are not strictly ordered after NPN transform."
    )


# ---------------------------------------------------------------------------
# NPN SKEPTIC tests
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def lognormal_data_large():
    """Larger log-normal matrix for SKEPTIC tests (needs more samples for Kendall)."""
    rng = np.random.default_rng(7)
    return np.exp(rng.standard_normal((300, 20)))


def test_skeptic_output_shape(lognormal_data_large):
    out = npn_skeptic(lognormal_data_large)
    assert out.shape == lognormal_data_large.shape


def test_skeptic_output_dtype(lognormal_data_large):
    out = npn_skeptic(lognormal_data_large)
    assert out.dtype == float


def test_skeptic_no_nan_or_inf(lognormal_data_large):
    out = npn_skeptic(lognormal_data_large)
    assert np.isfinite(out).all(), "SKEPTIC output contains NaN or Inf"


def test_skeptic_spearman_variant(lognormal_data_large):
    out = npn_skeptic(lognormal_data_large, corr_type="spearman")
    assert out.shape == lognormal_data_large.shape
    assert np.isfinite(out).all()


def test_skeptic_invalid_corr_type():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((50, 5))
    with pytest.raises(ValueError, match="corr_type"):
        npn_skeptic(X, corr_type="pearson")


def test_skeptic_invalid_input_raises():
    with pytest.raises(ValueError, match="2-D"):
        npn_skeptic(np.ones(10))


def test_skeptic_corr_structure_preserved():
    """After SKEPTIC, empirical Pearson correlations should reflect the
    latent Gaussian structure: a strongly correlated pair in X should
    remain positively correlated in the output.
    """
    rng = np.random.default_rng(42)
    n, p = 200, 10
    Z = rng.standard_normal((n, p))
    # Introduce strong positive correlation between columns 0 and 1
    Z[:, 1] = 0.9 * Z[:, 0] + 0.1 * rng.standard_normal(n)
    X = np.exp(Z)  # log-normal
    out = npn_skeptic(X, corr_type="kendall")
    emp_corr = np.corrcoef(out.T)[0, 1]
    assert emp_corr > 0.5, (
        f"Expected positive correlation between columns 0 and 1 after SKEPTIC; "
        f"got {emp_corr:.4f}."
    )


# ---------------------------------------------------------------------------
# npn_transform dispatcher tests
# ---------------------------------------------------------------------------

def test_npn_transform_shrinkage_dispatches(lognormal_data):
    out_direct = npn_shrinkage(lognormal_data)
    out_dispatch = npn_transform(lognormal_data, method="shrinkage")
    np.testing.assert_array_equal(out_direct, out_dispatch)


def test_npn_transform_skeptic_dispatches(lognormal_data_large):
    out_direct = npn_skeptic(lognormal_data_large)
    out_dispatch = npn_transform(lognormal_data_large, method="skeptic")
    np.testing.assert_array_equal(out_direct, out_dispatch)


def test_npn_transform_invalid_method():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((50, 5))
    with pytest.raises(ValueError, match="method"):
        npn_transform(X, method="unknown")
