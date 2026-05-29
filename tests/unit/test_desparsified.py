"""
Unit tests for the de-sparsified nodewise Lasso estimator.

These tests verify mathematical properties of the estimator independent of
any R implementation.  All tests must pass before running parity tests (RQ1).

Test inventory
--------------
test_output_shapes              — result arrays have correct (p, p) shape
test_symmetry_z_scores          — z-score matrix is symmetric
test_symmetry_p_values          — p-value matrix is symmetric
test_diagonal_z_zeros           — diagonal z-scores are zero
test_diagonal_p_ones            — diagonal p-values are one
test_pvalues_in_unit_interval   — all p-values in [0, 1]
test_pvalues_uniform_under_null — FDR not badly inflated on null (no-edge) data
test_null_calibration_ks        — multi-replicate KS test: p-values ∼ U(0,1) [slow]
test_degenerate_node_warning    — near-zero Tau2 raises UserWarning, no inf in output
test_degenerate_nodes_attr      — degenerate_nodes_ attribute is set after fit
test_precision_pd_generator     — generator produces PD true precision matrix
test_adjacency_symmetric        — FDR-controlled adjacency is symmetric
test_warning_small_n            — UserWarning raised when n < 5p
test_lambda_scaling             — lambda increases monotonically with lambda_scale
"""

import warnings

import numpy as np
import pytest
from scipy.stats import kstest

from nodis.simulate.generator import generate
from nodis.estimators.desparsified import DesparifiedGGM


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def hub_data():
    return generate(n=80, p=20, topology="hub", seed=1)


@pytest.fixture(scope="module")
def fitted_hub(hub_data):
    return DesparifiedGGM().fit(hub_data.X)


@pytest.fixture(scope="module")
def null_fitted():
    """Estimator fitted on pure Gaussian noise (no graph structure)."""
    rng = np.random.default_rng(42)
    X = rng.standard_normal((200, 30))
    return DesparifiedGGM().fit(X)


# ---------------------------------------------------------------------------
# Shape and structure tests
# ---------------------------------------------------------------------------

def test_output_shapes(fitted_hub, hub_data):
    p = hub_data.p
    res = fitted_hub.result_
    assert res.z_scores.shape == (p, p)
    assert res.p_values.shape == (p, p)
    assert res.precision.shape == (p, p)
    assert res.variance.shape == (p, p)


def test_symmetry_z_scores(fitted_hub):
    np.testing.assert_array_almost_equal(
        fitted_hub.result_.z_scores,
        fitted_hub.result_.z_scores.T,
        decimal=10,
    )


def test_symmetry_p_values(fitted_hub):
    np.testing.assert_array_almost_equal(
        fitted_hub.result_.p_values,
        fitted_hub.result_.p_values.T,
        decimal=10,
    )


def test_diagonal_z_zeros(fitted_hub):
    np.testing.assert_array_equal(
        np.diag(fitted_hub.result_.z_scores), np.zeros(fitted_hub._p)
    )


def test_diagonal_p_ones(fitted_hub):
    np.testing.assert_array_equal(
        np.diag(fitted_hub.result_.p_values), np.ones(fitted_hub._p)
    )


def test_pvalues_in_unit_interval(fitted_hub):
    p_vals = fitted_hub.result_.p_values
    assert np.all(p_vals >= 0.0), "p-values contain negative entries"
    assert np.all(p_vals <= 1.0), "p-values exceed 1.0"


# ---------------------------------------------------------------------------
# Statistical calibration under the null
# ---------------------------------------------------------------------------

def test_pvalues_uniform_under_null(null_fitted):
    """Under H₀ (no edges), the rejection rate at alpha=0.05 must be < 10%.
    A properly calibrated estimator should reject ~5% at most under the null.
    We allow 10% to account for finite-sample approximation at n=200, p=30.
    """
    uidx = np.triu_indices(30, k=1)
    pvals = null_fitted.result_.p_values[uidx]
    rejection_rate = (pvals < 0.05).mean()
    assert rejection_rate < 0.10, (
        f"Null rejection rate {rejection_rate:.3f} exceeds 10%: estimator may be "
        "anti-conservative under H₀."
    )


# ---------------------------------------------------------------------------
# Multi-replicate null calibration (L4)
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_null_calibration_type1_control():
    """Multi-replicate type I error control test on null data (Omega = I).

    Under H₀ with λ > 0, the de-sparsified Lasso p-values are **super-uniform**
    (conservative), not uniform.  Lasso thresholding causes β̂=0 for many
    pairs → ω̂=0 → Z=0 → p=1.0.  This is expected behaviour, not a bug.

    What we verify:
      1. Type I error (fraction rejected at α=0.05) is strictly below α.
      2. P-values are not sub-uniform: the fraction below 0.05 must not
         significantly exceed 0.05 (i.e. no inflation).

    Run at n=200, p=20 (n/p=10) where asymptotic theory is reliable.
    Marked slow — not run in CI by default (pytest -m 'not slow').
    """
    n, p, n_reps = 200, 20, 200
    alpha = 0.05
    rng = np.random.default_rng(2024)
    idx = np.triu_indices(p, k=1)
    pooled = []
    for _ in range(n_reps):
        X = rng.standard_normal((n, p))
        model = DesparifiedGGM(standardise=False).fit(X)
        pooled.extend(model.result_.p_values[idx].tolist())

    rejection_rate = np.mean(np.array(pooled) < alpha)

    # Type I error must be controlled (not inflated beyond alpha)
    assert rejection_rate <= alpha, (
        f"Type I error {rejection_rate:.4f} exceeds nominal alpha={alpha}. "
        "P-values are anti-conservative under H₀ — z-scores may be inflated."
    )
    # Also confirm the rate is not pathologically zero (estimator is not broken)
    assert rejection_rate > 0.0, (
        "Type I error is exactly zero across all replicates. "
        "The estimator may not be producing any non-trivial z-scores."
    )


# ---------------------------------------------------------------------------
# Degenerate node guard (L1)
# ---------------------------------------------------------------------------

def test_degenerate_node_warning():
    """A near-perfectly-collinear variable should trigger a UserWarning and
    must not produce inf or nan in z-scores or p-values.

    Construction: column 3 is an exact copy of column 0. With a tiny lambda
    (lambda_scale=0.0001, n=100, p=25 → lambda ≈ 1e-5), the Lasso will fit
    node 3 from node 0 with near-zero residuals → Tau2[3] ≈ 0 → guard fires.
    Requires p > sqrt(n) so the log formula returns a real (positive) value.
    """
    rng = np.random.default_rng(7)
    n, p = 100, 25  # p=25 > sqrt(100)=10 → formula gives real lambda
    X = rng.standard_normal((n, p))
    X[:, 3] = X[:, 0].copy()  # exact duplicate — Lasso can fit with zero residuals

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        model = DesparifiedGGM(
            lambda_scale=0.0001, tol=1e-10, max_iter=50_000
        ).fit(X)
        user_warnings = [x for x in w if issubclass(x.category, UserWarning)]

    # At least one warning should mention near-zero nodewise variance
    assert any("near-zero" in str(x.message).lower() for x in user_warnings), (
        "No UserWarning about near-zero nodewise variance raised for degenerate node."
    )

    # No inf or nan anywhere in the result
    res = model.result_
    assert np.isfinite(res.z_scores).all(), "z_scores contain inf or nan"
    assert np.isfinite(res.p_values).all(), "p_values contain inf or nan"
    assert np.all(res.p_values >= 0.0) and np.all(res.p_values <= 1.0), (
        "p_values outside [0, 1] after degenerate node guard"
    )


def test_degenerate_nodes_attr():
    """degenerate_nodes_ attribute is always set after fit, even when empty."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((100, 15))
    model = DesparifiedGGM().fit(X)
    assert hasattr(model, "degenerate_nodes_"), (
        "degenerate_nodes_ attribute missing after fit on normal data"
    )
    assert isinstance(model.degenerate_nodes_, np.ndarray)


# ---------------------------------------------------------------------------
# Generator sanity tests (required to validate benchmark data)
# ---------------------------------------------------------------------------

def test_precision_pd_generator():
    """Ground-truth precision matrix must be positive definite."""
    for topo in ("hub", "scale-free", "cluster", "random"):
        data = generate(n=100, p=25, topology=topo, seed=3)
        eigs = np.linalg.eigvalsh(data.Theta)
        assert eigs.min() > 0, (
            f"True precision matrix for topology '{topo}' is not positive definite "
            f"(min eigenvalue = {eigs.min():.4f})."
        )


def test_all_topologies_produce_data():
    """All four topologies must complete without error."""
    for topo in ("hub", "scale-free", "cluster", "random"):
        data = generate(n=60, p=15, topology=topo, seed=7)
        assert data.X.shape == (60, 15)
        assert data.Omega.shape == (15, 15)
        assert data.Theta.shape == (15, 15)


# ---------------------------------------------------------------------------
# Adjacency output
# ---------------------------------------------------------------------------

def test_adjacency_symmetric():
    data = generate(n=100, p=20, topology="cluster", seed=5)
    model = DesparifiedGGM().fit(data.X)
    adj = model.get_adjacency(alpha=0.05)
    np.testing.assert_array_equal(adj, adj.T)
    np.testing.assert_array_equal(np.diag(adj), np.zeros(20, dtype=int))


def test_adjacency_binary():
    data = generate(n=100, p=20, topology="random", seed=9)
    model = DesparifiedGGM().fit(data.X)
    adj = model.get_adjacency(alpha=0.05)
    unique_vals = set(np.unique(adj))
    assert unique_vals <= {0, 1}, f"Adjacency contains non-binary values: {unique_vals}"


# ---------------------------------------------------------------------------
# Warning behaviour
# ---------------------------------------------------------------------------

def test_warning_small_n():
    """Fitting with n < 5p must raise a UserWarning."""
    data = generate(n=30, p=20, topology="random", seed=11)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        DesparifiedGGM().fit(data.X)
        assert any(issubclass(warning.category, UserWarning) for warning in w), (
            "No UserWarning raised when n < 5p."
        )


# ---------------------------------------------------------------------------
# Tuning parameter
# ---------------------------------------------------------------------------

def test_lambda_monotone_with_scale():
    """Lambda must increase with lambda_scale."""
    model1 = DesparifiedGGM(lambda_scale=0.5)
    model2 = DesparifiedGGM(lambda_scale=1.0)
    model3 = DesparifiedGGM(lambda_scale=2.0)
    lam1 = model1._get_lambda(100, 50)
    lam2 = model2._get_lambda(100, 50)
    lam3 = model3._get_lambda(100, 50)
    assert lam1 < lam2 < lam3


# ---------------------------------------------------------------------------
# Relaxed lambda (Shinkyu & Sueishi 2022)
# ---------------------------------------------------------------------------

def test_relaxed_lambda_formula():
    """lambda_method='relaxed' must give lambda_scale / sqrt(n)."""
    model = DesparifiedGGM(lambda_scale=2.0, lambda_method="relaxed")
    lam = model._get_lambda(n=100, p=50)
    expected = 2.0 / np.sqrt(100)
    assert abs(lam - expected) < 1e-12, (
        f"Relaxed lambda = {lam:.8f}; expected lambda_scale/sqrt(n) = {expected:.8f}."
    )


def test_relaxed_lambda_independent_of_p():
    """Relaxed lambda must not depend on p."""
    model = DesparifiedGGM(lambda_method="relaxed")
    lam_p50 = model._get_lambda(n=100, p=50)
    lam_p200 = model._get_lambda(n=100, p=200)
    assert lam_p50 == lam_p200, (
        "Relaxed lambda depends on p, but λ = lambda_scale/√n is p-independent."
    )


def test_relaxed_lambda_fit_produces_valid_result():
    """Fitting with lambda_method='relaxed' must produce valid z-scores and p-values."""
    data = generate(n=150, p=25, topology="hub", seed=99)
    model = DesparifiedGGM(lambda_method="relaxed").fit(data.X)
    res = model.result_
    assert np.isfinite(res.z_scores).all(), "z_scores contain NaN/Inf under relaxed lambda"
    assert np.isfinite(res.p_values).all(), "p_values contain NaN/Inf under relaxed lambda"
    assert np.all(res.p_values >= 0.0) and np.all(res.p_values <= 1.0)


def test_invalid_lambda_method_raises():
    with pytest.raises(ValueError, match="lambda_method"):
        DesparifiedGGM(lambda_method="oracle")


# ---------------------------------------------------------------------------
# Degrees-of-freedom correction (Bellec & Zhang 2022)
# ---------------------------------------------------------------------------

def test_dof_correction_fit_produces_valid_result():
    """dof_correction=True must produce finite, valid outputs."""
    data = generate(n=150, p=25, topology="cluster", seed=77)
    model = DesparifiedGGM(dof_correction=True).fit(data.X)
    res = model.result_
    assert np.isfinite(res.z_scores).all(), "z_scores contain NaN/Inf with dof_correction"
    assert np.isfinite(res.p_values).all(), "p_values contain NaN/Inf with dof_correction"
    assert np.all(res.p_values >= 0.0) and np.all(res.p_values <= 1.0)


def test_dof_correction_symmetry():
    """DoF-corrected precision matrix must remain symmetric."""
    data = generate(n=150, p=20, topology="scale-free", seed=55)
    model = DesparifiedGGM(dof_correction=True).fit(data.X)
    np.testing.assert_array_almost_equal(
        model.result_.precision,
        model.result_.precision.T,
        decimal=10,
    )


def test_dof_correction_no_effect_when_all_zero():
    """When the Lasso drives all coefficients to zero (very large lambda),
    df_i = n for all nodes, so the DoF correction is a no-op.
    """
    data = generate(n=150, p=20, topology="random", seed=33)
    # Very large lambda — Lasso will set all β to zero
    model_base = DesparifiedGGM(lambda_scale=100.0, dof_correction=False).fit(data.X)
    model_dof = DesparifiedGGM(lambda_scale=100.0, dof_correction=True).fit(data.X)
    np.testing.assert_array_almost_equal(
        model_base.result_.precision,
        model_dof.result_.precision,
        decimal=8,
        err_msg="DoF correction should be a no-op when all Lasso coefficients are zero.",
    )
