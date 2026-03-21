"""Regression tests for PIGLassoEstimator convergence guard."""
import warnings
import numpy as np
import pytest


def _make_data(n=80, p=10, seed=0):
    rng = np.random.default_rng(seed)
    Theta = np.eye(p) * 3.0
    Sigma = np.linalg.inv(Theta)
    return rng.multivariate_normal(np.zeros(p), Sigma, size=n)


def test_piglasso_fit_completes():
    """Smoke test: fit produces a stability matrix of the right shape."""
    pytest.importorskip("gglasso", reason="gglasso not installed")
    from nodis.estimators.piglasso import PIGLassoEstimator

    X = _make_data()
    est = PIGLassoEstimator(Q=5, n_lambda=5, pi_thr=0.6).fit(X)
    assert hasattr(est, "precision_")
    assert est.precision_.shape == (10, 10)
    assert np.all((est.precision_ >= 0.0) & (est.precision_ <= 1.0))


def test_piglasso_convergence_guard_emits_warning_on_bad_solve(monkeypatch):
    """Convergence guard must warn and exclude non-converged ADMM runs."""
    import nodis.estimators.piglasso as pig_module
    from nodis.estimators.piglasso import PIGLassoEstimator

    def _always_unconverged(S, lam, Omega_0, **kwargs):
        # Return same dict structure as real ADMM_SGL but with non-optimal status.
        p = S.shape[0]
        result_vars = {"Omega": np.eye(p), "Theta": np.eye(p), "X": np.zeros((p, p))}
        info = {"status": "max iterations reached"}
        return (result_vars, info)

    monkeypatch.setattr(pig_module, "ADMM_SGL", _always_unconverged)

    X = _make_data()
    with pytest.warns(RuntimeWarning, match="converge|status"):
        PIGLassoEstimator(Q=3, n_lambda=3, pi_thr=0.6).fit(X)


def test_piglasso_stability_excludes_unconverged_runs(monkeypatch):
    """Non-converged runs must not increment success counts."""
    import nodis.estimators.piglasso as pig_module
    from nodis.estimators.piglasso import PIGLassoEstimator

    call_counts = {"total": 0, "converged": 0}

    def _sometimes_converged(S, lam, Omega_0, **kwargs):
        call_counts["total"] += 1
        p = S.shape[0]
        result_vars = {"Omega": np.eye(p), "Theta": np.eye(p), "X": np.zeros((p, p))}
        # First call converges, rest do not
        if call_counts["total"] == 1:
            call_counts["converged"] += 1
            return (result_vars, {"status": "optimal"})
        return (result_vars, {"status": "max iterations reached"})

    monkeypatch.setattr(pig_module, "ADMM_SGL", _sometimes_converged)

    X = _make_data()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        est = PIGLassoEstimator(Q=3, n_lambda=3, pi_thr=0.6).fit(X)

    # The single converged run returned Theta=eye(p); after zeroing the diagonal,
    # edge_mask is all zeros, so stability_ must be all-zero.
    assert np.allclose(est.stability_, 0.0), (
        "Expected all-zero stability when every ADMM solve either fails convergence "
        "or produces a diagonal-only Theta (no off-diagonal edges selected)."
    )
    assert call_counts["total"] > 1
