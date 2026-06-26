"""
Unit tests for nodis.estimators.group_glasso.

Covers MultiConditionGLasso and fit_multi_condition.
gglasso is required; tests are skipped if not installed.
"""

import numpy as np
import pytest

gglasso = pytest.importorskip("gglasso", reason="gglasso not installed")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def X_dict(rng):
    """Three conditions, n=80 samples, p=15 genes."""
    K, n, p = 3, 80, 15
    return {f"cond{k}": rng.standard_normal((n, p)) for k in range(K)}


@pytest.fixture
def X_dict_two(rng):
    n, p = 60, 12
    return {"treated": rng.standard_normal((n, p)),
            "control": rng.standard_normal((n, p))}


# ---------------------------------------------------------------------------
# MultiConditionGLasso — basic API
# ---------------------------------------------------------------------------

def test_fit_ggl_fixed_lambda(X_dict):
    from nodis.estimators.group_glasso import MultiConditionGLasso
    est = MultiConditionGLasso(reg="GGL", lambda1=0.15, lambda2=0.05)
    est.fit(X_dict)
    r = est.result_
    assert r.reg == "GGL"
    assert set(r.condition_names) == {"cond0", "cond1", "cond2"}
    assert r.lambda1_ == pytest.approx(0.15)
    assert r.lambda2_ == pytest.approx(0.05)
    assert not r.ebic_selected


def test_fit_fgl_fixed_lambda(X_dict):
    from nodis.estimators.group_glasso import MultiConditionGLasso
    est = MultiConditionGLasso(reg="FGL", lambda1=0.15, lambda2=0.05)
    est.fit(X_dict)
    assert est.result_.reg == "FGL"
    assert est.result_.n_conditions == 3


def test_precision_shape(X_dict):
    from nodis.estimators.group_glasso import MultiConditionGLasso
    est = MultiConditionGLasso(lambda1=0.15, lambda2=0.05)
    est.fit(X_dict)
    for c in est.result_.condition_names:
        prec = est.result_.precision_[c]
        assert prec.shape == (15, 15)
        # Precision should be symmetric
        np.testing.assert_allclose(prec, prec.T, atol=1e-8)


def test_adjacency_no_self_loops(X_dict):
    from nodis.estimators.group_glasso import MultiConditionGLasso
    est = MultiConditionGLasso(lambda1=0.15, lambda2=0.05)
    est.fit(X_dict)
    for c, adj in est.result_.adjacency_.items():
        assert adj.shape == (15, 15)
        assert np.all(np.diag(adj) == 0), f"Self-loop in condition {c}"
        # Symmetric
        np.testing.assert_array_equal(adj, adj.T)


def test_shared_adjacency_subset(X_dict):
    from nodis.estimators.group_glasso import MultiConditionGLasso
    est = MultiConditionGLasso(lambda1=0.15, lambda2=0.05)
    est.fit(X_dict)
    r = est.result_
    shared = r.shared_adjacency
    for adj in r.adjacency_.values():
        # Shared must be a subset of every condition
        assert np.all(shared <= adj)
    assert np.all(np.diag(shared) == 0)


def test_unique_adjacency_disjoint(X_dict):
    from nodis.estimators.group_glasso import MultiConditionGLasso
    est = MultiConditionGLasso(lambda1=0.15, lambda2=0.05)
    est.fit(X_dict)
    r = est.result_
    # Unique edges of condition A must not appear in any other condition
    for c, uniq in r.unique_adjacency.items():
        for c2, adj2 in r.adjacency_.items():
            if c2 == c:
                continue
            # uniq & adj2 should be zero everywhere
            assert np.all((uniq & adj2) == 0), (
                f"unique[{c}] overlaps with adjacency[{c2}]"
            )


def test_get_adjacency_single(X_dict):
    from nodis.estimators.group_glasso import MultiConditionGLasso
    est = MultiConditionGLasso(lambda1=0.15, lambda2=0.05).fit(X_dict)
    adj = est.get_adjacency("cond0")
    assert adj.shape == (15, 15)


def test_get_adjacency_all(X_dict):
    from nodis.estimators.group_glasso import MultiConditionGLasso
    est = MultiConditionGLasso(lambda1=0.15, lambda2=0.05).fit(X_dict)
    d = est.get_adjacency()
    assert isinstance(d, dict)
    assert set(d.keys()) == {"cond0", "cond1", "cond2"}


def test_get_adjacency_missing_condition(X_dict):
    from nodis.estimators.group_glasso import MultiConditionGLasso
    est = MultiConditionGLasso(lambda1=0.15, lambda2=0.05).fit(X_dict)
    with pytest.raises(KeyError, match="not found"):
        est.get_adjacency("nonexistent")


def test_get_shared_adjacency(X_dict):
    from nodis.estimators.group_glasso import MultiConditionGLasso
    est = MultiConditionGLasso(lambda1=0.15, lambda2=0.05).fit(X_dict)
    shared = est.get_shared_adjacency()
    assert shared.shape == (15, 15)


def test_not_fitted_raises(X_dict):
    from nodis.estimators.group_glasso import MultiConditionGLasso
    est = MultiConditionGLasso(lambda1=0.1, lambda2=0.05)
    with pytest.raises(RuntimeError, match="fit()"):
        est.get_adjacency()
    with pytest.raises(RuntimeError, match="fit()"):
        est.get_shared_adjacency()


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def test_single_condition_raises(rng):
    from nodis.estimators.group_glasso import MultiConditionGLasso
    est = MultiConditionGLasso(lambda1=0.1, lambda2=0.05)
    with pytest.raises(ValueError, match="at least 2 conditions"):
        est.fit({"only": rng.standard_normal((50, 10))})


def test_mismatched_p_raises(rng):
    from nodis.estimators.group_glasso import MultiConditionGLasso
    est = MultiConditionGLasso(lambda1=0.1, lambda2=0.05)
    with pytest.raises(ValueError, match="same number of genes"):
        est.fit({
            "a": rng.standard_normal((50, 10)),
            "b": rng.standard_normal((50, 12)),
        })


def test_invalid_reg_raises():
    from nodis.estimators.group_glasso import MultiConditionGLasso
    with pytest.raises(ValueError, match="reg must be"):
        MultiConditionGLasso(reg="INVALID")


def test_3d_input_raises(rng):
    from nodis.estimators.group_glasso import MultiConditionGLasso
    est = MultiConditionGLasso(lambda1=0.1, lambda2=0.05)
    with pytest.raises(ValueError, match="2-D"):
        est.fit({"a": rng.standard_normal((5, 10, 3)),
                 "b": rng.standard_normal((5, 10))})


# ---------------------------------------------------------------------------
# NPN preprocessing
# ---------------------------------------------------------------------------

def test_npn_flag(X_dict):
    from nodis.estimators.group_glasso import MultiConditionGLasso
    est = MultiConditionGLasso(lambda1=0.15, lambda2=0.05, npn=True)
    est.fit(X_dict)
    # Just verify it runs and returns valid shapes
    assert est.result_.precision_["cond0"].shape == (15, 15)


# ---------------------------------------------------------------------------
# eBIC model selection (small grid to keep test fast)
# ---------------------------------------------------------------------------

def test_ebic_model_selection(X_dict_two):
    from nodis.estimators.group_glasso import MultiConditionGLasso
    est = MultiConditionGLasso(
        reg="GGL",
        lambda1=None, lambda2=None,
        lambda1_range=np.array([0.2, 0.1]),
        lambda2_range=np.array([0.05, 0.02]),
        ebic_gamma=0.1,
    )
    est.fit(X_dict_two)
    r = est.result_
    assert r.ebic_selected is True
    assert r.lambda1_ > 0
    assert r.lambda2_ > 0
    assert r.precision_["treated"].shape == (12, 12)


# ---------------------------------------------------------------------------
# summary()
# ---------------------------------------------------------------------------

def test_summary_keys(X_dict):
    from nodis.estimators.group_glasso import MultiConditionGLasso
    est = MultiConditionGLasso(lambda1=0.15, lambda2=0.05).fit(X_dict)
    s = est.result_.summary()
    assert set(s.keys()) >= {
        "reg", "lambda1", "lambda2", "ebic_selected",
        "n_shared_edges", "edges_per_condition"
    }
    assert set(s["edges_per_condition"].keys()) == {"cond0", "cond1", "cond2"}


# ---------------------------------------------------------------------------
# to_anndata
# ---------------------------------------------------------------------------

def test_to_anndata(X_dict):
    anndata = pytest.importorskip("anndata")
    from nodis.estimators.group_glasso import MultiConditionGLasso

    p = 15
    adata = anndata.AnnData(np.zeros((10, p)))
    adata.var_names = [f"G{i}" for i in range(p)]

    est = MultiConditionGLasso(lambda1=0.15, lambda2=0.05).fit(X_dict)
    est.to_anndata(adata)

    assert "nodis_mgl_shared" in adata.varp
    for c in est.result_.condition_names:
        safe = c.replace(" ", "_")
        assert f"nodis_mgl_prec_{safe}" in adata.varp
        assert f"nodis_mgl_adj_{safe}" in adata.varp
        assert f"nodis_mgl_unique_{safe}" in adata.varp
    assert "nodis_mgl" in adata.uns
    assert "conditions" in adata.uns["nodis_mgl"]


def test_to_anndata_shape_mismatch(X_dict):
    anndata = pytest.importorskip("anndata")
    from nodis.estimators.group_glasso import MultiConditionGLasso

    adata = anndata.AnnData(np.zeros((5, 20)))  # wrong p
    est = MultiConditionGLasso(lambda1=0.15, lambda2=0.05).fit(X_dict)
    with pytest.raises(ValueError, match="does not match"):
        est.to_anndata(adata)


# ---------------------------------------------------------------------------
# fit_multi_condition (AnnData-native convenience function)
# ---------------------------------------------------------------------------

def test_fit_multi_condition_from_anndata(rng):
    anndata = pytest.importorskip("anndata")
    import pandas as pd
    from nodis.estimators.group_glasso import fit_multi_condition

    n_per_cond, p = 40, 10
    X = rng.standard_normal((n_per_cond * 3, p))
    obs = pd.DataFrame({
        "condition": ["A"] * n_per_cond + ["B"] * n_per_cond + ["C"] * n_per_cond
    })
    adata = anndata.AnnData(X, obs=obs)
    adata.var_names = [f"G{i}" for i in range(p)]

    est = fit_multi_condition(
        adata,
        condition_key="condition",
        reg="GGL",
        lambda1=0.15,
        lambda2=0.05,
    )

    assert set(est.result_.condition_names) == {"A", "B", "C"}
    assert "nodis_mgl_shared" in adata.varp
    assert "nodis_mgl" in adata.uns
    assert adata.uns["nodis_mgl"]["reg"] == "GGL"


def test_fit_multi_condition_missing_key(rng):
    anndata = pytest.importorskip("anndata")
    from nodis.estimators.group_glasso import fit_multi_condition

    adata = anndata.AnnData(rng.standard_normal((20, 8)))
    with pytest.raises(KeyError, match="not found in adata.obs"):
        fit_multi_condition(adata, condition_key="nonexistent")


def test_fit_multi_condition_single_group_raises(rng):
    anndata = pytest.importorskip("anndata")
    import pandas as pd
    from nodis.estimators.group_glasso import fit_multi_condition

    adata = anndata.AnnData(
        rng.standard_normal((30, 8)),
        obs=pd.DataFrame({"cond": ["A"] * 30}),
    )
    with pytest.raises(ValueError, match="at least 2"):
        fit_multi_condition(adata, condition_key="cond")


def test_fit_multi_condition_low_n_warns(rng):
    anndata = pytest.importorskip("anndata")
    import pandas as pd
    from nodis.estimators.group_glasso import fit_multi_condition

    p = 20
    n_per_cond = p - 5   # n=15 < p=20 → triggers warning
    total = n_per_cond * 2
    X = rng.standard_normal((total, p))
    obs = pd.DataFrame({"grp": ["A"] * n_per_cond + ["B"] * n_per_cond})
    adata = anndata.AnnData(X, obs=obs)

    with pytest.warns(UserWarning, match="n=.*< p="):
        fit_multi_condition(adata, condition_key="grp",
                            lambda1=0.3, lambda2=0.1)
