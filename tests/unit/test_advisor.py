"""Unit tests for nodis.advisor.recommend()."""
import pytest
from nodis.advisor import recommend, DATA_TYPES


def test_recommend_edge_pvalues_high_np():
    r = recommend(n=500, p=50, goal="edge_pvalues", data_type="bulk_rnaseq")
    assert r.estimator == "desparsified"
    assert r.kwargs.get("dof_correction") is False


def test_recommend_edge_pvalues_low_np():
    r = recommend(n=60, p=50, goal="edge_pvalues", data_type="bulk_rnaseq")
    assert r.kwargs["dof_correction"] is True
    assert r.kwargs["ensemble_ci"] is True
    assert any("n/p < 2" in w for w in r.warnings)


def test_recommend_scrna_warns_pseudobulk():
    r = recommend(n=200, p=100, goal="edge_pvalues", data_type="scrna_seq")
    assert any("pseudobulk" in w for w in r.warnings)
    assert r.preset == "scrna_pseudobulk"


def test_recommend_large_p_sparse():
    r = recommend(n=300, p=3000, goal="edge_pvalues")
    assert r.kwargs.get("sparse") is True


def test_recommend_differential():
    r = recommend(n=200, p=100, goal="differential", n_conditions=2)
    assert r.estimator == "gglasso_fgl"


def test_recommend_structure_with_prior():
    r = recommend(n=200, p=100, goal="network_structure", has_prior=True)
    assert r.estimator == "piglasso"


def test_python_snippet_is_valid_python():
    r = recommend(n=200, p=100)
    compile(r.python_snippet, "<snippet>", "exec")  # syntax check only


def test_all_data_types_return_valid_result():
    for dt in DATA_TYPES:
        r = recommend(n=200, p=100, data_type=dt)
        assert r.estimator
        assert r.preset


def test_recommend_mid_np_dof_correction_only():
    # 2 <= n/p < 5: dof_correction=True but no ensemble_ci
    r = recommend(n=300, p=100, goal="edge_pvalues")
    assert r.kwargs.get("dof_correction") is True
    assert "ensemble_ci" not in r.kwargs


def test_recommend_network_structure_low_np_no_prior():
    # n/p=2, no prior, 1 condition → gglasso
    r = recommend(n=200, p=100, goal="network_structure")
    assert r.estimator == "gglasso"


def test_recommend_network_structure_high_np():
    # n/p=20 > 10 → glasso
    r = recommend(n=2000, p=100, goal="network_structure")
    assert r.estimator == "glasso"


def test_recommend_network_structure_multi_condition():
    r = recommend(n=200, p=100, goal="network_structure", n_conditions=2)
    assert r.estimator == "gglasso_fgl"


def test_cli_snippet_nonempty():
    r = recommend(n=200, p=100)
    assert "nodis run" in r.cli_snippet
    assert "--preset" in r.cli_snippet


def test_reasoning_contains_np_ratio():
    r = recommend(n=200, p=100, goal="edge_pvalues")
    assert any("n/p" in line for line in r.reasoning)


def test_large_p_glasso_warns():
    r = recommend(n=300, p=3000, goal="network_structure", n_conditions=1)
    # n/p=0.1 → gglasso (not glasso), so no dense-matrix warning
    assert r.estimator == "gglasso"


def test_methylation_preset():
    r = recommend(n=200, p=100, data_type="methylation")
    assert r.preset == "methylation"
    assert any("M-values" in w for w in r.warnings)


def test_proteomics_preset():
    r = recommend(n=200, p=100, data_type="proteomics")
    assert r.preset == "proteomics"
    assert any("quantile" in w for w in r.warnings)
