"""Unit tests for nodis.report.generator."""
import numpy as np
import pytest

from nodis.report.generator import generate_report


@pytest.fixture(scope="module")
def minimal_result():
    from nodis.estimators.desparsified import DesparifiedGGM
    rng = np.random.default_rng(0)
    X = rng.standard_normal((80, 20))
    est = DesparifiedGGM()
    est.fit(X)
    est.get_adjacency(alpha=0.5)  # permissive to get some edges
    return est.result_


@pytest.fixture(scope="module")
def gene_names():
    return [f"G{i:03d}" for i in range(20)]


@pytest.fixture(scope="module")
def null_result():
    """Result with all-zero adjacency."""
    from nodis.estimators.desparsified import DesparifiedGGM
    rng = np.random.default_rng(1)
    X = rng.standard_normal((80, 20))
    est = DesparifiedGGM()
    est.fit(X)
    est.get_adjacency(alpha=0.0)  # strict → 0 edges
    return est.result_


def test_report_generates_html(tmp_path, minimal_result, gene_names):
    out = tmp_path / "report.html"
    generate_report(minimal_result, gene_names, out_path=out)
    assert out.exists()
    content = out.read_text()
    assert "<html" in content
    assert "NODIS" in content


def test_report_contains_pvalue_figure(tmp_path, minimal_result, gene_names):
    out = tmp_path / "report.html"
    generate_report(minimal_result, gene_names, out_path=out)
    content = out.read_text()
    assert "data:image/png;base64" in content


def test_report_handles_null_graph(tmp_path, null_result, gene_names):
    """Report generation should not crash when adj is all zeros."""
    out = tmp_path / "report.html"
    generate_report(null_result, gene_names, out_path=out)
    assert out.exists()


def test_report_with_advisor_warnings(tmp_path, minimal_result, gene_names):
    from nodis.advisor import recommend
    advisor = recommend(n=80, p=20, data_type="scrna_seq")
    out = tmp_path / "report.html"
    generate_report(minimal_result, gene_names, out_path=out, advisor_result=advisor)
    content = out.read_text()
    assert "pseudobulk" in content


def test_report_is_self_contained(tmp_path, minimal_result, gene_names):
    """File must not reference any external URLs for CSS/JS."""
    out = tmp_path / "report.html"
    generate_report(minimal_result, gene_names, out_path=out)
    content = out.read_text()
    assert "http://" not in content
    assert "https://" not in content


def test_report_returns_path(tmp_path, minimal_result, gene_names):
    import pathlib
    out = tmp_path / "report.html"
    result_path = generate_report(minimal_result, gene_names, out_path=out)
    assert isinstance(result_path, pathlib.Path)
    assert result_path == out


def test_report_size_reasonable(tmp_path, minimal_result, gene_names):
    """Report should be < 5 MB for small p."""
    out = tmp_path / "report.html"
    generate_report(minimal_result, gene_names, out_path=out)
    size_mb = out.stat().st_size / (1024 * 1024)
    assert size_mb < 5.0


def test_report_custom_title(tmp_path, minimal_result, gene_names):
    out = tmp_path / "report.html"
    generate_report(minimal_result, gene_names, out_path=out, title="My Custom Run")
    content = out.read_text()
    assert "My Custom Run" in content
