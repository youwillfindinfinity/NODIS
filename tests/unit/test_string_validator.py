"""Unit tests for nodis.validate.string_validator — STRING HTTP calls mocked."""
import json
import unittest.mock as mock

import numpy as np
import pytest

from nodis.validate.string_validator import validate_against_string, ValidationResult

MOCK_STRING_RESPONSE = [
    {"preferredName_A": "TP53", "preferredName_B": "MDM2",
     "score_combined": 950, "score": 950},
    {"preferredName_A": "TP53", "preferredName_B": "BRCA1",
     "score_combined": 700, "score": 700},
]


def _mock_urlopen(url, timeout=None):
    class FakeResp:
        def read(self):
            return json.dumps(MOCK_STRING_RESPONSE).encode()
        def __enter__(self):
            return self
        def __exit__(self, *a):
            pass
    return FakeResp()


_GENES = ["TP53", "MDM2", "BRCA1", "EGFR"]


def _make_adj(edges, n=4):
    adj = np.zeros((n, n), dtype=int)
    for i, j in edges:
        adj[i, j] = adj[j, i] = 1
    return adj


@mock.patch("nodis.priors.string_prior.urllib.request.urlopen",
            side_effect=_mock_urlopen)
def test_basic_result_type(mock_url):
    adj = _make_adj([(0, 1)])  # TP53-MDM2
    result = validate_against_string(adj, _GENES)
    assert isinstance(result, ValidationResult)


@mock.patch("nodis.priors.string_prior.urllib.request.urlopen",
            side_effect=_mock_urlopen)
def test_perfect_overlap(mock_url):
    # Both STRING edges present in adj: TP53-MDM2 (0,1) and TP53-BRCA1 (0,2)
    adj = _make_adj([(0, 1), (0, 2)])
    result = validate_against_string(adj, _GENES)
    assert result.n_overlap == 2
    assert result.precision == pytest.approx(1.0)
    assert result.recall == pytest.approx(1.0)
    assert result.f1 == pytest.approx(1.0)


@mock.patch("nodis.priors.string_prior.urllib.request.urlopen",
            side_effect=_mock_urlopen)
def test_no_overlap(mock_url):
    # Only EGFR-BRCA1 inferred (not in STRING mock)
    adj = _make_adj([(2, 3)])
    result = validate_against_string(adj, _GENES)
    assert result.n_overlap == 0
    assert result.precision == pytest.approx(0.0)
    assert result.f1 == pytest.approx(0.0)


@mock.patch("nodis.priors.string_prior.urllib.request.urlopen",
            side_effect=_mock_urlopen)
def test_partial_overlap_precision(mock_url):
    # Infer TP53-MDM2 (in STRING) and EGFR-BRCA1 (not in STRING): precision = 0.5
    adj = _make_adj([(0, 1), (2, 3)])
    result = validate_against_string(adj, _GENES)
    assert result.precision == pytest.approx(0.5)


@mock.patch("nodis.priors.string_prior.urllib.request.urlopen",
            side_effect=_mock_urlopen)
def test_empty_network(mock_url):
    adj = np.zeros((4, 4), dtype=int)
    result = validate_against_string(adj, _GENES)
    assert result.n_inferred_edges == 0
    assert result.precision == pytest.approx(0.0)
    assert result.recall == pytest.approx(0.0)
    assert result.jaccard == pytest.approx(0.0)


@mock.patch("nodis.priors.string_prior.urllib.request.urlopen",
            side_effect=_mock_urlopen)
def test_string_score_dist_shape(mock_url):
    adj = _make_adj([(0, 1), (2, 3)])
    result = validate_against_string(adj, _GENES)
    # 2 inferred edges → score_dist has 2 entries
    assert result.string_score_dist.shape == (2,)


@mock.patch("nodis.priors.string_prior.urllib.request.urlopen",
            side_effect=_mock_urlopen)
def test_jaccard_formula(mock_url):
    # TP53-MDM2 only; STRING has 2 edges: tp=1, fp=0, fn=1 → jaccard=1/(1+0+1)=0.5
    adj = _make_adj([(0, 1)])
    result = validate_against_string(adj, _GENES)
    assert result.jaccard == pytest.approx(0.5)


@mock.patch("nodis.priors.string_prior.urllib.request.urlopen",
            side_effect=_mock_urlopen)
def test_validate_against_string_method_on_result(mock_url):
    """GGMInferenceResult.validate_against_string() delegates correctly."""
    from nodis.estimators.desparsified import DesparifiedGGM
    rng = np.random.default_rng(0)
    X = rng.standard_normal((100, 4))
    est = DesparifiedGGM()
    est.fit(X)
    adj = est.get_adjacency(alpha=0.5)  # permissive alpha to get some edges
    result = est.result_.validate_against_string(_GENES)
    assert isinstance(result, ValidationResult)
