"""Unit tests for nodis.priors.string_prior — all HTTP calls mocked."""
import json
import unittest.mock as mock

import numpy as np
import pytest

from nodis.priors.string_prior import prior_from_string

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


@mock.patch("nodis.priors.string_prior.urllib.request.urlopen",
            side_effect=_mock_urlopen)
def test_basic_prior(mock_url):
    genes = ["TP53", "MDM2", "BRCA1", "EGFR"]
    prior = prior_from_string(genes, organism=9606)
    assert prior.shape == (4, 4)
    assert prior[0, 1] == pytest.approx(0.95)
    assert prior[1, 0] == pytest.approx(0.95)
    assert np.diag(prior).sum() == 0.0


@mock.patch("nodis.priors.string_prior.urllib.request.urlopen",
            side_effect=_mock_urlopen)
def test_symmetry(mock_url):
    genes = ["TP53", "MDM2", "BRCA1", "EGFR"]
    prior = prior_from_string(genes)
    assert np.allclose(prior, prior.T)


@mock.patch("nodis.priors.string_prior.urllib.request.urlopen",
            side_effect=_mock_urlopen)
def test_unknown_genes_get_zero(mock_url):
    genes = ["TP53", "MDM2", "FAKE_GENE"]
    prior = prior_from_string(genes)
    assert prior[2, :].sum() == 0.0
    assert prior[:, 2].sum() == 0.0


def test_invalid_score_type_raises():
    with pytest.raises(ValueError, match="score_type must be"):
        prior_from_string(["TP53", "MDM2"], score_type="invalid")


def test_duplicate_genes_raises():
    with pytest.raises(ValueError, match="duplicate"):
        prior_from_string(["TP53", "TP53", "MDM2"])


@mock.patch("nodis.priors.string_prior.urllib.request.urlopen",
            side_effect=_mock_urlopen)
def test_brca1_tp53_score(mock_url):
    genes = ["TP53", "MDM2", "BRCA1"]
    prior = prior_from_string(genes)
    # TP53 is idx 0, BRCA1 is idx 2
    assert prior[0, 2] == pytest.approx(0.70)
    assert prior[2, 0] == pytest.approx(0.70)


@mock.patch("nodis.priors.string_prior.urllib.request.urlopen",
            side_effect=_mock_urlopen)
def test_returns_float64(mock_url):
    genes = ["TP53", "MDM2"]
    prior = prior_from_string(genes)
    assert prior.dtype == np.float64


@mock.patch("nodis.priors.string_prior.urllib.request.urlopen",
            side_effect=_mock_urlopen)
def test_diagonal_zero(mock_url):
    genes = ["TP53", "MDM2", "BRCA1"]
    prior = prior_from_string(genes)
    np.testing.assert_array_equal(np.diag(prior), np.zeros(3))
