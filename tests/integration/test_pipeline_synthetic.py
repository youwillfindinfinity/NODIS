"""
Integration test: full pipeline on synthetic data.

Verifies that the end-to-end pipeline (generate → NPN → fit → FDR → evaluate)
runs without error and that AUPR > 0.5 on structured topologies with adequate n.

These tests are slower than unit tests (~10–30 s each); run with -m slow to include.
"""

import numpy as np
import pytest

from nodis.simulate.generator import generate
from nodis.preprocess.npn import npn_shrinkage
from nodis.estimators.desparsified import DesparifiedGGM
from nodis.benchmark.evaluate import evaluate_predictions


@pytest.mark.slow
@pytest.mark.parametrize("topology", ["hub", "cluster"])
def test_pipeline_structured_topology(topology):
    """NODIS should achieve AUPR > 0.5 on structured graphs with n=300, p=30."""
    data = generate(n=300, p=30, topology=topology, seed=42)
    X_npn = npn_shrinkage(data.X)
    model = DesparifiedGGM().fit(X_npn)
    adj = model.get_adjacency(alpha=0.05)
    scores = np.abs(model.result_.z_scores)
    metrics = evaluate_predictions(adj, data.Omega, scores=scores)

    assert metrics["aupr"] > 0.5, (
        f"AUPR = {metrics['aupr']:.3f} on '{topology}' (n=300, p=30): "
        "expected > 0.5 for a structured topology."
    )


@pytest.mark.slow
def test_pipeline_null_low_fdr():
    """On a null (empty) graph, the FDR-controlled adjacency should be sparse."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((300, 30))    # no graph structure
    true_adj = np.zeros((30, 30), dtype=int)

    model = DesparifiedGGM().fit(X)
    adj = model.get_adjacency(alpha=0.05)
    metrics = evaluate_predictions(adj, true_adj)

    # All detected edges are false positives; FDP ≤ 10% (generous bound)
    n_detected = adj.sum() // 2
    n_possible = 30 * 29 // 2
    assert n_detected / n_possible < 0.10, (
        f"Too many false edges on null graph: {n_detected}/{n_possible}."
    )


def test_pipeline_completes_no_error():
    """Minimal smoke test: pipeline from generate to adjacency without errors."""
    data = generate(n=50, p=10, topology="random", seed=7)
    model = DesparifiedGGM().fit(data.X)
    adj = model.get_adjacency(alpha=0.05)
    assert adj.shape == (10, 10)


@pytest.mark.parametrize("alpha", [0.01, 0.05, 0.10, 0.20])
def test_all_null_fpr_calibration(alpha):
    """Empirical FPR on a fully null graph must be ≤ 2·alpha (generous tolerance).

    Generates data from N(0, I) so all off-diagonal Theta_ij = 0.  Under the
    asymptotic null, BH at level alpha should produce FPR ≈ alpha on average.
    We use a fast n=200, p=50 setting and allow 2× slack for finite-sample
    variability at a single replicate.

    This test validates the Type I error calibration claim (Figure 2 panel).
    """
    import warnings

    rng = np.random.default_rng(seed=2024)
    n, p = 200, 50
    X = rng.standard_normal((n, p))
    n_pairs = p * (p - 1) // 2

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = DesparifiedGGM().fit(X)

    adj = model.get_adjacency(alpha=alpha)
    n_false_positives = adj.sum() // 2
    empirical_fpr = n_false_positives / n_pairs

    assert empirical_fpr <= 2.0 * alpha, (
        f"Empirical FPR = {empirical_fpr:.4f} > 2·alpha = {2 * alpha:.4f} "
        f"at alpha={alpha} on null graph (n={n}, p={p}). "
        "Type I error is not calibrated."
    )
