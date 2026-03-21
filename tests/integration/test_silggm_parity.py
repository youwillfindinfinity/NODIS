"""
Parity validation: NODIS vs SILGGM B_NW_SL (R).

Validates RQ1: does the native Python estimator produce statistically equivalent
results to SILGGM B_NW_SL on identical data?

Acceptance criteria
-------------------
  Pearson r (z-scores, upper triangle) > 0.99
  |ΔAUPR|                               < 0.01
  |ΔFDR rejection rate|                 < 0.005

These tests are skipped automatically when rpy2 is not installed (see conftest.py).

Reference
---------
Zhang R, Ren Z, Chen W (2018). SILGGM. PLoS Comput Biol 14(8): e1006369.
"""

import numpy as np
import pytest

from nodis.simulate.generator import generate
from nodis.estimators.desparsified import DesparifiedGGM
from nodis.benchmark.evaluate import evaluate_predictions


@pytest.mark.requires_r
@pytest.mark.parametrize("topology", ["hub", "scale-free", "cluster", "random"])
def test_zscore_parity(topology):
    """Z-scores from Python and R must correlate at r > 0.99."""
    from nodis.estimators._silggm_bridge import run_silggm_r

    data = generate(n=200, p=50, topology=topology, seed=42)
    py_model = DesparifiedGGM().fit(data.X)
    r_result = run_silggm_r(data.X, method="B_NW_SL")

    uidx = np.triu_indices(50, k=1)
    z_py = py_model.result_.z_scores[uidx]
    z_r = r_result["z_score"][uidx]

    r_corr = float(np.corrcoef(z_py, z_r)[0, 1])
    assert r_corr > 0.99, (
        f"Z-score Pearson r = {r_corr:.4f} for '{topology}' topology. "
        "Required: > 0.99 (RQ1)."
    )


@pytest.mark.requires_r
@pytest.mark.parametrize("topology", ["hub", "scale-free", "cluster", "random"])
def test_aupr_parity(topology):
    """AUPR from Python and R must differ by < 0.01."""
    from nodis.estimators._silggm_bridge import run_silggm_r

    data = generate(n=200, p=50, topology=topology, seed=42)
    py_model = DesparifiedGGM().fit(data.X)
    adj_py = py_model.get_adjacency(alpha=0.05)
    r_result = run_silggm_r(data.X, method="B_NW_SL", alpha=0.05)
    adj_r = r_result["adj"]

    scores_py = np.abs(py_model.result_.z_scores)
    scores_r = np.abs(r_result["z_score"])

    aupr_py = evaluate_predictions(adj_py, data.Omega, scores=scores_py)["aupr"]
    aupr_r = evaluate_predictions(adj_r, data.Omega, scores=scores_r)["aupr"]

    delta = abs(aupr_py - aupr_r)
    assert delta < 0.01, (
        f"ΔAUPR = {delta:.4f} for '{topology}' topology. "
        "Required: < 0.01 (RQ1)."
    )
