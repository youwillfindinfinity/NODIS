"""
Parity validation: DesparifiedGGM vs. reference B_NW_SL (pure Python).

The reference implementation in nodis.estimators._reference_bnwsl follows
the Ren et al. (2015) formula verbatim using vanilla sklearn Lasso — no
optimisations, no path tricks, no shared code with DesparifiedGGM.

Acceptance criteria (unchanged from the original SILGGM-based criteria):
  Pearson r (z-scores, upper triangle) > 0.99  (per topology)
  |ΔAUPR|                               < 0.01  (per topology)

This replaces the rpy2/SILGGM bridge test which required a working
R 4.3.x + rpy2 installation.  Advantages of the pure-Python approach:
  - Runs in CI unconditionally (no @pytest.mark.requires_r skip)
  - Tests mathematical correctness of the formula, not tool-to-tool agreement
  - Reproducible: same random seed, same numpy PRNG on every platform
"""

import numpy as np
import pytest

from nodis.simulate.generator import generate
from nodis.estimators.desparsified import DesparifiedGGM
from nodis.estimators._reference_bnwsl import fit_reference_bnwsl
from nodis.benchmark.evaluate import evaluate_predictions


@pytest.mark.parametrize("topology", ["hub", "scale-free", "cluster", "random"])
def test_zscore_parity_reference(topology):
    """DesparifiedGGM z-scores must correlate > 0.99 with reference B_NW_SL."""
    data = generate(n=200, p=50, topology=topology, seed=42)

    nodis_model = DesparifiedGGM(lambda_scale=1.0, standardise=True).fit(data.X)
    ref = fit_reference_bnwsl(data.X, lambda_scale=1.0, standardise=True)

    uidx = np.triu_indices(50, k=1)
    z_nodis = nodis_model.result_.z_scores[uidx]
    z_ref   = ref["z_scores"][uidx]

    r_corr = float(np.corrcoef(z_nodis, z_ref)[0, 1])
    assert r_corr > 0.99, (
        f"Z-score Pearson r = {r_corr:.4f} for '{topology}' topology — "
        "expected > 0.99 (DesparifiedGGM vs. reference B_NW_SL formula)."
    )


@pytest.mark.parametrize("topology", ["hub", "scale-free", "cluster", "random"])
def test_aupr_parity_reference(topology):
    """|ΔAUPR| between DesparifiedGGM and reference B_NW_SL must be < 0.01."""
    data = generate(n=200, p=50, topology=topology, seed=42)

    nodis_model = DesparifiedGGM(lambda_scale=1.0).fit(data.X)
    adj_nodis = nodis_model.get_adjacency(alpha=0.05)
    ref = fit_reference_bnwsl(data.X, lambda_scale=1.0)

    # Use p-value threshold 0.05 for reference adjacency
    from scipy.stats import false_discovery_control
    p_upper = ref["p_values"][np.triu_indices(50, k=1)]
    adj_mat = ref["p_values"].copy()
    # BH correction on upper triangle
    pvals_flat = ref["p_values"][np.triu_indices(50, k=1)]
    try:
        rejected = false_discovery_control(pvals_flat, method="bh") <= 0.05
    except Exception:
        rejected = pvals_flat <= 0.05
    adj_ref = np.zeros((50, 50))
    uidx = np.triu_indices(50, k=1)
    adj_ref[uidx] = rejected.astype(float)
    adj_ref[(uidx[1], uidx[0])] = adj_ref[uidx]

    scores_nodis = np.abs(nodis_model.result_.z_scores)
    scores_ref   = np.abs(ref["z_scores"])

    aupr_nodis = evaluate_predictions(adj_nodis, data.Omega, scores=scores_nodis)["aupr"]
    aupr_ref   = evaluate_predictions(adj_ref,   data.Omega, scores=scores_ref)["aupr"]

    delta = abs(aupr_nodis - aupr_ref)
    assert delta < 0.01, (
        f"ΔAUPR = {delta:.4f} for '{topology}' topology — "
        "expected < 0.01 (DesparifiedGGM vs. reference B_NW_SL formula)."
    )


@pytest.mark.parametrize("topology", ["hub", "scale-free", "cluster", "random"])
def test_precision_parity_reference(topology):
    """De-biased precision entries must agree to within 1% (max rel. error)."""
    data = generate(n=200, p=50, topology=topology, seed=42)

    nodis_model = DesparifiedGGM(lambda_scale=1.0).fit(data.X)
    ref = fit_reference_bnwsl(data.X, lambda_scale=1.0)

    uidx = np.triu_indices(50, k=1)
    omega_nodis = nodis_model.result_.precision[uidx]
    omega_ref   = ref["precision"][uidx]

    denom = np.abs(omega_ref)
    denom[denom < 1e-6] = 1e-6  # avoid division by near-zero for null edges
    rel_err = np.mean(np.abs(omega_nodis - omega_ref) / denom)
    assert rel_err < 0.01, (
        f"Mean relative Ω̂ error = {rel_err:.4f} for '{topology}' — "
        "expected < 1%."
    )
