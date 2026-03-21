"""
FDR control for GGM edge p-values.

Wraps ``scipy.stats.false_discovery_control`` (requires scipy >= 1.11).

References
----------
Benjamini Y, Hochberg Y (1995). Controlling the false discovery rate:
    a practical and powerful approach to multiple testing.
    J R Stat Soc B 57(1): 289–300.

Benjamini Y, Yekutieli D (2001). The control of the false discovery rate
    in multiple testing under dependency.
    Ann Stat 29(4): 1165–1188.
"""

import numpy as np
from scipy.stats import false_discovery_control

# scipy uses lowercase keys
_METHOD_MAP: dict[str, str] = {
    "BH": "bh",
    "bh": "bh",
    "BY": "by",
    "by": "by",
}


def fdr_control(
    p_values: np.ndarray,
    alpha: float = 0.05,
    method: str = "BH",
) -> np.ndarray:
    """
    Apply FDR control to the upper-triangle of a symmetric p-value matrix
    and return a binary symmetric adjacency matrix.

    Parameters
    ----------
    p_values : (p, p) ndarray — symmetric matrix of two-sided p-values
    alpha    : float, default 0.05 — target FDR level
    method   : 'BH' or 'BY'

    Returns
    -------
    adj : (p, p) integer ndarray — symmetric binary adjacency; diagonal = 0

    Notes
    -----
    The BH procedure (method='BH') assumes independence or positive regression
    dependence on a subset (PRDS). Precision matrix entries exhibit moderate
    positive dependence via the concentration graph structure, but PRDS is not
    formally guaranteed. For conservative FDR control under arbitrary dependency,
    use method='BY' (Benjamini-Yekutieli 2001).
    """
    if p_values.ndim != 2 or p_values.shape[0] != p_values.shape[1]:
        raise ValueError("p_values must be a square 2-D array.")

    scipy_method = _METHOD_MAP.get(method)
    if scipy_method is None:
        raise ValueError(f"method must be 'BH' or 'BY'; got '{method}'.")

    p = p_values.shape[0]
    uidx = np.triu_indices(p, k=1)
    pvals_u = p_values[uidx]

    # false_discovery_control returns adjusted p-values (q-values), not booleans
    adjusted = false_discovery_control(pvals_u, method=scipy_method)
    reject = adjusted <= alpha

    adj = np.zeros((p, p), dtype=int)
    adj[uidx] = reject.astype(int)
    adj = adj + adj.T          # symmetrise
    np.fill_diagonal(adj, 0)
    return adj
