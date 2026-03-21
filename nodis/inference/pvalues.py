"""
Conversion of asymptotic z-scores to two-sided p-values.
"""

import numpy as np
from scipy.stats import norm


def z_to_pvalues(z_scores: np.ndarray) -> np.ndarray:
    """
    Convert a matrix of asymptotic z-scores to two-sided p-values.

    p_ij = 2 · (1 − Φ(|Z_ij|))

    Parameters
    ----------
    z_scores : ndarray of any shape

    Returns
    -------
    p_values : ndarray, same shape as z_scores
    """
    return 2.0 * norm.sf(np.abs(z_scores))
