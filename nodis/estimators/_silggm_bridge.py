"""
Optional rpy2 bridge to SILGGM (R).

Used exclusively for parity validation (RQ1): the Python de-sparsified
estimator is compared against SILGGM B_NW_SL on identical synthetic data.
This module is NEVER used in the core inference path.

Requires: R, rpy2 >= 3.5, SILGGM R package.

Install SILGGM in R:
    install.packages("SILGGM")

Reference
---------
Zhang R, Ren Z, Chen W (2018). SILGGM: An extensive R package for efficient
    statistical inference in large-scale gene networks.
    PLoS Comput Biol 14(8): e1006369. doi:10.1371/journal.pcbi.1006369
"""

import numpy as np


def run_silggm_r(
    X: np.ndarray,
    method: str = "B_NW_SL",
    alpha: float = 0.05,
) -> dict:
    """
    Run SILGGM in R via rpy2 on data matrix X.

    Parameters
    ----------
    X      : (n, p) ndarray — expression matrix (will be passed to R as-is)
    method : SILGGM method string (default 'B_NW_SL')
    alpha  : significance level for FDR-controlled adjacency

    Returns
    -------
    dict with keys:
        'z_score'  : (p, p) ndarray — asymptotic z-scores
        'p_value'  : (p, p) ndarray — two-sided p-values
        'adj'      : (p, p) integer ndarray — FDR-controlled adjacency
    """
    try:
        import rpy2.robjects as ro
        import rpy2.robjects.numpy2ri as numpy2ri
        from rpy2.robjects.packages import importr
    except ImportError as exc:
        raise ImportError(
            "rpy2 is not installed.  Install with: pip install rpy2"
        ) from exc

    numpy2ri.activate()
    silggm = importr("SILGGM")

    r_X = ro.r.matrix(
        ro.FloatVector(X.flatten(order="F")),
        nrow=X.shape[0],
        ncol=X.shape[1],
    )

    result = silggm.SILGGM(r_X, method=method, alpha=alpha)

    p = X.shape[1]
    z_mat = np.array(result.rx2("z_score")).reshape(p, p, order="F")
    p_mat = np.array(result.rx2("p_value")).reshape(p, p, order="F")
    adj_mat = np.array(result.rx2("Graphs")).reshape(p, p, order="F").astype(int)

    numpy2ri.deactivate()

    return {"z_score": z_mat, "p_value": p_mat, "adj": adj_mat}
