"""
Parallel multi-method benchmark runner.

Runs a list of estimators on a dataset and collects evaluation metrics,
optionally in parallel using joblib.
"""

from __future__ import annotations

import time
import traceback
from typing import Any

import numpy as np
from joblib import Parallel, delayed

from nodis.benchmark.evaluate import evaluate_predictions


def _run_single(
    estimator_cls: type,
    estimator_kwargs: dict,
    X: np.ndarray,
    adj_true: np.ndarray,
    alpha: float,
    method_name: str,
) -> dict:
    """Fit one estimator and return metrics.  Safe: catches and logs exceptions."""
    result: dict[str, Any] = {"method": method_name, "error": None}
    t0 = time.perf_counter()
    try:
        est = estimator_cls(**estimator_kwargs)
        est.fit(X)

        # Collect scores and adjacency
        if hasattr(est, "result_"):
            # DesparifiedGGM
            scores = np.abs(est.result_.z_scores)
            adj_pred = est.get_adjacency(alpha=alpha)
        elif hasattr(est, "precision_"):
            # GLasso-type
            scores = np.abs(est.precision_)
            adj_pred = est.get_adjacency(threshold=0.0)
        else:
            raise AttributeError("Estimator has neither result_ nor precision_ attribute.")

        metrics = evaluate_predictions(adj_pred, adj_true, scores=scores)
        result.update(metrics)
    except Exception:
        result["error"] = traceback.format_exc()
    result["wall_seconds"] = time.perf_counter() - t0
    return result


def run_benchmark(
    estimators: list[tuple[type, dict, str]],
    X: np.ndarray,
    adj_true: np.ndarray,
    alpha: float = 0.05,
    n_jobs: int = 1,
) -> list[dict]:
    """
    Run multiple estimators on X and return per-method metrics.

    Parameters
    ----------
    estimators : list of (EstimatorClass, kwargs_dict, name_str)
    X          : (n, p) expression matrix
    adj_true   : (p, p) ground-truth binary adjacency
    alpha      : FDR level for inference-based methods
    n_jobs     : parallel jobs (1 = sequential)

    Returns
    -------
    list of metric dicts, one per estimator
    """
    jobs = [
        delayed(_run_single)(cls, kw, X, adj_true, alpha, name)
        for cls, kw, name in estimators
    ]
    return Parallel(n_jobs=n_jobs)(jobs)
