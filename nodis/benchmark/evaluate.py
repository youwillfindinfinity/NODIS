"""
Network inference evaluation metrics.

All metrics operate on a predicted binary adjacency matrix and a binary
ground-truth adjacency matrix.  Both are assumed symmetric with no self-loops.

Metrics
-------
AUPR    — Area Under the Precision-Recall Curve
AUROC   — Area Under the Receiver Operating Characteristic Curve
F1      — F1 score at the fixed decision threshold encoded in adj_pred
F1_opt  — F1 score at the score threshold that maximises F1 (oracle upper bound)
MCC     — Matthews Correlation Coefficient
SHD     — Structural Hamming Distance (number of edge insertions + deletions)
"""

import numpy as np


def _upper_triangle(A: np.ndarray) -> np.ndarray:
    """Return upper-triangle entries (k=1) as a 1-D array."""
    idx = np.triu_indices(A.shape[0], k=1)
    return A[idx]


def evaluate_predictions(
    adj_pred: np.ndarray,
    adj_true: np.ndarray,
    scores: np.ndarray | None = None,
) -> dict:
    """
    Compute all evaluation metrics for a predicted network.

    Parameters
    ----------
    adj_pred : (p, p) integer ndarray — predicted binary adjacency
    adj_true : (p, p) integer ndarray — ground-truth binary adjacency
    scores   : (p, p) ndarray or None — continuous edge scores (e.g., |z-scores|)
               for AUPR / AUROC computation.  If None, adj_pred is used as scores.

    Returns
    -------
    dict with keys: 'aupr', 'auroc', 'f1', 'f1_opt', 'mcc', 'shd',
                    'precision', 'recall', 'tp', 'fp', 'tn', 'fn'
    """
    y_true = _upper_triangle(adj_true).astype(int)
    y_pred = _upper_triangle(adj_pred).astype(int)
    y_score = _upper_triangle(scores) if scores is not None else y_pred.astype(float)

    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())

    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    denom_mcc = np.sqrt(np.float64(tp + fp) * np.float64(tp + fn) * np.float64(tn + fp) * np.float64(tn + fn))
    mcc = float(np.float64(tp * tn - fp * fn) / denom_mcc) if denom_mcc > 0 else 0.0

    shd = int((y_pred != y_true).sum())

    aupr = _aupr(y_true, y_score)
    auroc = _auroc(y_true, y_score)
    f1_opt = _f1_optimal(y_true, y_score)

    return {
        "aupr": aupr, "auroc": auroc, "f1": f1, "f1_opt": f1_opt,
        "mcc": mcc, "shd": shd,
        "precision": prec, "recall": rec,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    }


def _aupr(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute AUPR via the trapezoid rule on the precision-recall curve.

    Notes
    -----
    Uses ``sklearn.metrics.average_precision_score``, which computes AUPR via
    step-function (right-weighted) interpolation, not trapezoidal. Results may
    differ from linear-interpolation AUPR used in some DREAM challenge evaluations.
    When comparing with external benchmarks, confirm the interpolation convention.
    """
    from sklearn.metrics import average_precision_score
    if y_true.sum() == 0:
        return float("nan")
    return float(average_precision_score(y_true, y_score))


def _auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute AUROC."""
    from sklearn.metrics import roc_auc_score
    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def _f1_optimal(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """F1 at the score threshold that maximises F1 (oracle upper bound).

    Sweeps all unique score values as candidate thresholds and returns the
    highest achievable F1.  This is threshold-agnostic and comparable across
    methods regardless of how their adjacency matrices were binarised.
    """
    from sklearn.metrics import precision_recall_curve
    if y_true.sum() == 0:
        return float("nan")
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    denom = precision + recall
    f1_scores = np.where(denom > 0, 2 * precision * recall / denom, 0.0)
    return float(f1_scores.max())
