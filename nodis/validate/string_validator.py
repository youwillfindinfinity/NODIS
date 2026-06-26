"""
Validate an inferred GGM adjacency against STRING interactions.

validate_against_string(adj, gene_names, organism, score_threshold) → ValidationResult

ValidationResult fields
-----------------------
n_inferred_edges   : int    edges in FDR-controlled adjacency
n_string_edges     : int    STRING edges for this gene set at threshold
n_overlap          : int    edges in both
precision          : float  TP / (TP + FP)  vs STRING
recall             : float  TP / (TP + FN)  vs STRING
f1                 : float  harmonic mean
jaccard            : float  overlap / union
string_score_dist  : np.ndarray  STRING scores of edges in inferred network
                    (to assess whether inferred edges tend to be high-confidence)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ValidationResult:
    n_inferred_edges: int
    n_string_edges: int
    n_overlap: int
    precision: float
    recall: float
    f1: float
    jaccard: float
    string_score_dist: np.ndarray


def validate_against_string(
    adj: np.ndarray,
    gene_names: list[str],
    organism: int = 9606,
    score_threshold: int = 400,
    score_type: str = "combined",
) -> ValidationResult:
    """Compute precision/recall of inferred network vs STRING."""
    from nodis.priors.string_prior import prior_from_string

    string_mat = prior_from_string(
        gene_names, organism=organism,
        score_threshold=score_threshold, score_type=score_type,
    )
    # Binary STRING adjacency (any edge that passed score_threshold → score > 0)
    string_adj = (string_mat > 0).astype(int)

    # Upper triangle only (symmetric, no double-counting)
    uidx = np.triu_indices(len(gene_names), k=1)
    pred = adj[uidx].astype(bool)
    ref = string_adj[uidx].astype(bool)

    tp = int((pred & ref).sum())
    fp = int((pred & ~ref).sum())
    fn = int((~pred & ref).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0 else 0.0
    )
    jaccard = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0

    # STRING scores for the inferred edges (may be 0 if not in STRING)
    inferred_pairs = np.where(pred)[0]
    score_dist = string_mat[uidx][inferred_pairs]

    return ValidationResult(
        n_inferred_edges=int(pred.sum()),
        n_string_edges=int(ref.sum()),
        n_overlap=tp,
        precision=precision,
        recall=recall,
        f1=f1,
        jaccard=jaccard,
        string_score_dist=score_dist,
    )
