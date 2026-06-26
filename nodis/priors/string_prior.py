"""
Build a (p × p) prior matrix from STRING protein interaction data.

prior_from_string(gene_names, organism, score_threshold, score_type) → np.ndarray

Calls the STRING REST API v11.5:
  https://string-db.org/api/json/network?identifiers=...

Returns a symmetric float matrix where entry [i, j] is the normalised
combined STRING score for the gene pair, scaled to [0, 1].
Pairs not found in STRING are assigned 0.0 (no prior belief).

Parameters
----------
gene_names       : list[str]   HGNC gene symbols (or Ensembl IDs with species prefix)
organism         : int         NCBI taxon ID (9606 = human, 10090 = mouse, 511145 = E. coli)
score_threshold  : int         Minimum combined score [0, 1000]; default 400 (medium confidence)
score_type       : str         "combined" | "experimental" | "coexpression" | "database"
                               default "combined"
caller_identity  : str         Email or URL for STRING rate-limiting; default "nodis"

Returns
-------
np.ndarray  shape (p, p), dtype float64, symmetric, diagonal 0
"""
from __future__ import annotations

import json
import urllib.request
import urllib.parse
import warnings

import numpy as np


STRING_API_URL = "https://string-db.org/api/json/network"
STRING_SCORES = (
    "combined", "experimental", "coexpression", "database",
    "textmining", "cooccurence", "fusion", "neighborhood",
)
_CHUNK_SIZE = 2000  # STRING API limit per request


def prior_from_string(
    gene_names: list[str],
    organism: int = 9606,
    score_threshold: int = 400,
    score_type: str = "combined",
    caller_identity: str = "nodis",
) -> np.ndarray:
    """Query STRING REST API and return a prior matrix for gene_names."""
    if score_type not in STRING_SCORES:
        raise ValueError(f"score_type must be one of {STRING_SCORES}")

    if len(gene_names) != len(set(gene_names)):
        raise ValueError("duplicate gene names detected in gene_names")

    p = len(gene_names)
    prior = np.zeros((p, p), dtype=np.float64)
    idx = {g: i for i, g in enumerate(gene_names)}

    # Chunk into batches of _CHUNK_SIZE to respect the STRING API limit
    chunks = [gene_names[i:i + _CHUNK_SIZE] for i in range(0, p, _CHUNK_SIZE)]
    for chunk in chunks:
        _query_chunk(chunk, idx, prior, organism, score_threshold,
                     score_type, caller_identity)

    return prior


def _query_chunk(
    chunk: list[str],
    idx: dict[str, int],
    prior: np.ndarray,
    organism: int,
    score_threshold: int,
    score_type: str,
    caller_identity: str,
) -> None:
    """Fill `prior` in-place for one chunk of gene names."""
    identifiers = "%0d".join(chunk)
    params = urllib.parse.urlencode({
        "identifiers": identifiers,
        "species": organism,
        "required_score": score_threshold,
        "caller_identity": caller_identity,
        "network_type": "functional",
    })

    url = f"{STRING_API_URL}?{params}"
    try:
        with urllib.request.urlopen(url, timeout=60) as resp:
            data = json.loads(resp.read().decode())
    except OSError as exc:
        raise ConnectionError(
            f"STRING API request failed: {exc}. "
            "Check your network connection or try again later."
        ) from exc

    for edge in data:
        g1 = edge.get("preferredName_A")
        g2 = edge.get("preferredName_B")
        # STRING returns field "score" for combined; other types use "score_<type>"
        score_val = edge.get(f"score_{score_type}", edge.get("score", 0))

        if g1 not in idx or g2 not in idx:
            if g1 not in idx and g1 is not None:
                warnings.warn(f"Gene '{g1}' from STRING response not in gene_names.")
            if g2 not in idx and g2 is not None:
                warnings.warn(f"Gene '{g2}' from STRING response not in gene_names.")
            continue

        i, j = idx[g1], idx[g2]
        norm_score = float(score_val) / 1000.0
        prior[i, j] = norm_score
        prior[j, i] = norm_score
