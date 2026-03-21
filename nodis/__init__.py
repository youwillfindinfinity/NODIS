"""
NODIS (NOdewise De-sparsified Inference Statistics) — Python-native statistical
inference for Gaussian Graphical Models.

Core module: nodis.estimators.desparsified.DesparifiedGGM
"""

__version__ = "0.1.0"
__author__ = "Roland Bumbuc"

from nodis.estimators.desparsified import DesparifiedGGM, GGMInferenceResult
from nodis.preprocess.anndata_compat import from_anndata

__all__ = ["DesparifiedGGM", "GGMInferenceResult", "from_anndata"]
