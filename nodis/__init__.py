"""
NODIS (NOdewise De-sparsified Inference Statistics) — Python-native statistical
inference for Gaussian Graphical Models.

Core module: nodis.estimators.desparsified.DesparifiedGGM
"""

__version__ = "0.1.0"
__author__ = "Roland Bumbuc"

from nodis.estimators.desparsified import DesparifiedGGM, GGMInferenceResult
from nodis.preprocess.anndata_compat import from_anndata, to_anndata
from nodis.network import NetworkTopology, CommunityResult, HubResult
from nodis.estimators.group_glasso import MultiConditionGLasso, fit_multi_condition

__all__ = [
    "DesparifiedGGM", "GGMInferenceResult",
    "from_anndata", "to_anndata",
    "NetworkTopology", "CommunityResult", "HubResult",
    "MultiConditionGLasso", "fit_multi_condition",
]
