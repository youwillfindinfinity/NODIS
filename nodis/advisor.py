"""
Estimator recommendation engine for NODIS.

recommend(n, p, goal, data_type) → AdvisorResult
AdvisorResult.estimator      : str   ("desparsified" | "glasso" | "gglasso" | "piglasso")
AdvisorResult.preset         : str   (preset name)
AdvisorResult.reasoning      : list[str]  (human-readable bullet points)
AdvisorResult.warnings       : list[str]
AdvisorResult.kwargs         : dict  (kwargs to pass directly to the estimator)
AdvisorResult.cli_snippet    : str   (ready-to-paste CLI command)
AdvisorResult.python_snippet : str
"""
from __future__ import annotations

from dataclasses import dataclass, field

GOALS = ("edge_pvalues", "network_structure", "module_detection", "differential")
DATA_TYPES = ("bulk_rnaseq", "scrna_seq", "proteomics", "methylation", "generic")

PRESETS: dict[str, dict] = {
    "bulk_rnaseq": {
        "method": "desparsified",
        "npn": True,
        "fdr": "BH",
        "alpha": 0.05,
        "pseudobulk": False,
    },
    "scrna_pseudobulk": {
        "method": "desparsified",
        "npn": True,
        "fdr": "BY",
        "alpha": 0.05,
        "pseudobulk": True,
    },
    "proteomics": {
        "method": "desparsified",
        "npn": True,
        "fdr": "BH",
        "alpha": 0.05,
        "pseudobulk": False,
    },
    "methylation": {
        "method": "glasso",
        "npn": False,
        "fdr": "BH",
        "alpha": 0.05,
        "pseudobulk": False,
    },
    "differential": {
        "method": "gglasso_fgl",
        "npn": True,
        "fdr": "BH",
        "alpha": 0.05,
        "pseudobulk": False,
    },
}

_ESTIMATOR_IMPORT = {
    "desparsified": "from nodis.estimators.desparsified import DesparifiedGGM",
    "glasso": "from nodis.estimators.glasso import SklearnGLasso",
    "gglasso": "from nodis.estimators.glasso import GGLassoEstimator",
    "gglasso_fgl": "from nodis.estimators.glasso import GGLassoEstimator",
    "piglasso": "from nodis.estimators.piglasso import PIGLassoEstimator",
}

_ESTIMATOR_CLASS = {
    "desparsified": "DesparifiedGGM",
    "glasso": "SklearnGLasso",
    "gglasso": "GGLassoEstimator",
    "gglasso_fgl": "GGLassoEstimator",
    "piglasso": "PIGLassoEstimator",
}


@dataclass
class AdvisorResult:
    estimator: str
    preset: str
    reasoning: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    kwargs: dict = field(default_factory=dict)
    cli_snippet: str = ""
    python_snippet: str = ""


def recommend(
    n: int,
    p: int,
    goal: str = "edge_pvalues",
    data_type: str = "bulk_rnaseq",
    n_conditions: int = 1,
    has_prior: bool = False,
) -> AdvisorResult:
    """Core decision function — deterministic, no side effects."""
    reasoning: list[str] = []
    warnings: list[str] = []
    kwargs: dict = {}
    estimator: str = ""

    # --- Primary decision: goal × n_conditions × prior ---
    if goal == "differential" and n_conditions >= 2:
        estimator = "gglasso_fgl"
        kwargs = {"lambda2": "auto"}
        reasoning.append(
            f"goal=differential with {n_conditions} conditions → FusedGraphicalLasso"
        )

    elif goal in ("edge_pvalues", "module_detection"):
        estimator = "desparsified"
        ratio = n / p
        reasoning.append(
            f"goal={goal} → de-sparsified nodewise Lasso (edge-level p-values)"
        )
        reasoning.append(f"n/p = {ratio:.2f} (n={n}, p={p})")
        if ratio < 2:
            kwargs["dof_correction"] = True
            kwargs["ensemble_ci"] = True
            warnings.append("n/p < 2: power is low; CI coverage may be imperfect")
            reasoning.append("n/p < 2: enabling dof_correction and ensemble_ci")
        elif ratio < 5:
            kwargs["dof_correction"] = True
            reasoning.append("2 ≤ n/p < 5: enabling dof_correction")
        else:
            kwargs["dof_correction"] = False
            reasoning.append("n/p ≥ 5: dof_correction not needed")

    elif goal == "network_structure":
        ratio = n / p
        reasoning.append(f"goal=network_structure, n/p = {ratio:.2f} (n={n}, p={p})")
        if has_prior:
            estimator = "piglasso"
            reasoning.append(
                "has_prior=True → PIGLasso (prior-informed graphical Lasso)"
            )
        elif n_conditions >= 2:
            estimator = "gglasso_fgl"
            reasoning.append(
                f"{n_conditions} conditions → FusedGraphicalLasso for joint estimation"
            )
        elif ratio > 10:
            estimator = "glasso"
            reasoning.append(
                f"n/p = {ratio:.2f} > 10: GLasso with CV is reliable and fast"
            )
        else:
            estimator = "gglasso"
            reasoning.append(
                f"n/p = {ratio:.2f}: GGLasso StARS λ selection more stable at low n/p"
            )

    # --- Data-type preset ---
    if data_type == "scrna_seq":
        warnings.append(
            "pseudobulk aggregation required before GGM; run nodis pseudobulk first"
        )
        preset = "scrna_pseudobulk"
    elif data_type == "proteomics":
        warnings.append("use quantile normalization before NPN, not log-norm")
        preset = "proteomics"
    elif data_type == "methylation":
        warnings.append("use M-values (logit of beta), not raw beta values")
        preset = "methylation"
    else:
        preset = "bulk_rnaseq"

    # --- Large-p adjustments ---
    if p > 2000 and estimator == "desparsified":
        kwargs["sparse"] = True
        warnings.append("sparse=True enabled for p > 2000; peak memory ~ O(p*n)")
    if p > 2000 and estimator == "glasso":
        warnings.append(
            "GLasso-CV stores dense p×p precision matrix; "
            "consider desparsified with sparse=True instead"
        )

    result = AdvisorResult(
        estimator=estimator,
        preset=preset,
        reasoning=reasoning,
        warnings=warnings,
        kwargs=kwargs,
    )
    result.cli_snippet = _build_cli_snippet(result)
    result.python_snippet = _build_python_snippet(result)
    return result


def _build_cli_snippet(result: AdvisorResult, data_path: str = "data.h5ad") -> str:
    parts = [
        "nodis run",
        f"--data {data_path}",
        f"--preset {result.preset}",
    ]
    return " \\\n    ".join(parts)


def _build_python_snippet(result: AdvisorResult) -> str:
    import_line = _ESTIMATOR_IMPORT.get(
        result.estimator, f"# from nodis import {result.estimator}"
    )
    class_name = _ESTIMATOR_CLASS.get(result.estimator, result.estimator)
    preset_cfg = PRESETS.get(result.preset, {})
    alpha = preset_cfg.get("alpha", 0.05)
    fdr = preset_cfg.get("fdr", "BH")

    # Filter internal-only kwargs that don't map to estimator __init__
    estimator_kwargs = {
        k: v for k, v in result.kwargs.items() if k not in ("lambda2",)
    }
    kwargs_repr = ", ".join(f"{k}={v!r}" for k, v in estimator_kwargs.items())
    est_call = f"{class_name}({kwargs_repr})" if kwargs_repr else f"{class_name}()"

    lines = [
        import_line,
        f"X = load_expression_matrix()  # shape (n_samples, n_genes)",
        f"est = {est_call}",
        f"est.fit(X)",
        f"adj = est.get_adjacency(alpha={alpha}, method='{fdr}')",
    ]
    return "\n".join(lines)
