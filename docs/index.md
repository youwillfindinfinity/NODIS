# NODIS Documentation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyPI](https://img.shields.io/pypi/v/nodis.svg)](https://pypi.org/project/nodis/)

**NODIS** (NOdewise De-sparsified Inference Statistics) is the first Python-native
implementation of de-sparsified (de-biased) nodewise Lasso inference for Gaussian
Graphical Models (GGMs). It delivers **edge-level p-values**, **confidence intervals**,
and **FDR-controlled adjacency matrices** for high-dimensional gene co-expression
networks — no R dependency required for the core inference path.

## Why NODIS?

Most tools for sparse GGM estimation (graphical Lasso, SILGGM) either require R or
lack formal inferential guarantees. NODIS addresses this gap:

- **First Python-native de-sparsified GGM inference** — implements van de Geer et al.
  (2014) and Zhang & Zhang (2014) entirely in NumPy/SciPy/scikit-learn.
- **Edge-level p-values** — asymptotic Z-scores per edge, not just a point estimate of
  the precision matrix.
- **FDR control** — Benjamini–Hochberg and Benjamini–Yekutieli procedures via
  `scipy.stats.false_discovery_control`, turning raw p-values into a controlled
  adjacency matrix at any desired false-discovery rate.
- **Topology-aware enrichment** — integrated pathway/ontology enrichment via `nodis.enrich`,
  mapping inferred edges to biological annotations (GO, KEGG, Reactome, STRING, and more)
  at RNA or protein level.
- **Benchmark-ready** — built-in synthetic data generator (hub, scale-free, cluster,
  random topologies), AUROC/AUPR/F1/MCC/SHD metrics, and a parallel multi-method runner
  designed for Snellius HPC array jobs.

```{toctree}
:maxdepth: 2
:caption: Contents

installation
quickstart
api
changelog
```

## Quick Example

```python
from nodis.simulate.generator import generate
from nodis.estimators.desparsified import DesparifiedGGM
from nodis.inference.fdr import fdr_control

# 1. Synthetic data (hub topology, n=200 samples, p=50 genes)
X, adj_true, _ = generate(n=200, p=50, topology="hub", seed=42)

# 2. Fit de-sparsified GGM
est = DesparifiedGGM().fit(X)

# 3. FDR-controlled adjacency at 5%
adj_hat = est.get_adjacency(alpha=0.05)

print(f"Edges discovered: {adj_hat.sum() // 2}")
```

## Citation

If you use NODIS in your research, please cite:

> Bumbuc, R. V. (2026). *NODIS: NOdewise De-sparsified Inference Statistics for
> Gaussian Graphical Models*. Journal of Open Source Software.
> https://doi.org/10.21105/joss.XXXXX

## Indices and Tables

- {ref}`genindex`
- {ref}`modindex`
- {ref}`search`
