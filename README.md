# NODIS — NOdewise De-sparsified Inference Statistics

**Python-native statistical inference for Gaussian Graphical Models**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org)
[![Version](https://img.shields.io/badge/version-0.1.0-green)](pyproject.toml)

---

## Overview

NODIS is a Python package for edge-level statistical inference in Gaussian Graphical Models (GGMs). It implements the de-sparsified (de-biased) nodewise Lasso estimator of van de Geer et al. (2014), providing asymptotically valid p-values, confidence intervals, and FDR-controlled adjacency matrices for high-dimensional gene co-expression networks.

NODIS is the **first Python-native tool** to deliver these guarantees without requiring R or proprietary software. It is designed as a rigorous benchmark scaffold for network inference methods in transcriptomics.

---

## Features

- **De-sparsified nodewise Lasso** — asymptotically valid edge-level z-scores and p-values under the null hypothesis ω_ij = 0
- **FDR control** — Benjamini–Hochberg and Benjamini–Yekutieli procedures via `scipy.stats.false_discovery_control`
- **Confidence intervals** — asymptotic and ensemble-based CIs for precision matrix entries
- **Nonparanormal transform** — rank-based shrinkage NPN matching `huge::huge.npn` (pure NumPy/SciPy, no R)
- **Synthetic data generator** — four network topologies (hub, scale-free, cluster, random) with guaranteed positive-definite precision matrices
- **Benchmark suite** — parallel multi-method runner with AUPR, AUROC, F1, MCC, and SHD metrics against DREAM5 and SERGIO benchmarks
- **Baseline estimators** — sklearn GraphicalLassoCV and GGLasso wrappers with a uniform API
- **AnnData compatibility** — direct ingestion of `AnnData` objects for single-cell workflows
- **CLI** — `nodis simulate / run / evaluate / plot` via Click

---

## Installation

```bash
pip install nodis
```

**With optional GGLasso baseline:**
```bash
pip install "nodis[gglasso]"
```

**With R/SILGGM validation bridge:**
```bash
pip install "nodis[r]"
```

**Development install:**
```bash
git clone https://github.com/youwillfindinfinity/nodis
cd nodis
pip install -e ".[dev]"
```

**From `requirements.txt`:**
```bash
pip install -r requirements.txt
pip install -e .
```

### Virtual environment setup

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -e ".[dev,gglasso]"    # full development install
```

### HPC (Snellius) setup

```bash
cp .env.template .env
# Edit .env with your Snellius username and password
source .env
rsync -avz --exclude='.venv/' . ${SNELLIUS_USER}@${SNELLIUS_HOST}:${SNELLIUS_REMOTE_DIR}/
ssh ${SNELLIUS_USER}@${SNELLIUS_HOST} \
    "cd ${SNELLIUS_REMOTE_DIR} && python -m venv .venv && source .venv/bin/activate && pip install -e '.[gglasso]'"
```

---

## Quick Start

```python
import numpy as np
from nodis import DesparifiedGGM
from nodis.preprocess.npn import npn_shrinkage
from nodis.inference.fdr import fdr_adjacency

# Prepare expression matrix (samples × genes)
X = npn_shrinkage(your_expression_matrix)   # nonparanormal transform

# Fit de-sparsified estimator
est = DesparifiedGGM()
est.fit(X)

# Edge-level inference
result = est.result_
print(result.z_scores.shape)   # (p, p) matrix of z-scores
print(result.p_values.shape)   # (p, p) two-sided p-values

# FDR-controlled adjacency (BH at α = 0.05)
adj = fdr_adjacency(result.p_values, alpha=0.05, method="bh")
n_edges = adj[np.triu_indices(adj.shape[0], k=1)].sum()
print(f"Selected edges: {n_edges}")
```

**From AnnData (single-cell):**
```python
from nodis import from_anndata

est = from_anndata(adata, layer="log1p")
est.fit()
```

**CLI:**
```bash
# Simulate a hub network and run inference
nodis simulate --topology hub --n 200 --p 50 --out data/sim.pkl
nodis run     --input data/sim.pkl --method desparsified --out results/
nodis evaluate --results results/ --truth data/sim.pkl
```

---

## Mathematical Background

The de-sparsified estimator for column *j* proceeds as:

```
λ_j  = λ_scale · √(2 log(p / √n) / n)        [scaled Lasso tuning]

Regress X_j on X_{−j} via Lasso(α = λ_j)
z_j  = X_j − X_{−j} β̂_j                      [nodewise residuals]
τ²_j = ‖z_j‖² / n                             [nodewise variance]

De-biased precision entry:
ω̂_ij = −β̂_ij / τ²_i − (τ²_i z_j^T X_i + τ²_j z_i^T X_j) / (2n)

Asymptotic variance:  σ²_ij = τ²_i · τ²_j
Z-score:              Z_ij  = √n · ω̂_ij / σ_ij  →  N(0,1) under H₀
P-value:              p_ij  = 2(1 − Φ(|Z_ij|))
```

**References:**
- van de Geer S, Bühlmann P, Ritov Y, Dezeure R (2014). *On asymptotically optimal confidence regions and tests for high-dimensional models.* Ann Statist 42(3): 1166–1202.
- Zhang CH, Zhang SS (2014). *Confidence intervals for low dimensional parameters in high dimensional linear models.* J R Stat Soc B 76(1): 217–242.

---

## Package Structure

```
nodis/
├── estimators/
│   ├── desparsified.py     ← DesparifiedGGM (core estimator)
│   ├── glasso.py           ← SklearnGLasso, GGLassoEstimator (baselines)
│   ├── piglasso.py         ← PIGLassoEstimator (stability selection + prior)
│   └── prior_utils.py      ← build_corr_prior, build_noisy_oracle_prior
├── inference/
│   ├── fdr.py              ← BH/BY FDR control
│   ├── confidence.py       ← asymptotic & ensemble confidence intervals
│   ├── pvalues.py          ← p-value computation
│   └── stars.py            ← StARS stability selection
├── preprocess/
│   ├── npn.py              ← nonparanormal shrinkage transform
│   └── anndata_compat.py   ← AnnData ingestion
├── simulate/
│   ├── generator.py        ← synthetic GGM data (4 topologies)
│   └── loaders.py          ← DREAM5, SERGIO data loaders
├── benchmark/
│   ├── runner.py           ← parallel multi-method benchmark runner
│   └── evaluate.py         ← AUPR, AUROC, F1, MCC, SHD
├── visualise/
│   └── plots.py            ← network and metric visualisation
└── cli.py                  ← Click CLI (simulate/run/evaluate/plot)
```

---

## Benchmarks

NODIS is validated against SILGGM (R), DREAM5 Network 1 (E. coli in silico), and SERGIO single-cell simulations. Parity tests target Pearson r > 0.99 between NODIS and SILGGM z-scores on matched synthetic replicates.

Benchmark scripts are in `benchmarks/` and SLURM array jobs for HPC execution are in `jobs/`.

---

## Testing

```bash
pytest tests/ -v
pytest tests/unit/          # unit tests (fast)
pytest tests/integration/   # parity tests (requires R + SILGGM)
```

---

## Citation

If you use NODIS in your research, please cite:

> **Bumbuc RV, Blei ZA** (2026). *NODIS: Python-native de-sparsified inference for Gaussian Graphical Models.* S
---

## Authors

| Name | Role | Affiliation |
|------|------|-------------|
| **Roland V. Bumbuc** | Lead developer, First and corresponding author | Amsterdam UMC, University of Amsterdam |

**Corresponding author:** Roland V. Bumbuc — rbumbuc@gmail.com

---

## Licence

MIT © 2026 Roland V. Bumbuc
