<p align="center">
  <img src="NODIS_logo.png" alt="NODIS logo" width="480"/>
</p>

<h1 align="center">NODIS — NOdewise De-sparsified Inference Statistics</h1>

<p align="center"><strong>Python-native statistical inference for Gaussian Graphical Models</strong></p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License: MIT"/></a>
  <a href="https://www.python.org"><img src="https://img.shields.io/badge/Python-3.10%2B-blue" alt="Python"/></a>
  <a href="pyproject.toml"><img src="https://img.shields.io/badge/version-0.1.0-green" alt="Version"/></a>
</p>

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
- **CLI** — `nodis simulate / run / evaluate / enrich` via Click

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

---

## CLI Reference

NODIS ships a `nodis` command-line tool built with Click. The four commands follow the natural scientific workflow: **simulate → run → evaluate → enrich**.

```
nodis [--version] [--help] COMMAND [ARGS]...
```

---

### Scientific workflow order

```
1. nodis simulate   — generate benchmark data (or supply your own CSV)
2. nodis run        — preprocess + fit GGM → edges with p-values
3. nodis evaluate   — score predicted network against ground truth
4. nodis enrich     — interpret the network biologically
```

---

### 1. `nodis simulate` — Generate synthetic GGM datasets

Generates `--reps` replicate datasets, each saved as a pickle file (`{topology}_n{n}_p{p}_rep{rep:03d}.pkl`).

```bash
nodis simulate [OPTIONS]
```

| Option | Default | Description |
|---|---|---|
| `--n INT` | `200` | Number of samples |
| `--p INT` | `100` | Number of genes (nodes) |
| `--topology [hub\|scale-free\|cluster\|random]` | `hub` | Graph topology |
| `--reps INT` | `10` | Number of replicates |
| `--prob FLOAT` | `0.05` | Edge density (only used for `random` topology) |
| `--seed INT` | `42` | Base random seed (replicate *i* uses `seed + i`) |
| `--out PATH` | `results/simulated/` | Output directory |

**Topologies:**
- `hub` — star-shaped hubs, few high-degree nodes
- `scale-free` — Barabási–Albert power-law degree distribution
- `cluster` — block-diagonal (community) structure
- `random` — Erdős–Rényi with edge probability `--prob`

**Examples:**
```bash
# 20 hub replicates, 300 samples, 200 genes
nodis simulate --topology hub --n 300 --p 200 --reps 20 --out data/sim/

# Scale-free network, 5 replicates, fixed seed
nodis simulate --topology scale-free --n 150 --p 80 --reps 5 --seed 0

# Random sparse network (ER, 3% edge density)
nodis simulate --topology random --prob 0.03 --n 200 --p 100
```

---

### 2. `nodis run` — Fit a GGM and produce edge statistics

Reads a CSV expression matrix (samples × genes, first column used as index) and writes p-values, z-scores, and the FDR-controlled adjacency matrix to `--out`.

```bash
nodis run --data PATH [OPTIONS]
```

| Option | Default | Description |
|---|---|---|
| `--data PATH` | *required* | Expression matrix CSV (samples × genes) |
| `--method [desparsified\|glasso\|gglasso]` | `desparsified` | Inference method |
| `--alpha FLOAT` | `0.05` | FDR threshold α |
| `--fdr [BH\|BY]` | `BH` | FDR procedure (Benjamini–Hochberg or Benjamini–Yekutieli) |
| `--npn` | off | Apply nonparanormal shrinkage before fitting |
| `--out PATH` | `results/` | Output directory |

**Output files (method = `desparsified`):**

| File | Contents |
|---|---|
| `{stem}_pvalues.csv` | (p × p) two-sided edge p-values |
| `{stem}_zscores.csv` | (p × p) de-sparsified z-scores |
| `{stem}_adjacency.csv` | (p × p) binary FDR-controlled adjacency |

For `glasso` and `gglasso` only `{stem}_adjacency.csv` is written (no p-values).

**Methods:**
- `desparsified` — de-sparsified nodewise Lasso (van de Geer et al. 2014); provides edge p-values and CIs
- `glasso` — sklearn `GraphicalLassoCV`; produces a sparse precision matrix, no p-values
- `gglasso` — GGLasso group graphical Lasso; requires `pip install gglasso`

**Examples:**
```bash
# Full de-sparsified pipeline with NPN preprocessing
nodis run --data expr.csv --method desparsified --npn --alpha 0.05 --out results/

# Graphical Lasso baseline (no p-values)
nodis run --data expr.csv --method glasso --out results/glasso/

# Stricter FDR control with BY procedure
nodis run --data expr.csv --npn --fdr BY --alpha 0.01 --out results/strict/
```

---

### 3. `nodis evaluate` — Score predicted network against ground truth

Computes classification metrics comparing a predicted adjacency to a known ground-truth adjacency. Optionally accepts a continuous score matrix for AUPR/AUROC.

```bash
nodis evaluate --predicted PATH --ground-truth PATH [OPTIONS]
```

| Option | Default | Description |
|---|---|---|
| `--predicted PATH` | *required* | Predicted adjacency CSV (binary, 0/1) |
| `--ground-truth PATH` | *required* | Ground-truth adjacency CSV (binary, 0/1) |
| `--scores PATH` | `None` | Continuous score matrix CSV (e.g. 1 − p-value) for AUPR/AUROC |
| `--out PATH` | `results/metrics.csv` | Output metrics CSV |

**Metrics reported:**

| Metric | Notes |
|---|---|
| F1 | Harmonic mean of precision and recall |
| MCC | Matthews Correlation Coefficient |
| SHD | Structural Hamming Distance |
| AUPR | Area under precision-recall curve (requires `--scores`) |
| AUROC | Area under ROC curve (requires `--scores`) |

**Examples:**
```bash
# Binary adjacency comparison only
nodis evaluate \
  --predicted  results/expr_adjacency.csv \
  --ground-truth data/true_adj.csv \
  --out results/metrics.csv

# Full evaluation with continuous scores (AUPR + AUROC)
nodis evaluate \
  --predicted    results/expr_adjacency.csv \
  --ground-truth data/true_adj.csv \
  --scores       results/expr_pvalues.csv \
  --out          results/metrics_full.csv
```

---

### 4. `nodis enrich` — Topology-aware gene set enrichment

Interprets a GGM adjacency matrix biologically by extracting hub genes or community structure and running enrichment analysis across three biological levels.

```bash
nodis enrich --adj PATH --genes PATH [OPTIONS]
```

| Option | Default | Description |
|---|---|---|
| `--adj PATH` | *required* | Binary adjacency matrix (`.npy`) |
| `--genes PATH` | *required* | Text file with one gene name per line |
| `--pvalues PATH` | `None` | Edge p-value matrix (`.npy`); required for `prerank` method |
| `--level [rna\|post_transcriptional\|protein\|all]` | `all` | Biological level(s) to query |
| `--method [ora\|prerank]` | `ora` | Enrichment method |
| `--backend [gprofiler\|gseapy]` | `gprofiler` | Enrichment backend |
| `--extraction [hub\|prerank\|community]` | `hub` | Gene extraction strategy from network |
| `--organism STR` | `hsapiens` | Organism code (g:Profiler format) |
| `--out PATH` | `enrichment_results.csv` | Output CSV |

**Biological levels:**
- `rna` — GO Biological Process/Molecular Function, KEGG, Reactome
- `post_transcriptional` — miRNA targets, transcription factor motifs
- `protein` — CORUM protein complexes, InterPro domains
- `all` — all three combined

**Gene extraction strategies:**
- `hub` — top-degree nodes in the adjacency graph
- `community` — genes per detected network community (Louvain/Leiden)
- `prerank` — all genes ranked by edge p-value sum (requires `--pvalues`)

**Examples:**
```bash
# ORA on hub genes, all biological levels
nodis enrich \
  --adj   results/expr_adjacency.npy \
  --genes gene_list.txt \
  --level all --method ora --extraction hub \
  --out   enrichment.csv

# GSEA prerank using edge p-values
nodis enrich \
  --adj      results/expr_adjacency.npy \
  --genes    gene_list.txt \
  --pvalues  results/expr_pvalues.npy \
  --method   prerank --extraction prerank \
  --backend  gseapy \
  --out      gsea_results.csv

# Protein-level enrichment for mouse data
nodis enrich \
  --adj results/adj.npy --genes genes.txt \
  --level protein --organism mmusculus
```

---

### End-to-end example (synthetic benchmark)

```bash
# 1. Generate 10 hub-topology replicates
nodis simulate --topology hub --n 200 --p 100 --reps 10 --out data/sim/

# 2. Run de-sparsified inference with NPN on replicate 0
nodis run \
  --data   data/sim/hub_n200_p100_rep000.csv \
  --method desparsified \
  --npn \
  --alpha  0.05 \
  --fdr    BH \
  --out    results/rep000/

# 3. Score against ground truth
nodis evaluate \
  --predicted    results/rep000/hub_n200_p100_rep000_adjacency.csv \
  --ground-truth data/sim/hub_n200_p100_rep000_true_adj.csv \
  --scores       results/rep000/hub_n200_p100_rep000_pvalues.csv \
  --out          results/rep000/metrics.csv

# 4. Biological enrichment
nodis enrich \
  --adj   results/rep000/hub_n200_p100_rep000_adjacency.npy \
  --genes data/gene_names.txt \
  --level all \
  --out   results/rep000/enrichment.csv
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

> **Bumbuc RV** (2026). *NODIS: Python-native de-sparsified inference for Gaussian Graphical Models.* S
---

## Authors

| Name | Role | Affiliation |
|------|------|-------------|
| **Roland V. Bumbuc** | Lead developer, First and corresponding author | Amsterdam UMC, University of Amsterdam |

**Corresponding author:** Roland V. Bumbuc — rbumbuc@gmail.com

---

## Licence

MIT © 2026 Roland V. Bumbuc
