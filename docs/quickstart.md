# Quickstart

This page walks through the main NODIS workflow end-to-end: generating synthetic data,
running de-sparsified inference, controlling FDR, computing confidence intervals,
pathway enrichment, and benchmarking.

## 1. Generate Synthetic Data

```python
from nodis.simulate.generator import generate

# Hub topology: 200 samples, 50 genes, random seed for reproducibility
X, adj_true, precision_true = generate(
    n=200,
    p=50,
    topology="hub",   # options: "hub", "scale_free", "cluster", "random"
    seed=42,
)

print(X.shape)          # (200, 50)
print(adj_true.sum())   # number of directed edges (symmetric matrix)
```

Available topologies:

| Topology     | Description                                      |
|---|---|
| `hub`        | One central hub node connected to all others     |
| `scale_free` | Barabási–Albert preferential attachment          |
| `cluster`    | Block-diagonal community structure               |
| `random`     | Erdős–Rényi random graph                         |

## 2. Run De-Sparsified Inference

```python
from nodis.estimators.desparsified import DesparifiedGGM

est = DesparifiedGGM(
    lambda_scale=1.0,   # scaling factor for nodewise Lasso tuning parameter
    n_jobs=-1,          # parallelise nodewise regressions across all CPU cores
).fit(X)

# Fitted attributes
result = est.result_
print(result.pvalues.shape)    # (50, 50) symmetric p-value matrix
print(result.zscores.shape)    # (50, 50) Z-score matrix
```

The tuning parameter per node j is computed as:

```
lambda_j = lambda_scale * sqrt(2 * log(p / sqrt(n)) / n)
```

## 3. FDR-Controlled Adjacency

```python
# Binary adjacency matrix at 5% FDR (Benjamini-Hochberg)
adj_hat = est.get_adjacency(alpha=0.05, method="bh")

print(f"Edges discovered: {int(adj_hat.sum()) // 2}")

# Alternatively, call fdr_control directly on any p-value matrix
from nodis.inference.fdr import fdr_control

adj_by = fdr_control(result.pvalues, alpha=0.05, method="by")  # Benjamini-Yekutieli
```

## 4. Confidence Intervals

```python
from nodis.inference.confidence import asymptotic_ci, ensemble_ci

# Asymptotic 95% CIs from the de-sparsified variance estimate
lower, upper = asymptotic_ci(result, alpha=0.05)

print(lower.shape)   # (50, 50)
print(upper.shape)   # (50, 50)

# Ensemble CIs (aggregated over bootstrap replicates, if available)
lower_ens, upper_ens = ensemble_ci(result, alpha=0.05)
```

## 5. Topology-Aware Enrichment

```python
from nodis.enrich import from_result

gene_names = [f"GENE_{i}" for i in range(50)]   # replace with real HGNC symbols

enrichment = from_result(
    result,
    gene_names=gene_names,
    level="rna",         # "rna" or "protein"
    organism="hsapiens",
    sources=["GO:BP", "KEGG", "REAC"],
)

# EnrichmentResult attributes
print(enrichment.summary())          # top enriched terms
print(enrichment.to_dataframe())     # full results as pandas DataFrame
```

You can also enrich directly from a binary adjacency matrix:

```python
from nodis.enrich import from_adjacency

enrichment = from_adjacency(
    adj_hat,
    gene_names=gene_names,
    level="rna",
    organism="hsapiens",
)
```

## 6. Run Benchmark

```python
from nodis.estimators.desparsified import DesparifiedGGM
from nodis.estimators.glasso import SklearnGLasso
from nodis.benchmark.runner import run_benchmark

estimators = {
    "nodis": DesparifiedGGM(),
    "glasso": SklearnGLasso(),
}

# run_benchmark returns a dict of {method_name: metrics_dict}
results = run_benchmark(
    estimators=estimators,
    X=X,
    adj_true=adj_true,
    alpha=0.05,
)

for method, metrics in results.items():
    print(f"{method}: AUROC={metrics['auroc']:.3f}  AUPR={metrics['aupr']:.3f}")
```

## 7. CLI

NODIS ships a `nodis` command-line interface built with Click.

**Generate synthetic data:**

```bash
nodis simulate --n 200 --p 50 --topology hub --seed 42 --output data/hub_n200_p50.pkl
```

**Run inference on a data file:**

```bash
nodis run --input data/hub_n200_p50.pkl --method desparsified --alpha 0.05 \
    --output results/hub_n200_p50_nodis.csv
```

**Evaluate predictions against ground truth:**

```bash
nodis evaluate --pred results/hub_n200_p50_nodis.csv \
    --true data/hub_n200_p50_adj.csv
```

Run `nodis --help` or `nodis <subcommand> --help` for full option listings.
