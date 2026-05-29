# Changelog

All notable changes to NODIS are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [0.1.0] — 2026

### Added

- `nodis.estimators.desparsified` — `DesparifiedGGM` and `GGMInferenceResult`:
  Python-native de-sparsified nodewise Lasso inference (van de Geer et al. 2014;
  Zhang & Zhang 2014). Produces edge-level Z-scores and p-values.
- `nodis.estimators.glasso` — `SklearnGLasso` and `GGLassoEstimator`: scikit-learn
  and GGLasso wrappers with a uniform `fit` / `get_adjacency` API.
- `nodis.inference.fdr` — `fdr_control`: Benjamini–Hochberg and Benjamini–Yekutieli
  FDR correction via `scipy.stats.false_discovery_control`.
- `nodis.inference.confidence` — `asymptotic_ci`, `ensemble_ci`: asymptotic and
  ensemble-aggregated confidence intervals for precision matrix entries.
- `nodis.inference.stars` — `stars_select`: StARS stability-based regularisation
  selection for graphical Lasso.
- `nodis.preprocess.npn` — `npn_shrinkage`: rank-based nonparanormal transform
  matching `huge::huge.npn(method="shrinkage")`, pure NumPy/SciPy.
- `nodis.simulate.generator` — `generate`: synthetic GGM data generator supporting
  hub, scale-free (Barabási–Albert), cluster (block-diagonal), and random (Erdős–Rényi)
  topologies with positive-definiteness guarantees.
- `nodis.benchmark.evaluate` — `evaluate_predictions`: AUROC, AUPR, F1, MCC, and
  SHD metrics against ground-truth adjacency.
- `nodis.benchmark.runner` — `run_benchmark`: parallel multi-method benchmark runner
  designed for Snellius SLURM array jobs.
- `nodis.benchmark.diffusion_eval` — `evaluate_diffusion`, `evaluate_knockouts`:
  network diffusion and in-silico knockout evaluation utilities.
- `nodis.enrich` — `from_result`, `from_adjacency`, `EnrichmentResult`:
  topology-aware pathway/ontology enrichment supporting GO, KEGG, Reactome, STRING,
  and HP annotations at RNA and protein level.
- `nodis.cli` — Click-based CLI with `simulate`, `run`, `evaluate`, and `plot`
  subcommands.
- Sphinx documentation (this site).
- Full pytest suite with coverage for all public modules.
