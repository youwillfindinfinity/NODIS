# API Reference

Full autodoc reference for all public NODIS modules. Class members with leading
underscores are private and excluded. Inherited members from scikit-learn base
classes are shown where relevant.

---

## Core Inference

### DesparifiedGGM

```{eval-rst}
.. autoclass:: nodis.estimators.desparsified.DesparifiedGGM
   :members:
   :show-inheritance:
```

### GGMInferenceResult

```{eval-rst}
.. autoclass:: nodis.estimators.desparsified.GGMInferenceResult
   :members:
   :show-inheritance:
```

---

## Baseline Estimators

### SklearnGLasso

```{eval-rst}
.. autoclass:: nodis.estimators.glasso.SklearnGLasso
   :members:
   :show-inheritance:
```

### GGLassoEstimator

```{eval-rst}
.. autoclass:: nodis.estimators.glasso.GGLassoEstimator
   :members:
   :show-inheritance:
```

---

## Statistical Inference

### fdr_control

```{eval-rst}
.. autofunction:: nodis.inference.fdr.fdr_control
```

### asymptotic_ci

```{eval-rst}
.. autofunction:: nodis.inference.confidence.asymptotic_ci
```

### ensemble_ci

```{eval-rst}
.. autofunction:: nodis.inference.confidence.ensemble_ci
```

### stars_select

```{eval-rst}
.. autofunction:: nodis.inference.stars.stars_select
```

---

## Preprocessing

### npn_shrinkage

```{eval-rst}
.. autofunction:: nodis.preprocess.npn.npn_shrinkage
```

---

## Simulation

### generate

```{eval-rst}
.. autofunction:: nodis.simulate.generator.generate
```

---

## Benchmarking

### evaluate_predictions

```{eval-rst}
.. autofunction:: nodis.benchmark.evaluate.evaluate_predictions
```

### run_benchmark

```{eval-rst}
.. autofunction:: nodis.benchmark.runner.run_benchmark
```

### evaluate_diffusion

```{eval-rst}
.. autofunction:: nodis.benchmark.diffusion_eval.evaluate_diffusion
```

### evaluate_knockouts

```{eval-rst}
.. autofunction:: nodis.benchmark.diffusion_eval.evaluate_knockouts
```

---

## Enrichment

### from_result

```{eval-rst}
.. autofunction:: nodis.enrich.from_result
```

### from_adjacency

```{eval-rst}
.. autofunction:: nodis.enrich.from_adjacency
```

### EnrichmentResult

```{eval-rst}
.. autoclass:: nodis.enrich.EnrichmentResult
   :members:
   :show-inheritance:
```
