# Installation

## Requirements

- Python 3.10 or later
- All core dependencies are installed automatically (NumPy, SciPy, scikit-learn,
  NetworkX, pandas, click)

## Install from PyPI

```bash
pip install nodis
```

## Install from Source

```bash
git clone https://github.com/rbumbuc/nodis.git
cd nodis
pip install -e .
```

For development (includes test and documentation dependencies):

```bash
pip install -e ".[dev]"
```

## Optional Dependencies

### Enrichment extras

Topology-aware enrichment via `nodis.enrich` requires additional backends:

```bash
pip install 'nodis[enrich]'
```

This installs `gprofiler-official`, `pydeseq2`, and `pyranges`. The `gseapy`
backend is used when available for GSEA/fGSEA-style enrichment.

### R integration

For parity validation against SILGGM (R package) and rpy2-backed workflows:

```bash
pip install 'nodis[r]'
```

This installs `rpy2`. You must also have R (>= 4.0) installed on your system with
the SILGGM package available:

```r
install.packages("SILGGM")
```

Set the `R_HOME` environment variable if rpy2 cannot locate your R installation:

```bash
export R_HOME=$(R RHOME)
```

## Verify Installation

```bash
python -c "import nodis; print(nodis.__version__)"
```

You should see `0.1.0` (or the current release version).

## DREAM5 and SERGIO Data

NODIS loaders for DREAM5 and SERGIO benchmark datasets expect data to be
manually downloaded and placed in a `data/` directory relative to your working
directory. These datasets are not bundled with the package due to size and
licensing constraints.

- **DREAM5**: download from https://www.synapse.org/#!Synapse:syn3130840
- **SERGIO**: download from https://github.com/PayamDiba/SERGIO

The loaders (`nodis.benchmark.runner`) accept an explicit `data_dir` argument if
you prefer to store data elsewhere.
