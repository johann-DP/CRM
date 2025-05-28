# CRM Project

This repository contains scripts for various phases of data analysis.

## Preparing the environment

Set up a dedicated Python environment before running the code. If an
``environment.yaml`` file is available you can create it with::

    conda env create -f environment.yaml

Otherwise install the pinned dependencies from ``requirements.txt``::

    python -m pip install -r requirements.txt

Using a clean environment avoids binary incompatibilities between NumPy,
pandas and the plotting libraries.

## Running `phase4.py`

The main analysis pipeline lives in `phase4.py`. The script requires a
configuration file in YAML (or JSON) format. A template is provided in
`config.yaml`. Copy or modify it to suit your dataset and run:

```bash
python phase4.py --config config.yaml
```

To analyse several dataset versions concurrently, list them after the
``--datasets`` option. Results for each dataset are written to a subdirectory
of ``output_dir``.

```bash
python phase4.py --config config.yaml --datasets raw cleaned_1 cleaned_3_multi cleaned_3_univ
```

Use ``--dataset-jobs`` to control how many datasets run in parallel. The
threads defined by ``n_jobs`` in the configuration are divided between the
dataset workers. For example, on an 8‑core machine:

```bash
python phase4.py --config config.yaml --datasets raw cleaned_1 cleaned_3_multi cleaned_3_univ --dataset-jobs 4
```


Set `optimize_params: true` in the configuration to automatically tune the main
hyperparameters of each dimensionality reduction method (number of components
for FAMD/MFA/PCAmix, neighbors and distance for UMAP, perplexity for t-SNE).
When disabled, the script uses the provided values or sensible defaults.
You can control parallel execution with the `n_jobs` option in `config.yaml`. Set it to the number of worker threads to run independent methods concurrently.

Make sure the dependencies listed in `requirements.txt` are installed. The
file pins specific versions of NumPy, pandas and other libraries to avoid
binary incompatibilities. Create a fresh virtual environment and install the
packages with:

```bash
python -m pip install -r requirements.txt
```
In a Codex environment this step is automated: the `setup.sh` script in the
repository runs the same command before network access is disabled.

If you encounter an error similar to::

    ValueError: numpy.dtype size changed, may indicate binary incompatibility

or::

    ImportError: Detected an incompatible combination of pandas and NumPy

it means the currently installed versions of NumPy and pandas do not match the
pinned versions. Reinstall the dependencies in a fresh virtual environment using
the command above (``python -m pip install -r requirements.txt``) or force a
reinstall with:

```bash
python -m pip install --force-reinstall -r requirements.txt
```

The package `umap-learn` is required for the UMAP functionality. `phate` and
`pacmap` are optional; install them if you want to run the corresponding
analyses. UMAP accepts several parameters in `config.yaml`, including
`n_neighbors`, `min_dist` and the distance `metric` (default `euclidean`).

### UMAP warnings

UMAP may warn when a seed is given while using several threads. The default
configuration avoids setting a seed so that all CPU cores can be used.

## FAMD scripts removed

The standalone scripts `phase4_famd.py` and `phase4_famd_simple.py` were
removed on 27/05/2025. Their capabilities are now integrated into
`phase4.py` and the associated tuning utilities. Use `phase4.py` with a
configuration file to run FAMD and optionally optimise the number of components.

## Fine tuning MFA

The script `fine_tune_mfa.py` automates a small grid search over the number of
components and optional group weights for a Multiple Factor Analysis. Provide a
YAML configuration describing the groups and ranges:

```yaml
input_file: path/to/data.xlsx
output_dir: phase4_output/fine_tuning_mfa
group_defs:
  Financier: ["Total recette actualisé", "Budget client estimé"]
  Temporalité: ["duree_projet_jours", "taux_realisation"]
mfa_params:
  min_components: 2
  max_components: 6
  weights:
    - null
    - {Financier: 1.5, Temporalité: 1.0}
```

Run it with:

```bash
python fine_tune_mfa.py --config config_mfa.yaml
```

The script exports metrics for each configuration and saves the best model (by
silhouette and Calinski–Harabasz indices) in the configured output directory.

## Consolidated PDF report

The function `export_report_to_pdf` combines the main figures and tables from
phase 4 into a single PDF file. It uses the optional `fpdf` package when
available for better layout. Install it with:

```bash
python -m pip install fpdf
```

If `fpdf` is not installed, the code falls back to Matplotlib's `PdfPages`,
producing a simpler layout.

