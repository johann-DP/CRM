# CRM Project

This repository contains scripts for the four successive phases of the
Digora CRM analysis.  Phase 1 was a manual audit and data preparation
step documented in the project report, while phases 2 to 4 are
implemented as Python scripts in this repository.

## Preparing the environment

Set up a dedicated Python environment before running the code. If an
``environment.yaml`` file is available you can create it with::

    conda env create -f environment.yaml

Otherwise install the pinned dependencies from ``requirements.txt``::

    python -m pip install -r requirements.txt

Using a clean environment avoids binary incompatibilities between NumPy,
pandas and the plotting libraries.

## Phase 1 – Audit initial et préparation des données

The first phase focused on exploring the raw export of the Digora CRM.
The corresponding script is no longer in the repository, but the report
provides the following outline:

1. **Structure du jeu de données et couverture temporelle** – inventory
   of available fields and time span.
2. **Nettoyage et uniformisation des données**
   - standardisation of text fields;
   - conversion of numeric values and dates;
   - removal of empty or redundant columns;
   - consistent categories and values;
   - creation of derived variables.
3. **Qualité des données** – analysis of missing values, duplicates and
   logical inconsistencies (chronology, status vs. amounts, outliers,
   irregular updates and seasonality).
4. **Synthèse et recommandations métier** – enforce data entry controls,
   standardise field usage and train CRM users.

## Phase 2 – Analyse exploratoire globale

`phase2.py` loads the cleaned export from phase 1 and produces a series
of high-level indicators and figures:

- bucketing of rare categories and export of missing values for key
  fields;
- generation of a data dictionary and business variables (margins,
  budget variance, project delays);
- computation of global KPIs (opportunity counts, conversion rate,
  revenue, average deal size and project duration);
- quarterly trends with breakdowns by status, entity and opportunity
  type;
- pipeline visualisations, top clients and revenue per sales person;
- correlation matrix of numeric variables and conversion rate per
  opportunity type;
- automatic export of all charts and a consolidated PDF.

## Phase 3 – Analyse détaillée et détection d'anomalies

`phase3.py` performs a deeper inspection of the raw CRM extract:

- categorical overviews with detection of rare modalities and unusual
  combinations;
- date range checks and export of records outside the 1990–2050 window;
- descriptive statistics with skewness and kurtosis for numeric fields;
- univariate outlier detection (IQR method) and multivariate approaches
  (Isolation Forest and Local Outlier Factor) illustrated with PCA;
- summary of all anomalies and export of cleaned datasets.

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

When `output_pdf` is specified in the configuration, this command also
produces a consolidated PDF report following the [report page order](#report-page-order).

The optional ``--dataset-jobs`` flag controls how many worker processes run
those datasets in parallel. ``--dataset-backend`` selects the joblib backend
used for that parallelism (default ``multiprocessing``).

When the configuration sets ``output_pdf``, all generated figures are compiled
into that file via ``export_report_to_pdf`` once the analyses finish.


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


### Report page order

For each dataset and factor method, the combined PDF includes exactly three pages:
1. **Raw scatter plots** – the 2D (and 3D when available) projections without clustering.
2. **Clustered scatter grid** – a 2×2 grid comparing K-means, Agglomerative, Spectral Clustering and Gaussian Mixture partitions.
3. **Analysis summary** – a 2×2 figure gathering variable contributions, correlation circle, scree plot and silhouette curves.

Additional pages such as heatmaps or segment summaries are appended after the per-method sections.

The report now groups these figures into four main sections with dedicated title pages:

1. **Analyses Factorielles (ACP, FAMD, AFM)** – heatmaps of cos² and inertia tables.
2. **Méthodes de Projection Non-Linéaires** – t-SNE, UMAP, PaCMAP and PHATE visualisations.
3. **Analyse de Clustering** – silhouette curves and clustering quality indices.
4. **Comparaisons Croisées** – synthesis figures across methods and datasets.

## Standalone report builder

generate_phase4_report.py collects the images exported by `phase4.py` and merges them into a single PDF with the same layout as `export_report_to_pdf`.
The script accepts the optional ``--config`` and ``--datasets`` arguments to match your setup.

```bash
python generate_phase4_report.py --config config.yaml --datasets raw cleaned_1
```

When omitted, ``--config`` defaults to ``config.yaml`` and ``--datasets`` processes `raw`, `cleaned_1`, `cleaned_3_multi` and `cleaned_3_univ` in order.

## Export PCA coordinates

The script `export_pca_coordinates.py` extracts the PCA factor scores for each
row of a dataset defined in `config.yaml`. The resulting CSV lists the
identifier and the first principal components. Example usage:

```bash
python export_pca_coordinates.py --config config.yaml --dataset raw \
    --components 3 --output ACP_coordonnees_individus.csv --sep ';'
```

The `--components` option controls how many axes are saved while `--sep`
selects the CSV separator.
