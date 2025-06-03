# CRM Project

This repository gathers the Python scripts used to analyse the Digora CRM data.
It is organised in four successive phases that build upon each other.

## Setup

Create a dedicated environment and install the dependencies listed in
`requirements.txt` (or use `environment.yaml` if available). You can also run
`bash setup.sh` which automates the installation and installs the project in
editable mode.

```bash
conda env create -f environment.yaml  # or
python -m pip install -r requirements.txt
bash setup.sh
```

## Workflow overview

### Phase 1 – Data audit
A manual exploration and clean-up step (not included in this repository) that
prepares the CRM export for the following scripts.

### Phase 2 – Global exploration
`phase2.py` computes high-level indicators such as missing values, data
dictionary, business variables, KPIs and quarterly trends. It automatically
exports all charts and a consolidated PDF report.

### Phase 3 – Anomaly detection
`phase3.py` inspects the raw CRM extract in depth. It checks rare modalities,
date ranges and numeric distributions, performs outlier detection and exports
cleaned datasets along with summaries of all anomalies.

### Phase 4 – Advanced analyses
The main pipeline is implemented in the `phase4` package. Provide a configuration
file in YAML (see `config.yaml`) and run:

```bash
python -m phase4 --config config.yaml
```

Multiple dataset versions can be processed by listing them after `--datasets`.
Setting `output_pdf` in the configuration combines all figures into a single PDF
following the [report page order](#report-page-order). The `optimize_params`
option automatically tunes key hyperparameters of each method.

## Additional utilities

* `phase4/generate_report.py` – merge exported figures into a single PDF.
* `fine_tune_mfa.py` – small grid search over MFA components and weights.
* `phase4bis/export_pca_coordinates.py` – save PCA scores for each record.
* `phase4bis/compare_pca_umap.py` – side-by-side PCA vs UMAP scatter plot.
* `phase4bis/simple_pca.py` – quick PCA on a CSV export.
* `phase4bis/run_all_since_commit.py` – execute every helper script.
* `pred_aggregated_amount/run_all.py` – evaluate forecasting models on cleaned data.

Install the optional `fpdf` package if you want nicer PDF layouts; otherwise
Matplotlib's `PdfPages` is used.

### Report page order
For each dataset and factor method the combined PDF includes:
1. **Raw scatter plots** – 2D/3D projections without clustering.
2. **Clustered scatter grid** – K-means, Agglomerative, Spectral and Gaussian
   Mixture partitions.
3. **Analysis summary** – variable contributions, correlation circle, scree plot
   and silhouette curves.

Additional pages such as heatmaps or segment summaries are appended after the
per-method sections.

