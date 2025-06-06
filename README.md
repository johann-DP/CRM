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

## Utilisation du module `pred_aggregated_amount`

### Installation et dépendances

Installez les bibliothèques nécessaires via le fichier `requirements.txt` :

```bash
python -m pip install -r requirements.txt
```

Les dépendances majeures sont `pandas`, `numpy`, `scikit-learn`, `xgboost`,
`catboost`, `prophet`, `tensorflow`, `statsforecast` et `matplotlib`.

### Structure du dossier

```
pred_aggregated_amount/
├── aggregate_revenue.py  # agrégation mensuelle, trimestrielle, annuelle
├── preprocess_dates.py   # correction des dates erronées
├── preprocess_timeseries.py  # nettoyage des séries agrégées
├── train_xgboost.py      # entraînement du modèle XGBoost
├── catboost_forecast.py  # prévisions CatBoost
├── train_arima.py        # modèles ARIMA via statsforecast
├── lstm_forecast.py      # réseau de neurones LSTM
├── prophet_models.py     # entraînement Prophet
├── evaluate_models.py    # évaluation rolling des modèles
├── future_forecast.py    # génération de prévisions futures
├── make_plots.py         # figures illustratives
└── run_all.py            # pipeline complet
```

Les résultats sont écrits dans un `output_dir/` contenant trois sous-dossiers :
`data/`, `models/` et `report/`.

### Utilisation pas à pas

1. **Préparation des données**

```bash
python data_preparation.py --input_dir path/to/raw_data --output_dir output_dir
```

2. **Entraînement des modèles**

```bash
python train_models.py --data_dir output_dir/data --models_dir output_dir/models
```

3. **Génération du rapport**

```bash
python generate_report.py --models_dir output_dir/models \
    --data_dir output_dir/data --report_dir output_dir/report
```

Chaque commande accepte les arguments `--input_dir`, `--output_dir`,
`--data_dir`, `--models_dir` ou `--report_dir` selon le script. Les valeurs par
défaut utilisent `output_dir/` dans le dossier courant.

### Résultats et interprétation

Le dossier `output_dir/report/` contient :

* `rapport_performance.txt` – tableau des métriques (MAE, RMSE, MAPE).
* `previsions_mensuelles.png` – comparaison entre CA réel et prédit sur 12 mois.
* `erreurs_par_modele.png` – barres d'erreur par algorithme.

D'autres figures trimestrielles ou annuelles peuvent également être produites.

### Anticipation des bugs et bonnes pratiques

* Vérifiez l'existence des fichiers d'entrée avant exécution des scripts.
* Exécutez dans un environnement virtuel avec les versions recommandées.
* Nettoyez le dossier `output_dir/` entre deux lancements pour éviter les
  collisions de fichiers.
* Surveillez les warnings générés par pandas, CatBoost et XGBoost.
* Activez le mode verbeux (`--verbose` ou `--debug`) pour obtenir des logs
  détaillés en cas de problème.

