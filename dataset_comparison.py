# -*- coding: utf-8 -*-
"""Dataset comparison utilities for CRM analyses.

This module implements a `compare_datasets_versions` function applying the
complete dimensionality reduction pipeline on multiple dataset versions. It
re-uses the standalone helper functions provided in this repository (data
preparation, variable selection, factor methods, non-linear methods and
metrics evaluation) and does **not** depend on legacy scripts such as
``phase4v2.py`` or ``fine_tune_*``.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

import pandas as pd

from data_preparation import prepare_data
from variable_selection import select_variables
from factor_methods import run_pca, run_mca, run_famd, run_mfa
from nonlinear_methods import run_all_nonlinear
from evaluate_methods import evaluate_methods
from visualization import generate_figures

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Missing value handling (copied from phase4v2.py)
# ---------------------------------------------------------------------------

def handle_missing_values(df: pd.DataFrame, quant_vars: List[str], qual_vars: List[str]) -> pd.DataFrame:
    """Impute and drop remaining NA values if needed.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame to process.
    quant_vars : list of str
        Names of quantitative variables.
    qual_vars : list of str
        Names of qualitative variables.

    Returns
    -------
    pandas.DataFrame
        DataFrame with missing values handled.
    """
    logger = logging.getLogger(__name__)
    na_count = int(df.isna().sum().sum())
    if na_count > 0:
        logger.info("Imputation des %d valeurs manquantes restantes", na_count)
        if quant_vars:
            df[quant_vars] = df[quant_vars].fillna(df[quant_vars].median())
        for col in qual_vars:
            if df[col].dtype.name == "category" and "Non renseigné" not in df[col].cat.categories:
                df[col] = df[col].cat.add_categories("Non renseigné")
            df[col] = df[col].fillna("Non renseigné").astype("category")
        remaining = int(df.isna().sum().sum())
        if remaining > 0:
            logger.warning(
                "%d NA subsistent après imputation → suppression des lignes concernées",
                remaining,
            )
            df.dropna(inplace=True)
    else:
        logger.info("Aucune valeur manquante détectée après sanity_check")

    if df.isna().any().any():
        logger.error("Des NA demeurent dans df après traitement")
    else:
        logger.info("DataFrame sans NA prêt pour FAMD")
    return df


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compare_datasets_versions(
    datasets: Dict[str, pd.DataFrame], *, exclude_lost: bool = True, min_modalite_freq: int = 5
) -> Dict[str, Any]:
    """Compare dimensionality reduction results between dataset versions.

    Parameters
    ----------
    datasets : dict
        Mapping of version name to raw ``DataFrame``.
    exclude_lost : bool, default ``True``
        Whether to remove lost/cancelled opportunities during preparation.
    min_modalite_freq : int, default ``5``
        Frequency threshold passed to :func:`variable_selection.select_variables`.

    Returns
    -------
    dict
        Dictionary with two keys:
        ``"metrics"`` containing the concatenated metrics table and
        ``"details"`` mapping each version name to its individual results
        (metrics, figures and intermediate objects).
    """
    if not isinstance(datasets, dict):
        raise TypeError("datasets must be a dictionary")

    results_by_version: Dict[str, Any] = {}
    metrics_frames: List[pd.DataFrame] = []

    for name, df in datasets.items():
        logger.info("Processing dataset version '%s'", name)
        df_prep = prepare_data(df, exclude_lost=exclude_lost)
        df_active, quant_vars, qual_vars = select_variables(
            df_prep, min_modalite_freq=min_modalite_freq
        )
        df_active = handle_missing_values(df_active, quant_vars, qual_vars)

        # Factorial methods
        factor_results: Dict[str, Any] = {}
        if quant_vars:
            factor_results["pca"] = run_pca(df_active, quant_vars, optimize=True)
        if qual_vars:
            factor_results["mca"] = run_mca(df_active, qual_vars, optimize=True)
        if quant_vars and qual_vars:
            try:
                factor_results["famd"] = run_famd(
                    df_active, quant_vars, qual_vars, optimize=True
                )
            except ValueError as exc:
                logger.warning("FAMD skipped: %s", exc)
        groups = []
        if quant_vars:
            groups.append(quant_vars)
        if qual_vars:
            groups.append(qual_vars)
        if len(groups) > 1:
            factor_results["mfa"] = run_mfa(df_active, groups, optimize=True)

        # Non-linear methods
        nonlin_results = run_all_nonlinear(df_active)

        # Metrics and figures
        cleaned_nonlin = {
            k: v
            for k, v in nonlin_results.items()
            if "embeddings" in v and isinstance(v["embeddings"], pd.DataFrame) and not v["embeddings"].empty
        }
        all_results = {**factor_results, **cleaned_nonlin}
        n_clusters = 3 if len(df_active) > 3 else 2
        metrics = evaluate_methods(
            all_results, df_active, quant_vars, qual_vars, n_clusters=n_clusters
        )
        metrics["dataset_version"] = name
        try:
            figures = generate_figures(
                factor_results, nonlin_results, df_active, quant_vars, qual_vars
            )
        except Exception as exc:  # pragma: no cover - visualization failure
            logger.warning("Figure generation failed: %s", exc)
            figures = {}

        results_by_version[name] = {
            "metrics": metrics,
            "figures": figures,
            "factor_results": factor_results,
            "nonlinear_results": nonlin_results,
            "quant_vars": quant_vars,
            "qual_vars": qual_vars,
            "df_active": df_active,
        }
        metrics_frames.append(metrics)

    combined = pd.concat(metrics_frames).reset_index().rename(columns={"index": "method"})
    return {"metrics": combined, "details": results_by_version}


if __name__ == "__main__":  # pragma: no cover - manual testing helper
    import pprint

    logging.basicConfig(level=logging.INFO)
    # Example usage with dummy data
    df = pd.DataFrame({
        "Code": [1, 2, 3],
        "Date de début actualisée": ["2024-01-01", "2024-01-02", "2024-01-03"],
        "Date de fin réelle": ["2024-01-05", "2024-01-06", "2024-01-07"],
        "Total recette réalisé": [1000, 2000, 1500],
        "Budget client estimé": [1100, 2100, 1600],
        "Charge prévisionnelle projet": [800, 1800, 1300],
        "Statut commercial": ["Gagné", "Perdu", "Gagné"],
        "Type opportunité": ["T1", "T2", "T1"],
    })
    datasets = {"v1": df, "v2": df.drop(1)}
    out = compare_datasets_versions(datasets)
    pprint.pprint(out["metrics"].head())
