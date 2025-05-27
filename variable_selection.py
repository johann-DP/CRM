"""Utility functions for selecting active variables for CRM analyses."""

from __future__ import annotations

import logging
from typing import List, Tuple

import pandas as pd
from sklearn.preprocessing import StandardScaler


def select_variables(df: pd.DataFrame, min_modalite_freq: int = 5) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """Return the dataframe restricted to relevant variables.

    Parameters
    ----------
    df:
        Cleaned dataframe coming from ``prepare_data``.
    min_modalite_freq:
        Minimum frequency below which categorical levels are grouped in
        ``"Autre"``.

    Returns
    -------
    tuple
        ``(df_active, quantitative_vars, qualitative_vars)`` where
        ``df_active`` contains the scaled numeric columns and categorical
        columns cast to ``category``.
    """
    logger = logging.getLogger(__name__)

    candidate_quant = [
        "Total recette actualisé",
        "Total recette réalisé",
        "Total recette produit",
        "Budget client estimé",
        "duree_projet_jours",
        "taux_realisation",
        "marge_estimee",
    ]
    candidate_qual = [
        "Statut commercial",
        "Statut production",
        "Type opportunité",
        "Catégorie",
        "Sous-catégorie",
        "Pilier",
        "Entité opérationnelle",
        "Présence partenaire",
    ]

    exclude = {"Code", "Client", "Contact principal", "Titre"}

    quant_vars: List[str] = []
    for col in candidate_quant:
        if col not in df.columns or col in exclude:
            continue
        series = pd.to_numeric(df[col], errors="coerce")
        if series.var(skipna=True) == 0 or series.isna().all():
            logger.warning("Variable quantitative '%s' exclue", col)
            continue
        df[col] = series.astype(float)
        quant_vars.append(col)

    qual_vars: List[str] = []
    for col in candidate_qual:
        if col not in df.columns or col in exclude:
            continue
        series = df[col].astype("category")
        counts = series.value_counts(dropna=False)
        rares = counts[counts < min_modalite_freq].index
        if len(rares) > 0:
            logger.info("%d modalités rares dans '%s' regroupées en 'Autre'", len(rares), col)
            if "Autre" not in series.cat.categories:
                series = series.cat.add_categories(["Autre"])
            series = series.apply(lambda x: "Autre" if x in rares else x).astype("category")
        if series.nunique(dropna=False) <= 1:
            logger.warning("Variable qualitative '%s' exclue", col)
            continue
        df[col] = series
        qual_vars.append(col)

    df_active = df[quant_vars + qual_vars].copy()

    if quant_vars:
        scaler = StandardScaler()
        df_active[quant_vars] = scaler.fit_transform(df_active[quant_vars])

    for col in qual_vars:
        df_active[col] = df_active[col].astype("category")

    logger.info("DataFrame actif avec %d variables", len(df_active.columns))
    return df_active, quant_vars, qual_vars

