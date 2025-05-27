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

    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")

    df = df.copy()

    # Columns explicitly ignored based on the data dictionary / prior phases
    exclude = {
        "Code",
        "ID",
        "Id",
        "Identifiant",
        "Client",
        "Contact principal",
        "Titre",
        "texte",
        "commentaire",
        "Commentaires",
    }

    # Drop constant columns and those in the exclusion list
    n_unique = df.nunique(dropna=False)
    constant_cols = n_unique[n_unique <= 1].index.tolist()
    drop_cols = set(constant_cols) | {c for c in df.columns if c in exclude}

    # Remove datetime columns
    drop_cols.update([c for c in df.select_dtypes(include="datetime").columns])

    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    quantitative_vars: List[str] = []
    qualitative_vars: List[str] = []

    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            series = pd.to_numeric(df[col], errors="coerce")
            if series.var(skipna=True) == 0 or series.isna().all():
                logger.warning("Variable quantitative '%s' exclue", col)
                continue
            df[col] = series.astype(float)
            quantitative_vars.append(col)
        else:
            series = df[col].astype("category")
            unique_ratio = series.nunique(dropna=False) / len(series)
            if unique_ratio > 0.8:
                logger.warning("Variable textuelle '%s' exclue", col)
                continue
            counts = series.value_counts(dropna=False)
            if len(series) < min_modalite_freq:
                threshold = 0  # no grouping on tiny samples
            else:
                threshold = min_modalite_freq
            rares = counts[counts < threshold].index
            if len(rares) > 0:
                logger.info(
                    "%d modalités rares dans '%s' regroupées en 'Autre'",
                    len(rares),
                    col,
                )
                if "Autre" not in series.cat.categories:
                    series = series.cat.add_categories(["Autre"])
                series = series.apply(lambda x: "Autre" if x in rares else x).astype("category")
            if series.nunique(dropna=False) <= 1:
                logger.warning("Variable qualitative '%s' exclue", col)
                continue
            df[col] = series
            qualitative_vars.append(col)

    df_active = df[quantitative_vars + qualitative_vars].copy()

    if quantitative_vars:
        scaler = StandardScaler()
        df_active[quantitative_vars] = scaler.fit_transform(df_active[quantitative_vars])

    for col in qualitative_vars:
        df_active[col] = df_active[col].astype("category")

    logger.info("DataFrame actif avec %d variables", len(df_active.columns))
    return df_active, quantitative_vars, qualitative_vars

