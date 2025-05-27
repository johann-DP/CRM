"""Data preparation utilities for Phase 4.

This module provides a self-contained ``prepare_data`` function used to
clean and standardise the CRM datasets before dimensionality reduction. It
re-implements the relevant logic previously found in ``phase4v2.py`` and
the fine tuning scripts so that these legacy files can be removed.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def prepare_data(df: pd.DataFrame, exclude_lost: bool = True) -> pd.DataFrame:
    """Return a cleaned and standardised copy of ``df``.

    Parameters
    ----------
    df : pd.DataFrame
        Raw or already partially cleaned CRM dataset.
    exclude_lost : bool, default ``True``
        If ``True``, rows marked as lost or cancelled opportunities are
        removed from the returned DataFrame.

    Returns
    -------
    pd.DataFrame
        The cleaned DataFrame with numerical columns scaled to zero mean and
        unit variance.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")

    df_clean = df.copy()

    # ------------------------------------------------------------------
    # 1) Dates: parse and drop obvious out-of-range values
    # ------------------------------------------------------------------
    date_cols = [c for c in df_clean.columns if "date" in c.lower()]
    min_date = pd.Timestamp("1990-01-01")
    max_date = pd.Timestamp("2050-12-31")
    for col in date_cols:
        df_clean[col] = pd.to_datetime(df_clean[col], errors="coerce")
        mask = df_clean[col].lt(min_date) | df_clean[col].gt(max_date)
        if mask.any():
            logger.warning("%d invalid dates replaced by NaT in '%s'", mask.sum(), col)
            df_clean.loc[mask, col] = pd.NaT

    # ------------------------------------------------------------------
    # 2) Monetary amounts: numeric conversion and negative values
    # ------------------------------------------------------------------
    amount_cols = [
        "Total recette actualisé",
        "Total recette réalisé",
        "Total recette produit",
        "Budget client estimé",
    ]
    for col in amount_cols:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")
            neg = df_clean[col] < 0
            if neg.any():
                logger.warning("%d negative values set to NaN in '%s'", neg.sum(), col)
                df_clean.loc[neg, col] = np.nan

    # ------------------------------------------------------------------
    # 3) Remove duplicate opportunity identifiers if present
    # ------------------------------------------------------------------
    if "Code" in df_clean.columns:
        before = len(df_clean)
        df_clean = df_clean.drop_duplicates(subset=["Code"])
        if len(df_clean) != before:
            logger.info("%d duplicated rows dropped", before - len(df_clean))

    # ------------------------------------------------------------------
    # 4) Derived indicators used in later analyses
    # ------------------------------------------------------------------
    if {"Date de début actualisée", "Date de fin réelle"} <= set(df_clean.columns):
        df_clean["duree_projet_jours"] = (
            df_clean["Date de fin réelle"] - df_clean["Date de début actualisée"]
        ).dt.days
    if {"Total recette réalisé", "Budget client estimé"} <= set(df_clean.columns):
        denom = df_clean["Budget client estimé"].replace(0, np.nan)
        df_clean["taux_realisation"] = df_clean["Total recette réalisé"] / denom
        df_clean["taux_realisation"] = df_clean["taux_realisation"].replace([np.inf, -np.inf], np.nan)
    if {"Total recette réalisé", "Charge prévisionnelle projet"} <= set(df_clean.columns):
        df_clean["marge_estimee"] = df_clean["Total recette réalisé"] - df_clean["Charge prévisionnelle projet"]

    # ------------------------------------------------------------------
    # 5) Simple missing value handling
    # ------------------------------------------------------------------
    impute_cols: list[str] = [c for c in amount_cols if c in df_clean.columns]
    if "taux_realisation" in df_clean.columns:
        impute_cols.append("taux_realisation")
    for col in impute_cols:
        median = df_clean[col].median()
        df_clean[col] = df_clean[col].fillna(median)
    for col in df_clean.select_dtypes(include="object"):
        df_clean[col] = df_clean[col].fillna("Non renseigné").astype("category")

    # ------------------------------------------------------------------
    # 6) Filter multivariate outliers flagged during phase 3
    # ------------------------------------------------------------------
    if "flag_multivariate" in df_clean.columns:
        out = df_clean["flag_multivariate"].astype(bool)
        if out.any():
            logger.info("%d outliers removed via 'flag_multivariate'", int(out.sum()))
            df_clean = df_clean.loc[~out]

    # ------------------------------------------------------------------
    # 7) Exclude lost or cancelled opportunities if requested
    # ------------------------------------------------------------------
    if exclude_lost and "Statut commercial" in df_clean.columns:
        lost_mask = df_clean["Statut commercial"].astype(str).str.contains(
            "perdu|annul|aband", case=False, na=False
        )
        if lost_mask.any():
            logger.info("%d lost opportunities removed", int(lost_mask.sum()))
            df_clean = df_clean.loc[~lost_mask]
    if exclude_lost and "Motif_non_conformité" in df_clean.columns:
        mask_nc = df_clean["Motif_non_conformité"].notna() & df_clean["Motif_non_conformité"].astype(str).str.strip().ne("")
        if mask_nc.any():
            logger.info("%d non conformities removed", int(mask_nc.sum()))
            df_clean = df_clean.loc[~mask_nc]

    # ------------------------------------------------------------------
    # 8) Standardise numerical columns
    # ------------------------------------------------------------------
    num_cols = df_clean.select_dtypes(include=[np.number]).columns
    num_cols = [c for c in num_cols if c != "Code"]
    if len(num_cols) > 0:
        scaler = StandardScaler()
        df_clean[num_cols] = scaler.fit_transform(df_clean[num_cols])

    return df_clean
