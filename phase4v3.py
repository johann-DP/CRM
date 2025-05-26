#!/usr/bin/env python3
"""Bloc 1 – Chargement et structuration des jeux de données.

This module defines a helper ``load_datasets`` used in phase 4.
It loads the raw Excel/CSV export along with the cleaned datasets
produced during phases 1–3, applying minimal type conversions and
column harmonisation.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Any

from sklearn.preprocessing import StandardScaler

import pandas as pd


def _read_generic(path: Path) -> pd.DataFrame:
    """Read a CSV or Excel file depending on its suffix."""
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    return pd.read_excel(path, engine="openpyxl")


def _coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    """Attempt basic type conversions (dates and numerics)."""
    df = df.copy()
    for col in df.columns:
        low = col.lower()
        if "date" in low:
            df[col] = pd.to_datetime(df[col], errors="ignore")
        elif df[col].dtype == object:
            cleaned = pd.to_numeric(df[col].str.replace(",", ".", regex=False),
                                    errors="ignore")
            if cleaned.notna().any():
                df[col] = cleaned
    return df


def _apply_dictionary(df: pd.DataFrame, mapping: Dict[str, str]) -> pd.DataFrame:
    if not mapping:
        return df
    rename = {c: mapping.get(c, c) for c in df.columns}
    return df.rename(columns=rename)


def load_datasets(config: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
    """Load raw and cleaned datasets from various phases.

    Parameters
    ----------
    config : dict
        Configuration dictionary containing at least ``input_file``.
        Optional keys ``phase1_file``, ``phase2_file`` and ``phase3_file`` can
        point to the cleaned datasets. ``data_dictionary`` may specify an
        Excel file mapping original to harmonised column names.

    Returns
    -------
    dict
        ``{"raw": df_raw, "phase1": df_p1, ...}`` with only available phases.
    """
    logger = logging.getLogger(__name__)

    datasets: Dict[str, pd.DataFrame] = {}

    # --- dictionnaire de données optionnel ---------------------------------
    mapping: Dict[str, str] = {}
    ddict = config.get("data_dictionary")
    if ddict:
        try:
            ddf = pd.read_excel(ddict)
            cols = {c.lower(): c for c in ddf.columns}
            old_col = next((cols[c] for c in ["original", "ancien", "variable"] if c in cols), None)
            new_col = next((cols[c] for c in ["new", "standard", "nouveau"] if c in cols), None)
            if old_col and new_col:
                mapping = dict(zip(ddf[old_col].astype(str), ddf[new_col].astype(str)))
        except Exception as exc:
            logger.warning("Impossible de lire le dictionnaire de données %s: %s", ddict, exc)

    # --- raw dataset -------------------------------------------------------
    raw_path = Path(config["input_file"])
    logger.info("Chargement du fichier brut: %s", raw_path)
    df_raw = _coerce_types(_read_generic(raw_path))
    df_raw = _apply_dictionary(df_raw, mapping)
    datasets["raw"] = df_raw

    # --- cleaned datasets --------------------------------------------------
    phase_paths = {
        "phase1": config.get("phase1_file"),
        "phase2": config.get("phase2_file"),
        "phase3": config.get("phase3_file"),
    }
    for name, path_str in phase_paths.items():
        if not path_str:
            continue
        path = Path(path_str)
        logger.info("Chargement %s : %s", name, path)
        df = _coerce_types(_read_generic(path))
        df = _apply_dictionary(df, mapping)
        datasets[name] = df

    # --- simple cohérence colonnes ---------------------------------------
    ref_cols = set(df_raw.columns)
    for name, df in datasets.items():
        miss = ref_cols - set(df.columns)
        extra = set(df.columns) - ref_cols
        if miss or extra:
            logger.warning("Colonnes incohérentes pour %s (manquantes=%s, en_trop=%s)",
                           name, sorted(miss), sorted(extra))
    return datasets



def prepare_data(df: pd.DataFrame, exclude_lost: bool = True) -> pd.DataFrame:
    """Clean and standardise a CRM dataset.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame loaded via :func:`load_datasets`.
    exclude_lost : bool, optional
        Whether to drop rows marked as lost or cancelled, by default ``True``.

    Returns
    -------
    pd.DataFrame
        Cleaned and standardised DataFrame ready for analysis.
    """
    logger = logging.getLogger(__name__)
    df_clean = df.copy()

    # --- remove flagged outliers -----------------------------------------
    flag_cols = [c for c in df_clean.columns if c.lower().startswith("flag_")]
    for col in flag_cols:
        try:
            mask = df_clean[col].astype(bool)
        except Exception:
            continue
        if mask.any():
            logger.info("%s lignes exclues via %s", int(mask.sum()), col)
            df_clean = df_clean.loc[~mask]
        df_clean.drop(columns=col, inplace=True)

    # --- date parsing and out-of-range filtering -------------------------
    min_date = pd.Timestamp("1990-01-01")
    max_date = pd.Timestamp("2050-12-31")
    for col in df_clean.columns:
        if "date" in col.lower():
            df_clean[col] = pd.to_datetime(df_clean[col], errors="coerce")
            mask = df_clean[col].lt(min_date) | df_clean[col].gt(max_date)
            if mask.any():
                logger.info("%s dates hors plage dans %s", int(mask.sum()), col)
                df_clean.loc[mask, col] = pd.NaT

    # --- basic missing value handling ------------------------------------
    num_cols = df_clean.select_dtypes(include=[float, int]).columns.tolist()
    for col in num_cols:
        median = df_clean[col].median()
        df_clean[col].fillna(median, inplace=True)
    for col in df_clean.select_dtypes(include="object"):
        df_clean[col] = df_clean[col].fillna("Non renseigné").astype("category")

    # --- optional exclusion of lost deals --------------------------------
    if exclude_lost:
        if "Statut commercial" in df_clean.columns:
            lost_values = {"Perdu", "Annulé", "Abandonné"}
            mask = df_clean["Statut commercial"].isin(lost_values)
            if mask.any():
                logger.info("%s lignes perdues/annulées retirées", int(mask.sum()))
                df_clean = df_clean.loc[~mask]
        elif "Motif_non_conformité" in df_clean.columns:
            mask = df_clean["Motif_non_conformité"].notna()
            if mask.any():
                logger.info("%s lignes retirées via Motif_non_conformité", int(mask.sum()))
                df_clean = df_clean.loc[~mask]

    # --- numeric standardisation ----------------------------------------
    scaler = StandardScaler()
    if num_cols:
        df_clean[num_cols] = scaler.fit_transform(df_clean[num_cols])

    return df_clean

