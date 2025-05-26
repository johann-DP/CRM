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

