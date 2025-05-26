"""Data loading and variable selection utilities."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import pandas as pd
import numpy as np


def _read_generic(path: Path) -> pd.DataFrame:
    """Read a CSV or Excel file depending on its extension."""
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    return pd.read_excel(path, engine="openpyxl")


def _coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    """Perform simple type coercions on dates and numbers."""
    df = df.copy()
    for col in df.columns:
        low = col.lower()
        if "date" in low:
            df[col] = pd.to_datetime(df[col], errors="ignore")
        elif df[col].dtype == object:
            cleaned = pd.to_numeric(df[col].str.replace(",", ".", regex=False), errors="ignore")
            if cleaned.notna().any():
                df[col] = cleaned
    return df


def _apply_dictionary(df: pd.DataFrame, mapping: Dict[str, str]) -> pd.DataFrame:
    """Rename columns using a mapping dictionary."""
    if not mapping:
        return df
    rename = {c: mapping.get(c, c) for c in df.columns}
    return df.rename(columns=rename)


def load_datasets(config: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
    """Load raw and cleaned datasets according to configuration."""
    logger = logging.getLogger(__name__)
    datasets: Dict[str, pd.DataFrame] = {}

    # optional data dictionary ------------------------------------------------
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
        except Exception as exc:  # pragma: no cover - best effort
            logger.warning("Impossible de lire le dictionnaire de données %s: %s", ddict, exc)

    # raw file ----------------------------------------------------------------
    raw_path = Path(config["input_file"])
    logger.info("Chargement du fichier brut: %s", raw_path)
    df_raw = _coerce_types(_read_generic(raw_path))
    df_raw = _apply_dictionary(df_raw, mapping)
    datasets["raw"] = df_raw

    # cleaned phases ----------------------------------------------------------
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

    # simple column consistency check ----------------------------------------
    ref_cols = set(df_raw.columns)
    for name, df in datasets.items():
        miss = ref_cols - set(df.columns)
        extra = set(df.columns) - ref_cols
        if miss or extra:
            logger.warning(
                "Colonnes incohérentes pour %s (manquantes=%s, en_trop=%s)",
                name,
                sorted(miss),
                sorted(extra),
            )

    return datasets


def select_variables(
    df: pd.DataFrame,
    *,
    data_dict: Optional[pd.DataFrame] = None,
    min_modalite_freq: int = 5,
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """Return active dataframe with lists of quantitative and qualitative vars."""
    logger = logging.getLogger(__name__)
    df = df.copy()

    # -- exclusion of irrelevant columns -------------------------------------
    exclude: set[str] = set()
    if data_dict is not None:
        cols = {c.lower(): c for c in data_dict.columns}
        name_col = next((cols[c] for c in ["variable", "column", "colonne"] if c in cols), None)
        keep_col = next((cols[c] for c in ["keep", "active", "actif"] if c in cols), None)
        if name_col and keep_col:
            excl = data_dict.loc[~data_dict[keep_col].astype(bool), name_col].astype(str)
            exclude.update(excl.tolist())

    keywords = ["id", "code", "ident", "uuid", "titre", "comment", "desc", "texte"]
    for col in df.columns:
        low = col.lower()
        if any(k in low for k in keywords):
            exclude.add(col)
        elif df[col].nunique(dropna=False) <= 1:
            exclude.add(col)
        elif df[col].isna().mean() > 0.9:
            exclude.add(col)
        elif (
            df[col].dtype == object
            and df[col].astype(str).str.len().mean() > 50
            and df[col].nunique() > 20
        ):
            exclude.add(col)

    if exclude:
        logger.info("Exclusion de %s colonnes non pertinentes", len(exclude))
        df = df.drop(columns=[c for c in exclude if c in df.columns])

    # -- split quantitative/qualitative --------------------------------------
    quant_vars = list(df.select_dtypes(include=["number"]).columns)
    qual_vars = [c for c in df.columns if c not in quant_vars]

    for col in quant_vars:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        if (df[col] > 0).all() and abs(df[col].skew()) > 3:
            df[col] = np.log10(df[col] + 1)

    final_qual: List[str] = []
    for col in qual_vars:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = df[col].dt.strftime("%Y-%m-%d")
        df[col] = df[col].astype("category")
        counts = df[col].value_counts(dropna=False)
        rares = counts[counts < min_modalite_freq].index
        if len(rares):
            df[col] = df[col].cat.add_categories("Autre")
            df[col] = df[col].where(~df[col].isin(rares), "Autre").astype("category")
        if df[col].nunique() > 1:
            final_qual.append(col)

    qual_vars = final_qual
    quant_vars = [c for c in quant_vars if df[c].var(skipna=True) not in (0, float("nan"))]

    df_active = df[quant_vars + qual_vars].copy()

    logger.info("%s variables quantitatives conservées", len(quant_vars))
    logger.info("%s variables qualitatives conservées", len(qual_vars))

    return df_active, quant_vars, qual_vars
