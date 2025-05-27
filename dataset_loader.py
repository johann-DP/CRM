# -*- coding: utf-8 -*-
"""Utilities for loading CRM datasets.

This module contains a :func:`load_datasets` function extracted from
``phase4v3.py`` so that the old monolithic script can be removed.
The API is kept identical for backward compatibility.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import pandas as pd

CONFIG: Dict[str, Any] = {}


def _read_dataset(path: Path) -> pd.DataFrame:
    """Read a CSV or Excel file with basic type handling."""
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        df = pd.read_excel(path)
    df.columns = [str(c).strip() for c in df.columns]
    for col in df.columns:
        if "date" in col.lower():
            df[col] = pd.to_datetime(df[col], errors="coerce")
    for col in df.select_dtypes(include="object"):
        if any(k in col.lower() for k in ["montant", "recette", "budget", "total"]):
            series = df[col].astype(str).str.replace("\xa0", "", regex=False)
            series = series.str.replace(" ", "", regex=False)
            series = series.str.replace(",", ".", regex=False)
            df[col] = pd.to_numeric(series, errors="coerce")
    return df


def _load_data_dictionary(path: Optional[Path]) -> Dict[str, str]:
    """Load column rename mapping from an Excel data dictionary."""
    if path is None or not path.exists():
        return {}
    try:
        df = pd.read_excel(path)
    except Exception as exc:  # pragma: no cover - optional dependency
        logging.getLogger(__name__).warning("Could not read data dictionary: %s", exc)
        return {}
    cols = {c.lower(): c for c in df.columns}
    src = next((cols[c] for c in ["original", "colonne", "column"] if c in cols), None)
    dst = next((cols[c] for c in ["clean", "standard", "renamed"] if c in cols), None)
    if src is None or dst is None:
        return {}
    mapping = dict(zip(df[src].astype(str), df[dst].astype(str)))
    return mapping


def load_datasets(config: Optional[Mapping[str, Any]] = None) -> Dict[str, pd.DataFrame]:
    """Load raw and processed datasets according to ``config``."""
    logger = logging.getLogger(__name__)

    cfg = CONFIG if config is None else config
    if not isinstance(cfg, Mapping):
        raise TypeError("config must be a mapping or None")
    if "input_file" not in cfg:
        raise ValueError("'input_file' missing from config")

    mapping = _load_data_dictionary(Path(cfg.get("data_dictionary", "")))

    def _apply_mapping(df: pd.DataFrame) -> pd.DataFrame:
        if mapping:
            df = df.rename(columns={c: mapping.get(c, c) for c in df.columns})
        return df

    datasets: Dict[str, pd.DataFrame] = {}
    raw_path = Path(cfg["input_file"])
    datasets["raw"] = _read_dataset(raw_path)
    logger.info(
        "Raw dataset loaded from %s [%d rows, %d cols]",
        raw_path,
        datasets["raw"].shape[0],
        datasets["raw"].shape[1],
    )

    datasets["raw"] = _apply_mapping(datasets["raw"])

    for key, cfg_key in [
        ("phase1", "phase1_file"),
        ("phase2", "phase2_file"),
        ("phase3", "phase3_file"),
        ("phase3_multi", "phase3_multi_file"),
        ("phase3_univ", "phase3_univ_file"),
    ]:
        path_str = cfg.get(cfg_key)
        if not path_str:
            continue
        path = Path(path_str)
        if not path.exists():
            logger.warning("Dataset %s not found: %s", key, path)
            continue
        df = _read_dataset(path)
        datasets[key] = _apply_mapping(df)
        logger.info(
            "Loaded %s dataset from %s [%d rows, %d cols]",
            key,
            path,
            df.shape[0],
            df.shape[1],
        )

    ref_cols = set(datasets["raw"].columns)
    for name, df in list(datasets.items()):
        extra = set(df.columns) - ref_cols
        if extra:
            logger.debug("%s has %d additional columns", name, len(extra))
    return datasets
