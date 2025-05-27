# !/usr/bin/env python3
"""Phase 4 Version 3 utilities.

Bloc 1: Chargement et structuration des jeux de donnees.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

# scripts de fine-tuning fournis
try:
    from fine_tune_famd import run_famd as tune_famd
except Exception as exc:  # pragma: no cover - optional dependency
    logging.getLogger(__name__).warning("fine_tune_famd import failed: %s", exc)
    tune_famd = None
try:
    from fine_tune_pca import run_pca as tune_pca
except Exception as exc:  # pragma: no cover - optional dependency
    logging.getLogger(__name__).warning("fine_tune_pca import failed: %s", exc)
    tune_pca = None
try:
    from fine_tuning_mca import run_mca as tune_mca
except Exception as exc:  # pragma: no cover - optional dependency
    logging.getLogger(__name__).warning("fine_tuning_mca import failed: %s", exc)
    tune_mca = None
try:
    from fine_tune_mfa import run_mfa as tune_mfa
except Exception as exc:  # pragma: no cover - optional dependency
    logging.getLogger(__name__).warning("fine_tune_mfa import failed: %s", exc)
    tune_mfa = None
try:
    from fine_tuning_umap import run_umap as tune_umap
except Exception as exc:  # pragma: no cover - optional dependency
    logging.getLogger(__name__).warning("fine_tuning_umap import failed: %s", exc)
    tune_umap = None
try:
    from phase4_fine_tune_phate import run_phate as tune_phate
except Exception as exc:  # pragma: no cover - optional dependency
    logging.getLogger(__name__).warning("phase4_fine_tune_phate import failed: %s", exc)
    tune_phate = None
try:
    from pacmap_fine_tune import run_pacmap as tune_pacmap
except Exception as exc:  # pragma: no cover - optional dependency
    logging.getLogger(__name__).warning("pacmap_fine_tune import failed: %s", exc)
    tune_pacmap = None


# ---------------------------------------------------------------------------
# Utilitaires internes
# ---------------------------------------------------------------------------

def _read_dataset(path: Path) -> pd.DataFrame:
    """Read a CSV or Excel file into a ``DataFrame`` with basic type handling."""
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        df = pd.read_excel(path)

    # Clean column names
    df.columns = [str(c).strip() for c in df.columns]

    # Parse dates
    for col in df.columns:
        if "date" in col.lower():
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # Parse numerical amounts
    for col in df.select_dtypes(include="object"):
        if any(k in col.lower() for k in ["montant", "recette", "budget", "total"]):
            series = df[col].astype(str).str.replace("\xa0", "", regex=False)
            series = series.str.replace(" ", "", regex=False)
            series = series.str.replace(",", ".", regex=False)
            df[col] = pd.to_numeric(series, errors="coerce")
    return df


def _load_data_dictionary(path: Optional[Path]) -> Dict[str, str]:
    """Load column rename mapping from a data dictionary Excel file."""
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


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_datasets(config: Dict[str, str]) -> Dict[str, pd.DataFrame]:
    """Load raw and cleaned datasets for phase 4.

    Parameters
    ----------
    config : dict
        Configuration dictionary with at least ``input_file``. Optional keys are
        ``phase1_file``, ``phase2_file``, ``phase3_file`` (or ``phase3_multi_file`` and
        ``phase3_univ_file``) and ``data_dictionary``.

    Returns
    -------
    dict
        Mapping of dataset name to ``DataFrame``. Keys include ``raw`` and the
        phases present in ``config``.
    """
    logger = logging.getLogger(__name__)
    if "input_file" not in config:
        raise ValueError("'input_file' missing from config")

    datasets: Dict[str, pd.DataFrame] = {}

    raw_path = Path(config["input_file"])
    datasets["raw"] = _read_dataset(raw_path)
    logger.info("Raw dataset loaded from %s", raw_path)

    mapping = _load_data_dictionary(Path(config.get("data_dictionary", "")))

    def _apply_mapping(df: pd.DataFrame) -> pd.DataFrame:
        if mapping:
            df = df.rename(columns={c: mapping.get(c, c) for c in df.columns})
        return df

    for key, cfg_key in [
        ("phase1", "phase1_file"),
        ("phase2", "phase2_file"),
        ("phase3", "phase3_file"),
        ("phase3_multi", "phase3_multi_file"),
        ("phase3_univ", "phase3_univ_file"),
    ]:
        path_str = config.get(cfg_key)
        if not path_str:
            continue
        path = Path(path_str)
        if not path.exists():
            logger.warning("Dataset %s not found: %s", key, path)
            continue
        df = _read_dataset(path) if path.suffix.lower() != ".csv" else pd.read_csv(path)
        datasets[key] = _apply_mapping(df)
        logger.info("Loaded %s dataset from %s", key, path)

    # Basic coherence check: try to align column names with raw data when possible
    ref_cols = set(datasets["raw"].columns)
    for name, df in list(datasets.items()):
        extra = set(df.columns) - ref_cols
        if extra:
            logger.debug("%s has %d additional columns", name, len(extra))
    return datasets

