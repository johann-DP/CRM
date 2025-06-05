"""Advanced feature engineering pipeline for lead scoring.

This module defines a set of utilities to build complex features for the
lead scoring dataset. The functions defined here can be combined to create
custom preprocessing pipelines involving external data sources and advanced
encoding strategies.
"""

from __future__ import annotations

from typing import Tuple

import pandas as pd
import numpy as np
import requests
from sklearn.base import BaseEstimator, TransformerMixin


def create_internal_features(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    lead_cfg: dict,
) -> None:
    """Create additional features directly from the input datasets.

    Parameters
    ----------
    train, val, test : pd.DataFrame
        Raw datasets split into training, validation and testing sets.
    lead_cfg : dict
        Configuration dictionary describing the lead scoring setup.
    """
    pass


def reduce_categorical_levels(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    cat_cols: list[str],
    min_freq: int,
) -> None:
    """Reduce the number of modalities for categorical variables.

    This function groups rare categories together based on ``min_freq``
    occurrences in ``X_train`` and applies the mapping to the validation
    and test sets.
    """
    pass


def enrich_with_sirene(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
) -> None:
    """Augment the datasets with company information from the SIRENE API."""
    pass


def enrich_with_geo_data(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    lead_cfg: dict | None = None,
) -> None:
    """Add geographic information based on the French GEO API.

    This helper queries `https://geo.api.gouv.fr` using the client's postal
    code (``"Code postal"`` column) to retrieve two open data attributes:

    ``population_commune``
        Population of the commune associated with the postal code.
    ``code_region``
        Administrative region code of that commune.

    The function adds these columns to ``X_train``, ``X_val`` and
    ``X_test``.  Missing or invalid postal codes yield ``population_commune = 0``
    and ``code_region = "inconnu"``. The external data only describes the
    environment of a lead and therefore does not introduce target leakage.

    Parameters
    ----------
    X_train, X_val, X_test : pd.DataFrame
        Datasets to enrich with geographic information.
    lead_cfg : dict | None
        Lead scoring configuration updated in-place with the new feature names.
    """

    postal_col = "Code postal"
    for df in (X_train, X_val, X_test):
        if postal_col not in df.columns:
            raise KeyError(f"'{postal_col}' column missing from dataset")

    cache: dict[str, tuple[int, str]] = {}

    def _lookup(cp: str | float | int) -> tuple[int, str]:
        if pd.isna(cp):
            return 0, "inconnu"

        code = str(cp).strip()
        # Normalise CPs that may be stored as floats
        if code.endswith('.0'):
            code = code[:-2]

        if code in cache:
            return cache[code]

        url = (
            "https://geo.api.gouv.fr/communes?codePostal="
            f"{code}&fields=population,codeRegion&format=json"
        )
        try:
            resp = requests.get(url, timeout=5)
            resp.raise_for_status()
            data = resp.json() or []
        except Exception:
            data = []

        if not isinstance(data, list) or not data:
            result = (0, "inconnu")
        else:
            # Choose commune with highest population
            best = max(data, key=lambda d: d.get("population") or 0)
            pop = int(best.get("population") or 0)
            region = str(best.get("codeRegion") or "inconnu")
            result = (pop, region)

        cache[code] = result
        return result

    for df in (X_train, X_val, X_test):
        pops, regs = zip(*df[postal_col].apply(_lookup))
        df["population_commune"] = np.array(pops, dtype=int)
        df["code_region"] = np.array(regs, dtype=object)

    if lead_cfg is not None:
        lead_cfg.setdefault("numeric_features", [])
        lead_cfg.setdefault("cat_features", [])
        if "population_commune" not in lead_cfg["numeric_features"]:
            lead_cfg["numeric_features"].append("population_commune")
        if "code_region" not in lead_cfg["cat_features"]:
            lead_cfg["cat_features"].append("code_region")


def advanced_feature_engineering(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    lead_cfg: dict,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return the fully engineered feature matrices.

    This high-level helper orchestrates the various feature engineering
    steps defined in this module and returns processed copies of the
    ``train``, ``val`` and ``test`` datasets.
    """
    pass
