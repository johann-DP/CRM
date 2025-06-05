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
        Configuration dictionary describing the lead scoring setup. The
        function is designed to augment ``train``, ``val`` and ``test`` in
        place with new descriptive variables that do not depend on the target
        column.

    Notes
    -----
    The following features are created when the relevant columns are present:

    ``month`` and ``year``
        Extracted from ``lead_cfg["date_col"]``.

    ``duree_entre_debut_fin``
        Number of days between ``Date de début actualisée`` and
        ``Date de fin réelle``.

    Missing values are replaced with ``0`` so that subsequent encoding steps do
    not produce NaNs. ``lead_cfg['numeric_features']`` is updated with the names
    of the newly created features if necessary.
    """
    if not isinstance(lead_cfg, dict):
        raise TypeError("lead_cfg must be a dictionary")

    date_col = lead_cfg.get("date_col")
    if date_col and date_col in train.columns:
        for df in (train, val, test):
            if date_col not in df.columns:
                continue
            dates = pd.to_datetime(df[date_col], errors="coerce")
            df["month"] = dates.dt.month.fillna(0).astype(int)
            df["year"] = dates.dt.year.fillna(0).astype(int)

    duration_cols = {"Date de début actualisée", "Date de fin réelle"}
    if duration_cols <= set(train.columns):
        for df in (train, val, test):
            if not duration_cols <= set(df.columns):
                continue
            start = pd.to_datetime(df["Date de début actualisée"], errors="coerce")
            end = pd.to_datetime(df["Date de fin réelle"], errors="coerce")
            df["duree_entre_debut_fin"] = (end - start).dt.days.fillna(0.0)

    # Update numeric_features list with the newly created features
    num_feats = lead_cfg.get("numeric_features", [])
    for feat in ["month", "year", "duree_entre_debut_fin"]:
        if feat in train.columns and feat not in num_feats:
            num_feats.append(feat)
    lead_cfg["numeric_features"] = num_feats


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
) -> None:
    """Add geographic information such as coordinates or region codes."""
    pass


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
