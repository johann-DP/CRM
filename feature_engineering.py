"""Advanced feature engineering pipeline for lead scoring.

This module defines a set of utilities to build complex features for the
lead scoring dataset. The functions defined here can be combined to create
custom preprocessing pipelines involving external data sources and advanced
encoding strategies.
"""

from __future__ import annotations

from typing import Tuple

import logging

import pandas as pd
import numpy as np

from sklearn.preprocessing import OrdinalEncoder, StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import mutual_info_classif, chi2


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

    logger = logging.getLogger(__name__)

    # Work on copies to avoid modifying the original inputs
    X_train = train.copy()
    X_val = val.copy()
    X_test = test.copy()

    # Target extraction (removed from feature matrices)
    y_train = X_train.pop("is_won") if "is_won" in X_train.columns else None
    _ = X_val.pop("is_won") if "is_won" in X_val.columns else None
    _ = X_test.pop("is_won") if "is_won" in X_test.columns else None

    # ------------------------------------------------------------------
    # 1) Feature generation steps
    # ------------------------------------------------------------------
    create_internal_features(X_train, X_val, X_test, lead_cfg)

    cat_cols = lead_cfg.get("cat_features", [])
    min_freq = lead_cfg.get("min_cat_freq", 10)
    reduce_categorical_levels(X_train, X_val, X_test, cat_cols, min_freq)

    enrich_with_sirene(X_train, X_val, X_test)
    enrich_with_geo_data(X_train, X_val, X_test)

    # ------------------------------------------------------------------
    # 2) Update feature lists after enrichment
    # ------------------------------------------------------------------
    target_col = lead_cfg.get("target_col", "Statut commercial")
    exclude = {target_col}

    numeric_features: list[str] = []
    cat_features: list[str] = []
    for col in X_train.columns:
        if col in exclude:
            continue
        if pd.api.types.is_numeric_dtype(X_train[col]):
            numeric_features.append(col)
        else:
            cat_features.append(col)

    lead_cfg["numeric_features"] = numeric_features
    lead_cfg["cat_features"] = cat_features

    # ------------------------------------------------------------------
    # 3) Imputation + categorical encoding
    # ------------------------------------------------------------------
    if numeric_features:
        num_imp = SimpleImputer(strategy="median")
        X_train_num = num_imp.fit_transform(X_train[numeric_features])
        if len(X_val):
            X_val_num = num_imp.transform(X_val[numeric_features])
        else:
            X_val_num = np.empty((0, len(numeric_features)))
        if len(X_test):
            X_test_num = num_imp.transform(X_test[numeric_features])
        else:
            X_test_num = np.empty((0, len(numeric_features)))
    else:
        X_train_num = np.empty((len(X_train), 0))
        X_val_num = np.empty((len(X_val), 0))
        X_test_num = np.empty((len(X_test), 0))

    if cat_features:
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        X_train_cat = enc.fit_transform(X_train[cat_features].astype(str))
        if len(X_val):
            X_val_cat = enc.transform(X_val[cat_features].astype(str))
        else:
            X_val_cat = np.empty((0, len(cat_features)))
        if len(X_test):
            X_test_cat = enc.transform(X_test[cat_features].astype(str))
        else:
            X_test_cat = np.empty((0, len(cat_features)))
    else:
        X_train_cat = np.empty((len(X_train), 0))
        X_val_cat = np.empty((len(X_val), 0))
        X_test_cat = np.empty((len(X_test), 0))

    # ------------------------------------------------------------------
    # 4) Advanced feature selection on numeric variables
    # ------------------------------------------------------------------
    selected_numeric_features = []
    if numeric_features:
        mi_scores = mutual_info_classif(X_train_num, y_train, discrete_features=False)

        scaler_mm = MinMaxScaler()
        X_train_num_mm = scaler_mm.fit_transform(X_train_num)
        chi2_scores, _ = chi2(X_train_num_mm, y_train)

        mi_ranks = (-mi_scores).argsort().argsort()
        chi2_ranks = (-chi2_scores).argsort().argsort()
        combined = mi_ranks + chi2_ranks
        top_idx = np.argsort(combined)[: min(20, len(numeric_features))]
        selected_numeric_features = [numeric_features[i] for i in top_idx]

        # keep only selected columns
        X_train_num = X_train_num[:, top_idx]
        X_val_num = X_val_num[:, top_idx] if len(X_val) else np.empty((0, len(top_idx)))
        X_test_num = X_test_num[:, top_idx] if len(X_test) else np.empty((0, len(top_idx)))

        lead_cfg["numeric_features"] = selected_numeric_features
        logger.debug("Selected %d numeric features", len(selected_numeric_features))

    # ------------------------------------------------------------------
    # 5) Final scaling of numeric variables
    # ------------------------------------------------------------------
    if selected_numeric_features:
        scaler = StandardScaler()
        X_train_num = scaler.fit_transform(X_train_num)
        X_val_num = scaler.transform(X_val_num) if len(X_val) else X_val_num
        X_test_num = scaler.transform(X_test_num) if len(X_test) else X_test_num

    # ------------------------------------------------------------------
    # 6) Assemble final DataFrames
    # ------------------------------------------------------------------
    cols = selected_numeric_features + cat_features
    X_train_final = pd.DataFrame(
        np.column_stack([X_train_num, X_train_cat]) if cols else np.empty((len(X_train), 0)),
        columns=cols,
        index=X_train.index,
    )
    X_val_final = pd.DataFrame(
        np.column_stack([X_val_num, X_val_cat]) if cols else np.empty((len(X_val), 0)),
        columns=cols,
        index=X_val.index,
    )
    X_test_final = pd.DataFrame(
        np.column_stack([X_test_num, X_test_cat]) if cols else np.empty((len(X_test), 0)),
        columns=cols,
        index=X_test.index,
    )

    return X_train_final, X_val_final, X_test_final
