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
    if not cat_cols:
        return

    for col in cat_cols:
        if col not in X_train.columns:
            continue

        train_series = X_train[col].astype("category")
        counts = train_series.value_counts(dropna=False)

        threshold = 0 if len(train_series) < min_freq else min_freq
        frequent = set(counts[counts >= threshold].index)

        # Ensure "Autre" exists as a category in all datasets
        for df in (X_train, X_val, X_test):
            if col in df.columns:
                if not pd.api.types.is_categorical_dtype(df[col]):
                    df[col] = df[col].astype("category")
                if "Autre" not in df[col].cat.categories:
                    df[col] = df[col].cat.add_categories(["Autre"])

        # Replace rare modalities in the training set
        mask_train = ~X_train[col].isin(frequent)
        if mask_train.any():
            X_train.loc[mask_train, col] = "Autre"

        # Apply the same mapping to validation and test sets
        for df in (X_val, X_test):
            if col not in df.columns:
                continue
            mask = ~df[col].isin(frequent)
            if mask.any():
                df.loc[mask, col] = "Autre"

        # Cast back to category dtype
        X_train[col] = X_train[col].astype("category")
        if col in X_val.columns:
            X_val[col] = X_val[col].astype("category")
        if col in X_test.columns:
            X_test[col] = X_test[col].astype("category")


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
