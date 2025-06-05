"""Advanced feature engineering pipeline for lead scoring.

This module defines a set of utilities to build complex features for the
lead scoring dataset. The functions defined here can be combined to create
custom preprocessing pipelines involving external data sources and advanced
encoding strategies.
"""

from __future__ import annotations

from typing import Tuple, List

import logging

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, MinMaxScaler, StandardScaler
from sklearn.feature_selection import mutual_info_classif, chi2

logger = logging.getLogger(__name__)


def create_temporal_features(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    lead_cfg: dict,
) -> None:
    """Create month, year, and duration features based on date columns.

    New columns added:
        - 'month': month extracted from lead_cfg['date_col']
        - 'year': year extracted from lead_cfg['date_col']
        - 'duree_entre_debut_fin': duration in days between
          'Date de début actualisée' and 'Date de fin réelle'.

    Missing values are replaced with 0 so that subsequent encoding steps do
    not produce NaNs. lead_cfg['numeric_features'] is updated with the names
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
    col: str,
    min_freq: int = 5,
) -> None:
    """Group rare categories into 'Autre' for a given column.

    Any category appearing fewer than min_freq times in X_train is
    replaced by 'Autre' in all datasets. Ensures 'Autre' is present
    in the category index.
    """
    if col not in X_train.columns:
        return

    train_series = X_train[col].astype("category")
    counts = train_series.value_counts(dropna=False)

    threshold = min_freq
    frequent = set(counts[counts >= threshold].index)

    for df in (X_train, X_val, X_test):
        if col not in df.columns:
            continue
        df[col] = df[col].astype("category")
        if "Autre" not in df[col].cat.categories:
            df[col] = df[col].cat.add_categories(["Autre"])
        df[col] = df[col].apply(lambda x: x if x in frequent else "Autre")
        df[col] = df[col].cat.remove_unused_categories()


def encode_features(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    cat_feats: List[str],
    num_feats: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Apply imputation, scaling, and encoding to both categorical and numeric features.

    Steps:
    1. Impute numeric features with median.
    2. Scale numeric features using StandardScaler.
    3. Impute categorical features with constant 'missing', then apply
       OrdinalEncoder to categories, followed by OneHotEncoder for
       high-cardinality variables if needed.
    4. Return transformed X_train, X_val, X_test DataFrames.
    """
    # Numeric pipeline
    if num_feats:
        num_imputer = SimpleImputer(strategy="median")
        num_scaler = StandardScaler()

        X_train_num = num_imputer.fit_transform(X_train[num_feats])
        X_train_num = num_scaler.fit_transform(X_train_num)

        X_val_num = num_imputer.transform(X_val[num_feats])
        X_val_num = num_scaler.transform(X_val_num)

        X_test_num = num_imputer.transform(X_test[num_feats])
        X_test_num = num_scaler.transform(X_test_num)

        X_train_num = pd.DataFrame(X_train_num, columns=num_feats, index=X_train.index)
        X_val_num = pd.DataFrame(X_val_num, columns=num_feats, index=X_val.index)
        X_test_num = pd.DataFrame(X_test_num, columns=num_feats, index=X_test.index)
    else:
        X_train_num = pd.DataFrame(index=X_train.index)
        X_val_num = pd.DataFrame(index=X_val.index)
        X_test_num = pd.DataFrame(index=X_test.index)

    # Categorical pipeline
    X_train_cat = pd.DataFrame(index=X_train.index)
    X_val_cat = pd.DataFrame(index=X_val.index)
    X_test_cat = pd.DataFrame(index=X_test.index)

    if cat_feats:
        # Impute missing categories
        for col in cat_feats:
            X_train[col] = X_train[col].fillna("missing").astype("category")
            X_val[col] = X_val[col].fillna("missing").astype("category")
            X_test[col] = X_test[col].fillna("missing").astype("category")

            # Reduce rare levels
            reduce_categorical_levels(X_train, X_val, X_test, col)

        # Ordinal encoding followed by one-hot
        ord_enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        X_train_ord = ord_enc.fit_transform(X_train[cat_feats])
        X_val_ord = ord_enc.transform(X_val[cat_feats])
        X_test_ord = ord_enc.transform(X_test[cat_feats])

        X_train_ord = pd.DataFrame(X_train_ord, columns=cat_feats, index=X_train.index)
        X_val_ord = pd.DataFrame(X_val_ord, columns=cat_feats, index=X_val.index)
        X_test_ord = pd.DataFrame(X_test_ord, columns=cat_feats, index=X_test.index)

        # One-hot encode ordinal labels
        onehot_enc = OneHotEncoder(sparse=False, handle_unknown="ignore")
        X_train_ohe = onehot_enc.fit_transform(X_train_ord)
        X_val_ohe = onehot_enc.transform(X_val_ord)
        X_test_ohe = onehot_enc.transform(X_test_ord)

        ohe_cols = onehot_enc.get_feature_names_out(cat_feats)
        X_train_cat = pd.DataFrame(X_train_ohe, columns=ohe_cols, index=X_train.index)
        X_val_cat = pd.DataFrame(X_val_ohe, columns=ohe_cols, index=X_val.index)
        X_test_cat = pd.DataFrame(X_test_ohe, columns=ohe_cols, index=X_test.index)

    # Concatenate numeric and categorical
    X_train_final = pd.concat([X_train_num, X_train_cat], axis=1)
    X_val_final = pd.concat([X_val_num, X_val_cat], axis=1)
    X_test_final = pd.concat([X_test_num, X_test_cat], axis=1)

    return X_train_final, X_val_final, X_test_final


def select_features_univariate(
    X: pd.DataFrame,
    y: pd.Series,
    k: int = 10,
    method: str = "mutual_info",
) -> List[str]:
    """Select top-k features based on univariate statistical tests.

    Arguments:
        X: DataFrame of features.
        y: Target binary Series (0/1).
        k: Number of features to select.
        method: 'mutual_info' or 'chi2'.

    Returns:
        List of top-k feature names.
    """
    if method == "mutual_info":
        scores = mutual_info_classif(X, y, discrete_features="auto")
    elif method == "chi2":
        # chi2 requires non-negative features; ensure X >= 0
        X_nonneg = X.copy()
        for col in X_nonneg.columns:
            if X_nonneg[col].min() < 0:
                X_nonneg[col] = X_nonneg[col] - X_nonneg[col].min()
        scores = chi2(X_nonneg, y)[0]
    else:
        raise ValueError("method must be 'mutual_info' or 'chi2'")

    feature_scores = pd.Series(scores, index=X.columns)
    topk = feature_scores.sort_values(ascending=False).index[:k].tolist()
    return topk


def bin_numerical_feature(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    col: str,
    n_bins: int = 4,
) -> None:
    """Discretize a numerical column into quantile-based bins and update datasets.

    The new column will be named '{col}_bin'. Original column remains unchanged.
    """
    if col not in X_train.columns:
        return

    # Compute bin edges on X_train
    edges = pd.qcut(X_train[col], q=n_bins, duplicates="drop", retbins=True)[1]
    bin_labels = [f"{col}_bin_{i}" for i in range(len(edges) - 1)]

    # Apply binning to each DataFrame
    for df in (X_train, X_val, X_test):
        if col in df.columns:
            df[f"{col}_bin"] = pd.cut(df[col], bins=edges, labels=bin_labels, include_lowest=True)
            df[f"{col}_bin"] = df[f"{col}_bin"].cat.add_categories(["missing"]).fillna("missing")


def run_full_feature_engineering(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    lead_cfg: dict,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run the complete feature engineering pipeline on train, val, and test sets.

    Steps executed in order:
        1. Create temporal features (month, year, duration).
        2. Bin/quantize selected numeric features if specified in lead_cfg.
        3. Encode categorical and numeric features fully.

    Returns:
        Transformed (X_train, X_val, X_test).
    """
    # 1. Temporal features
    create_temporal_features(train, val, test, lead_cfg)

    # 2. Binning if requested
    bin_features = lead_cfg.get("binning_features", [])
    for col in bin_features:
        bin_numerical_feature(train, val, test, col, n_bins=lead_cfg.get("n_bins", 4))

    # 3. Encoding
    cat_feats = lead_cfg.get("cat_features", [])
    num_feats = lead_cfg.get("numeric_features", [])
    X_train_fe, X_val_fe, X_test_fe = encode_features(train, val, test, cat_feats, num_feats)

    return X_train_fe, X_val_fe, X_test_fe
