"""Unsupervised cross-validation and temporal robustness tests."""

from __future__ import annotations

import logging
from typing import Sequence, Tuple, Optional, Dict

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from scipy.spatial.distance import pdist

__all__ = ["unsupervised_cv_and_temporal_tests"]


def _find_date_column(df: pd.DataFrame) -> Optional[str]:
    """Return the first column name containing 'date' (case-insensitive)."""
    for col in df.columns:
        if "date" in col.lower():
            return col
    return None


def _fit_preprocess(
    df: pd.DataFrame, quant_vars: Sequence[str], qual_vars: Sequence[str]
) -> Tuple[np.ndarray, Optional[StandardScaler], Optional[OneHotEncoder]]:
    """Scale numeric columns and one-hot encode categoricals."""
    scaler: Optional[StandardScaler] = None
    encoder: Optional[OneHotEncoder] = None

    X_num = np.empty((len(df), 0))
    if quant_vars:
        scaler = StandardScaler()
        X_num = scaler.fit_transform(df[quant_vars])

    X_cat = np.empty((len(df), 0))
    if qual_vars:
        try:
            encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        except TypeError:  # pragma: no cover - older scikit-learn
            encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
        X_cat = encoder.fit_transform(df[qual_vars])

    if X_num.size and X_cat.size:
        X = np.hstack([X_num, X_cat])
    elif X_num.size:
        X = X_num
    else:
        X = X_cat

    return X, scaler, encoder


def _transform(
    df: pd.DataFrame,
    quant_vars: Sequence[str],
    qual_vars: Sequence[str],
    scaler: Optional[StandardScaler],
    encoder: Optional[OneHotEncoder],
) -> np.ndarray:
    """Apply preprocessing fitted on the training data."""
    X_num = np.empty((len(df), 0))
    if quant_vars and scaler is not None:
        X_num = scaler.transform(df[quant_vars])

    X_cat = np.empty((len(df), 0))
    if qual_vars and encoder is not None:
        X_cat = encoder.transform(df[qual_vars])

    if X_num.size and X_cat.size:
        return np.hstack([X_num, X_cat])
    if X_num.size:
        return X_num
    return X_cat


def _axis_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Return the mean absolute cosine similarity between corresponding axes."""
    sims = []
    for v1, v2 in zip(a, b):
        num = np.abs(np.dot(v1, v2))
        denom = np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-12
        sims.append(num / denom)
    return float(np.mean(sims)) if sims else float("nan")


def _distance_discrepancy(X1: np.ndarray, X2: np.ndarray) -> float:
    """Return the relative Frobenius norm between distance matrices."""
    d1 = pdist(X1)
    d2 = pdist(X2)
    norm = np.linalg.norm(d2) + 1e-12
    return float(np.linalg.norm(d1 - d2) / norm)


def unsupervised_cv_and_temporal_tests(
    df_active: pd.DataFrame,
    quant_vars: Sequence[str],
    qual_vars: Sequence[str],
    *,
    n_splits: int = 5,
    random_state: Optional[int] = None,
) -> Dict[str, Dict[str, float]]:
    """Assess stability of PCA/UMAP with cross-validation and temporal splits."""

    logger = logging.getLogger(__name__)

    if not isinstance(df_active, pd.DataFrame):
        raise TypeError("df_active must be a DataFrame")
    if n_splits < 2:
        raise ValueError("n_splits must be >= 2")

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    pca_axis_scores: list[float] = []
    pca_dist_scores: list[float] = []
    umap_dist_scores: list[float] = []

    for train_idx, test_idx in kf.split(df_active):
        df_train = df_active.iloc[train_idx]
        df_test = df_active.iloc[test_idx]

        X_train, scaler, encoder = _fit_preprocess(df_train, quant_vars, qual_vars)
        X_test = _transform(df_test, quant_vars, qual_vars, scaler, encoder)

        n_comp = min(2, X_train.shape[1]) or 1
        pca_train = PCA(n_components=n_comp, random_state=random_state).fit(X_train)
        emb_proj = pca_train.transform(X_test)

        pca_test = PCA(n_components=n_comp, random_state=random_state)
        emb_test = pca_test.fit_transform(X_test)

        pca_axis_scores.append(_axis_similarity(pca_train.components_, pca_test.components_))
        pca_dist_scores.append(_distance_discrepancy(emb_proj, emb_test))

        try:
            import umap  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency
            logger.warning("UMAP unavailable: %s", exc)
            umap = None  # type: ignore
        if umap is not None:
            reducer_train = umap.UMAP(n_components=2, random_state=random_state)
            reducer_train.fit(X_train)
            emb_umap_proj = reducer_train.transform(X_test)
            reducer_test = umap.UMAP(n_components=2, random_state=random_state)
            emb_umap_test = reducer_test.fit_transform(X_test)
            umap_dist_scores.append(_distance_discrepancy(emb_umap_proj, emb_umap_test))

    cv_stability = {
        "pca_axis_corr_mean": float(np.nanmean(pca_axis_scores)) if pca_axis_scores else float("nan"),
        "pca_axis_corr_std": float(np.nanstd(pca_axis_scores)) if pca_axis_scores else float("nan"),
        "pca_distance_mse_mean": float(np.mean(pca_dist_scores)) if pca_dist_scores else float("nan"),
        "pca_distance_mse_std": float(np.std(pca_dist_scores)) if pca_dist_scores else float("nan"),
        "umap_distance_mse_mean": float(np.mean(umap_dist_scores)) if umap_dist_scores else float("nan"),
        "umap_distance_mse_std": float(np.std(umap_dist_scores)) if umap_dist_scores else float("nan"),
    }

    # Temporal robustness -------------------------------------------------
    date_col = _find_date_column(df_active)
    temporal_shift: Optional[Dict[str, float]] = None
    if date_col:
        df_sorted = df_active.sort_values(date_col)
        split_point = len(df_sorted) // 2
        df_old = df_sorted.iloc[:split_point]
        df_new = df_sorted.iloc[split_point:]

        X_old, scaler, encoder = _fit_preprocess(df_old, quant_vars, qual_vars)
        X_new = _transform(df_new, quant_vars, qual_vars, scaler, encoder)

        n_comp = min(2, X_old.shape[1]) or 1
        pca_old = PCA(n_components=n_comp, random_state=random_state).fit(X_old)
        emb_proj = pca_old.transform(X_new)

        pca_new = PCA(n_components=n_comp, random_state=random_state)
        emb_new = pca_new.fit_transform(X_new)

        axis_corr = _axis_similarity(pca_old.components_, pca_new.components_)
        dist_diff = _distance_discrepancy(emb_proj, emb_new)
        mean_shift = float(np.linalg.norm(emb_proj.mean(axis=0) - pca_old.transform(X_old).mean(axis=0)))

        umap_dist = float("nan")
        if umap is not None:
            reducer_old = umap.UMAP(n_components=2, random_state=random_state)
            reducer_old.fit(X_old)
            emb_old_umap = reducer_old.transform(X_old)
            emb_proj_umap = reducer_old.transform(X_new)
            reducer_new = umap.UMAP(n_components=2, random_state=random_state)
            emb_new_umap = reducer_new.fit_transform(X_new)
            umap_dist = _distance_discrepancy(emb_proj_umap, emb_new_umap)

        temporal_shift = {
            "pca_axis_corr": axis_corr,
            "pca_distance_mse": dist_diff,
            "pca_mean_shift": mean_shift,
            "umap_distance_mse": umap_dist,
        }

    return {"cv_stability": cv_stability, "temporal_shift": temporal_shift}
