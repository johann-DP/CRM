"""Tools for unsupervised cross-validation and temporal robustness (Block 9).

This module implements :func:`unsupervised_cv_and_temporal_tests` which
assesses the stability of dimensionality reduction methods when the data
are split in random folds or along a chronological axis.
"""

from __future__ import annotations

from typing import Sequence, Dict, Any
import logging

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances

try:
    import umap  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    umap = None  # type: ignore

from block7_evaluation import _prepare_matrix
from block4_factor_methods import _auto_components


def _find_date_column(df: pd.DataFrame) -> str | None:
    """Return the first column containing 'date' if any."""
    for col in df.columns:
        if "date" in col.lower():
            return col
    return None


def _pca_similarity(train: pd.DataFrame, test: pd.DataFrame, cols: Sequence[str]) -> float:
    """Cosine similarity between PCA axes from train and test sets."""
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train[list(cols)])
    X_test = scaler.transform(test[list(cols)])

    pca_train = PCA().fit(X_train)
    n_comp = _auto_components(pca_train.explained_variance_)
    pca_train = PCA(n_components=n_comp).fit(X_train)

    pca_test = PCA(n_components=n_comp).fit(X_test)

    a = pca_train.components_.reshape(n_comp, -1)
    b = pca_test.components_.reshape(n_comp, -1)
    sims = []
    for i in range(n_comp):
        num = np.dot(a[i], b[i])
        denom = np.linalg.norm(a[i]) * np.linalg.norm(b[i])
        sims.append(abs(float(num / denom)))
    return float(np.mean(sims))


def _umap_distance_diff(train: pd.DataFrame, test: pd.DataFrame) -> float:
    """Average pairwise distance difference for UMAP embeddings."""
    if umap is None:
        return float("nan")

    num_cols = train.select_dtypes(include="number").columns.tolist()
    cat_cols = [c for c in train.columns if c not in num_cols]
    X_train = _prepare_matrix(train, num_cols, cat_cols)

    num_cols_t = test.select_dtypes(include="number").columns.tolist()
    cat_cols_t = [c for c in test.columns if c not in num_cols_t]
    X_test = _prepare_matrix(test, num_cols_t, cat_cols_t)

    reducer = umap.UMAP(random_state=0)
    reducer.fit(X_train)
    proj_test = reducer.transform(X_test)

    reducer2 = umap.UMAP(random_state=0)
    emb_test = reducer2.fit_transform(X_test)

    d1 = pairwise_distances(proj_test)
    d2 = pairwise_distances(emb_test)
    return float(np.abs(d1 - d2).mean())


def unsupervised_cv_and_temporal_tests(
    df_active: pd.DataFrame,
    quant_vars: Sequence[str],
    qual_vars: Sequence[str],
    *,
    n_splits: int = 5,
    date_column: str | None = None,
) -> Dict[str, Any]:
    """Evaluate stability of PCA/UMAP across folds and time.

    Parameters
    ----------
    df_active : pandas.DataFrame
        Preprocessed dataset including quantitative and qualitative variables.
    quant_vars : sequence of str
        Names of quantitative variables.
    qual_vars : sequence of str
        Names of qualitative variables.
    n_splits : int, default=5
        Number of folds for cross-validation.
    date_column : str, optional
        Name of column to use for chronological split. If ``None`` an attempt is
        made to infer it.

    Returns
    -------
    dict
        Dictionary with keys ``"cv"`` and ``"temporal"`` storing the computed
        metrics.
    """

    logger = logging.getLogger(__name__)
    results: Dict[str, Any] = {"cv": {}, "temporal": {}}

    if n_splits < 2:
        n_splits = 2

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=0)
    pca_sims: list[float] = []
    umap_diffs: list[float] = []
    for train_idx, test_idx in kf.split(df_active):
        train = df_active.iloc[train_idx]
        test = df_active.iloc[test_idx]

        if quant_vars:
            pca_sims.append(_pca_similarity(train, test, quant_vars))
        if umap is not None:
            umap_diffs.append(_umap_distance_diff(train, test))

    if pca_sims:
        results["cv"]["pca_axis_similarity_mean"] = float(np.mean(pca_sims))
        results["cv"]["pca_axis_similarity_std"] = float(np.std(pca_sims))
    if umap_diffs:
        results["cv"]["umap_distance_diff_mean"] = float(np.mean(umap_diffs))
        results["cv"]["umap_distance_diff_std"] = float(np.std(umap_diffs))

    # --- temporal robustness ----------------------------------------------
    date_column = date_column or _find_date_column(df_active)
    if date_column and pd.api.types.is_datetime64_any_dtype(df_active[date_column]):
        df_sorted = df_active.sort_values(date_column)
        split_date = pd.Timestamp("2020-01-01")
        old = df_sorted[df_sorted[date_column] < split_date]
        new = df_sorted[df_sorted[date_column] >= split_date]

        if len(old) >= 5 and len(new) >= 5 and quant_vars:
            scaler = StandardScaler()
            X_old = scaler.fit_transform(old[quant_vars])
            X_new = scaler.transform(new[quant_vars])

            pca_old = PCA().fit(X_old)
            n_comp = _auto_components(pca_old.explained_variance_)
            pca_old = PCA(n_components=n_comp).fit(X_old)
            pca_new = PCA(n_components=n_comp).fit(X_new)

            proj_old = pca_old.transform(X_old)
            proj_new = pca_old.transform(X_new)
            mean_diff = float(np.linalg.norm(proj_new.mean(axis=0) - proj_old.mean(axis=0)))
            cov_diff = float(np.linalg.norm(np.cov(proj_new, rowvar=False) - np.cov(proj_old, rowvar=False)))
            axis_sim = _pca_similarity(old, new, quant_vars)

            results["temporal"] = {
                "pca_mean_shift": mean_diff,
                "pca_cov_shift": cov_diff,
                "pca_axis_similarity": axis_sim,
            }
    else:
        logger.info("No valid date column for temporal tests")

    return results


__all__ = ["unsupervised_cv_and_temporal_tests"]
