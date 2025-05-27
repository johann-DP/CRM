# -*- coding: utf-8 -*-
"""Utility functions for factorial analyses (PCA, MCA, FAMD, MFA).

This module implements standalone wrappers around ``scikit-learn`` and
``prince`` to run the main factorial analysis methods used in the project.
The functions do not depend on other local modules so they can be reused
independently of ``phase4v2.py`` or the fine-tuning scripts.
"""

from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional

from pandas.api.types import is_object_dtype, is_categorical_dtype

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import prince


def _get_explained_inertia(model: object) -> List[float]:
    """Return the explained inertia ratio for a fitted model."""
    inertia = getattr(model, "explained_inertia_", None)
    if inertia is not None:
        return list(np.asarray(inertia, dtype=float))

    eigenvalues = getattr(model, "eigenvalues_", None)
    if eigenvalues is None:
        return []
    total = float(np.sum(eigenvalues))
    if total == 0:
        return []
    return list(np.asarray(eigenvalues, dtype=float) / total)


def _select_n_components(eigenvalues: np.ndarray, threshold: float = 0.8) -> int:
    """Select a number of components using Kaiser and inertia criteria."""
    ev = np.asarray(eigenvalues, dtype=float)
    if ev.sum() <= 1.0:
        ev = ev * len(ev)

    n_kaiser = max(1, int(np.sum(ev >= 1)))
    ratios = ev / ev.sum()
    cum = np.cumsum(ratios)
    n_inertia = int(np.searchsorted(cum, threshold) + 1)
    return max(n_kaiser, n_inertia)


def run_pca(
    df_active: pd.DataFrame,
    quant_vars: List[str],
    n_components: Optional[int] = None,
    *,
    optimize: bool = False,
    variance_threshold: float = 0.8,
    random_state: Optional[int] = None,
    whiten: Optional[bool] = None,
    svd_solver: Optional[str] = None,
) -> Dict[str, object]:
    """Run a Principal Component Analysis on quantitative variables.

    Parameters
    ----------
    df_active : pandas.DataFrame
        DataFrame containing the active observations.
    quant_vars : list of str
        Names of the quantitative columns to use.
    n_components : int, optional
        Number of components to keep. If ``None`` and ``optimize`` is ``True``
        the value is determined automatically with a variance threshold.
    optimize : bool, default ``False``
        Activate automatic selection of ``n_components`` when ``n_components`` is
        not provided.
    variance_threshold : float, default ``0.8``
        Cumulative explained variance ratio threshold when ``optimize`` is true.
    random_state : int, optional
        Random state forwarded to :class:`sklearn.decomposition.PCA`.
    whiten : bool, optional
        If provided, sets the ``whiten`` parameter of :class:`~sklearn.decomposition.PCA`.
    svd_solver : str, optional
        If provided, sets the ``svd_solver`` parameter of :class:`~sklearn.decomposition.PCA`.

    Returns
    -------
    dict
        ``{"model", "inertia", "embeddings", "loadings", "runtime_s"}``
    """
    start = time.perf_counter()
    logger = logging.getLogger(__name__)

    X = StandardScaler().fit_transform(df_active[quant_vars])
    max_dim = min(X.shape)

    if optimize and n_components is None:
        tmp = PCA(n_components=max_dim, random_state=random_state).fit(X)
        n_components = _select_n_components(tmp.explained_variance_, threshold=variance_threshold)
        logger.info("PCA: selected %d components automatically", n_components)

    n_components = n_components or max_dim
    kwargs = {}
    if whiten is not None:
        kwargs["whiten"] = whiten
    if svd_solver is not None:
        kwargs["svd_solver"] = svd_solver
    pca = PCA(n_components=n_components, random_state=random_state, **kwargs)
    emb = pca.fit_transform(X)

    inertia = pd.Series(pca.explained_variance_ratio_,
                        index=[f"F{i+1}" for i in range(pca.n_components_)])
    embeddings = pd.DataFrame(emb, index=df_active.index,
                              columns=[f"F{i+1}" for i in range(pca.n_components_)])
    loadings = pd.DataFrame(pca.components_.T, index=quant_vars,
                            columns=[f"F{i+1}" for i in range(pca.n_components_)])

    runtime = time.perf_counter() - start
    return {
        "model": pca,
        "inertia": inertia,
        "embeddings": embeddings,
        "loadings": loadings,
        "runtime_s": runtime,
    }


def run_mca(
    df_active: pd.DataFrame,
    qual_vars: List[str],
    n_components: Optional[int] = None,
    *,
    optimize: bool = False,
    variance_threshold: float = 0.8,
    random_state: Optional[int] = None,
    normalize: bool = True,
    n_iter: int = 3,
) -> Dict[str, object]:
    """Run Multiple Correspondence Analysis on qualitative variables.

    Parameters
    ----------
    df_active : pandas.DataFrame
        Input data with qualitative variables.
    qual_vars : list of str
        Names of the qualitative columns to use.
    n_components : int, optional
        Number of dimensions to compute. If ``None`` and ``optimize`` is
        ``True`` the value is selected automatically.
    optimize : bool, default ``False``
        Activate automatic selection of ``n_components`` when not provided.
    variance_threshold : float, default ``0.8``
        Cumulative inertia threshold when ``optimize`` is enabled.
    random_state : int, optional
        Random state passed to :class:`prince.MCA`.
    normalize : bool, default ``True``
        If ``True`` applies the Benzecri correction (``correction='benzecri'``).
    n_iter : int, default ``3``
        Number of iterations for the underlying algorithm.
    """
    start = time.perf_counter()
    logger = logging.getLogger(__name__)

    df_cat = df_active[qual_vars].astype("category")

    if optimize and n_components is None:
        max_dim = sum(df_cat[c].nunique() - 1 for c in df_cat.columns)
        tmp = prince.MCA(n_components=max_dim, random_state=random_state).fit(df_cat)
        eig = getattr(tmp, "eigenvalues_", None)
        if eig is None:
            eig = np.asarray(_get_explained_inertia(tmp)) * max_dim
        n_components = _select_n_components(eig, threshold=variance_threshold)
        logger.info("MCA: selected %d components automatically", n_components)

    n_components = n_components or 5
    corr = "benzecri" if normalize else None
    mca = prince.MCA(
        n_components=n_components,
        n_iter=n_iter,
        correction=corr,
        random_state=random_state,
    )
    mca = mca.fit(df_cat)

    inertia = pd.Series(_get_explained_inertia(mca),
                        index=[f"F{i+1}" for i in range(mca.n_components)])
    embeddings = mca.row_coordinates(df_cat)
    embeddings.index = df_active.index
    col_coords = mca.column_coordinates(df_cat)

    runtime = time.perf_counter() - start
    return {
        "model": mca,
        "inertia": inertia,
        "embeddings": embeddings,
        "column_coords": col_coords,
        "runtime_s": runtime,
    }


def run_famd(df_active: pd.DataFrame, quant_vars: List[str], qual_vars: List[str],
             n_components: Optional[int] = None, *, optimize: bool = False,
             variance_threshold: float = 0.8, random_state: Optional[int] = None) -> Dict[str, object]:
    """Run Factor Analysis of Mixed Data (FAMD)."""
    start = time.perf_counter()
    logger = logging.getLogger(__name__)

    scaler = StandardScaler()
    X_quanti = scaler.fit_transform(df_active[quant_vars])
    df_quanti = pd.DataFrame(X_quanti, index=df_active.index, columns=quant_vars)
    df_mix = pd.concat([df_quanti, df_active[qual_vars].astype("category")], axis=1)

    if df_mix.isnull().any().any():
        raise ValueError("Input contains NaN values")

    if optimize and n_components is None:
        max_dim = df_mix.shape[1]
        tmp = prince.FAMD(n_components=max_dim, n_iter=3, random_state=random_state).fit(df_mix)
        eig = getattr(tmp, "eigenvalues_", None)
        if eig is None:
            eig = np.asarray(_get_explained_inertia(tmp)) * max_dim
        n_components = _select_n_components(eig, threshold=variance_threshold)
        logger.info("FAMD: selected %d components automatically", n_components)

    n_components = n_components or df_mix.shape[1]
    famd = prince.FAMD(n_components=n_components, n_iter=3, random_state=random_state)
    famd = famd.fit(df_mix)

    inertia = pd.Series(_get_explained_inertia(famd),
                        index=[f"F{i+1}" for i in range(famd.n_components)])
    embeddings = famd.row_coordinates(df_mix)
    embeddings.index = df_active.index
    if hasattr(famd, "column_coordinates"):
        col_coords = famd.column_coordinates(df_mix)
    elif hasattr(famd, "column_coordinates_"):
        col_coords = famd.column_coordinates_
    else:
        col_coords = pd.DataFrame()

    if hasattr(famd, "column_contributions"):
        contrib = famd.column_contributions(df_mix)
    elif hasattr(famd, "column_contributions_"):
        contrib = famd.column_contributions_
    else:
        contrib = (col_coords ** 2).div((col_coords ** 2).sum(axis=0), axis=1) * 100

    runtime = time.perf_counter() - start
    return {
        "model": famd,
        "inertia": inertia,
        "embeddings": embeddings,
        "column_coords": col_coords,
        "contributions": contrib,
        "runtime_s": runtime,
    }


def run_mfa(df_active: pd.DataFrame, groups: List[List[str]], n_components: Optional[int] = None,
            *, optimize: bool = False, variance_threshold: float = 0.8,
            normalize: bool = True, random_state: Optional[int] = None,
            n_iter: int = 3) -> Dict[str, object]:
    """Run Multiple Factor Analysis."""
    start = time.perf_counter()
    logger = logging.getLogger(__name__)

    # one-hot encode qualitative variables that appear in groups
    qual_cols = []
    for group in groups:
        for col in group:
            if col in df_active.columns and (
                is_object_dtype(df_active[col]) or is_categorical_dtype(df_active[col])
            ):
                qual_cols.append(col)
    # remove duplicates while preserving order
    seen = set()
    qual_cols = [c for c in qual_cols if not (c in seen or seen.add(c))]
    if qual_cols:
        enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        encoded = enc.fit_transform(df_active[qual_cols])
        df_dummies = pd.DataFrame(encoded, index=df_active.index,
                                  columns=enc.get_feature_names_out(qual_cols))
    else:
        df_dummies = pd.DataFrame(index=df_active.index)

    df_num = df_active.drop(columns=qual_cols)
    df_all = pd.concat([df_num, df_dummies], axis=1)

    groups_dict: Dict[str, List[str]] = {}
    used_cols: List[str] = []
    for i, g in enumerate(groups, 1):
        cols: List[str] = []
        for v in g:
            if v in df_all.columns:
                cols.append(v)
            else:
                # qualitative variables have been expanded
                cols.extend([c for c in df_all.columns if c.startswith(f"{v}_")])
        if cols:
            name = f"G{i}"
            groups_dict[name] = cols
            used_cols.extend(cols)

    remaining = [c for c in df_all.columns if c not in used_cols]
    if remaining:
        groups_dict[f"G{len(groups_dict)+1}"] = remaining
        used_cols.extend(remaining)
    df_all = df_all[used_cols]

    if normalize:
        scaler = StandardScaler()
        for cols in groups_dict.values():
            if cols:
                df_all[cols] = scaler.fit_transform(df_all[cols])

    if optimize and n_components is None:
        max_dim = df_all.shape[1]
        tmp = prince.MFA(n_components=max_dim, n_iter=n_iter, random_state=random_state)
        tmp = tmp.fit(df_all, groups=groups_dict)
        eig = getattr(tmp, "eigenvalues_", None)
        if eig is None:
            eig = (tmp.percentage_of_variance_ / 100) * max_dim
        n_components = _select_n_components(np.asarray(eig), threshold=variance_threshold)
        logger.info("MFA: selected %d components automatically", n_components)

    n_components = n_components or 5
    mfa = prince.MFA(n_components=n_components, n_iter=n_iter, random_state=random_state)
    mfa = mfa.fit(df_all, groups=groups_dict)
    mfa.explained_inertia_ = mfa.percentage_of_variance_ / 100
    embeddings = mfa.row_coordinates(df_all)
    embeddings.index = df_active.index

    inertia = pd.Series(mfa.explained_inertia_,
                        index=[f"F{i+1}" for i in range(len(mfa.explained_inertia_))])

    runtime = time.perf_counter() - start
    return {
        "model": mfa,
        "inertia": inertia,
        "embeddings": embeddings,
        "runtime_s": runtime,
    }

