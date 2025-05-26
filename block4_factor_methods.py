"""Utility functions for fine-tuning factorial methods (Block 4).

Each function runs a dimensionality reduction technique on the active dataset
and automatically chooses the number of components using classic rules
(Kaiser and cumulative explained variance > 80%). The runtime in seconds is
returned together with the fitted model and the individual coordinates.
"""

from __future__ import annotations

from typing import List, Sequence, Dict, Optional
from time import time

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import prince


# ---------------------------------------------------------------------------
# Helper to select number of components
# ---------------------------------------------------------------------------

def _auto_components(eigenvalues: Sequence[float], threshold: float = 0.8) -> int:
    """Determine the number of components to keep.

    Parameters
    ----------
    eigenvalues : Sequence[float]
        Eigenvalues of the decomposition.
    threshold : float, optional
        Cumulative inertia threshold.
    """
    ev = np.asarray(eigenvalues, dtype=float)
    n_kaiser = int((ev > 1).sum())
    ratio = ev / ev.sum()
    n_inertia = np.searchsorted(np.cumsum(ratio), threshold) + 1
    n = max(1, max(n_kaiser, n_inertia))
    return int(n)


# ---------------------------------------------------------------------------
# PCA
# ---------------------------------------------------------------------------

def run_pca(df_active: pd.DataFrame, quant_vars: List[str]) -> Dict[str, object]:
    """Run PCA on quantitative variables with automatic dimension selection."""
    start = time()

    scaler = StandardScaler()
    X = scaler.fit_transform(df_active[quant_vars])

    tmp = PCA().fit(X)
    n_comp = _auto_components(tmp.explained_variance_)

    pca = PCA(n_components=n_comp, random_state=0)
    scores = pca.fit_transform(X)
    inertia = pd.Series(
        pca.explained_variance_ratio_,
        index=[f"F{i+1}" for i in range(n_comp)],
    )
    embeddings = pd.DataFrame(scores, index=df_active.index, columns=inertia.index)

    runtime = time() - start
    return {
        "model": pca,
        "inertia": inertia,
        "embeddings": embeddings,
        "runtime_s": runtime,
    }


# ---------------------------------------------------------------------------
# MCA
# ---------------------------------------------------------------------------

def run_mca(df_active: pd.DataFrame, qual_vars: List[str]) -> Dict[str, object]:
    """Run Multiple Correspondence Analysis with automatic dimension selection."""
    start = time()

    df_cat = df_active[qual_vars].astype("category")
    df_cat.replace([np.inf, -np.inf], np.nan, inplace=True)
    if df_cat.isna().any().any():
        df_cat = df_cat.fillna("Non renseigné").astype("category")
    max_dim = sum(df_cat[c].nunique() for c in df_cat.columns) - len(df_cat.columns)

    tmp = prince.MCA(n_components=max_dim).fit(df_cat)
    n_comp = _auto_components(tmp.eigenvalues_)

    mca = prince.MCA(n_components=n_comp)
    mca = mca.fit(df_cat)
    rows = mca.row_coordinates(df_cat)
    rows.index = df_active.index
    inertia = pd.Series(
        mca.explained_inertia_, index=[f"F{i+1}" for i in range(n_comp)]
    )

    runtime = time() - start
    return {
        "model": mca,
        "inertia": inertia,
        "embeddings": rows,
        "runtime_s": runtime,
    }


# ---------------------------------------------------------------------------
# FAMD
# ---------------------------------------------------------------------------

def run_famd(
    df_active: pd.DataFrame, quant_vars: List[str], qual_vars: List[str]
) -> Dict[str, object]:
    """Run Factor Analysis for Mixed Data with automatic component selection."""
    start = time()

    scaler = StandardScaler()
    df_quanti = pd.DataFrame(
        scaler.fit_transform(df_active[quant_vars]),
        index=df_active.index,
        columns=quant_vars,
    )
    df_cat = df_active[qual_vars].astype("category")
    df_mix = pd.concat([df_quanti, df_cat], axis=1)
    df_mix.replace([np.inf, -np.inf], np.nan, inplace=True)
    if df_mix.isna().any().any():
        num_cols = df_mix.select_dtypes(include="number").columns
        cat_cols = [c for c in df_mix.columns if c not in num_cols]
        if len(num_cols):
            df_mix[num_cols] = df_mix[num_cols].fillna(0)
        for col in cat_cols:
            df_mix[col] = df_mix[col].cat.add_categories("Non renseigné")
            df_mix[col] = df_mix[col].fillna("Non renseigné")

    tmp = prince.FAMD(n_components=df_mix.shape[1], n_iter=3, engine="sklearn").fit(df_mix)
    eigenvalues = getattr(tmp, "eigenvalues_", None)
    if eigenvalues is None:
        eigenvalues = np.array(tmp.explained_inertia_) * df_mix.shape[1]
    n_comp = _auto_components(eigenvalues)

    famd = prince.FAMD(n_components=n_comp, n_iter=3, engine="sklearn")
    famd = famd.fit(df_mix)
    rows = famd.row_coordinates(df_mix)
    rows.index = df_active.index
    inertia = pd.Series(
        famd.explained_inertia_, index=[f"F{i+1}" for i in range(n_comp)]
    )

    runtime = time() - start
    return {
        "model": famd,
        "inertia": inertia,
        "embeddings": rows,
        "runtime_s": runtime,
    }


# ---------------------------------------------------------------------------
# MFA
# ---------------------------------------------------------------------------

def run_mfa(
    df_active: pd.DataFrame,
    groups: Sequence[Sequence[str]],
) -> Dict[str, object]:
    """Run Multiple Factor Analysis with automatic component selection."""
    start = time()

    # Build dataframe with encoded qualitative variables
    df_proc = pd.DataFrame(index=df_active.index)
    group_dict: Dict[str, List[str]] = {}
    for i, cols in enumerate(groups):
        gname = f"G{i+1}"
        sub = df_active[list(cols)].copy()
        num_cols = sub.select_dtypes(include="number").columns
        cat_cols = [c for c in sub.columns if c not in num_cols]
        if len(num_cols) > 0:
            scaler = StandardScaler()
            sub[num_cols] = scaler.fit_transform(sub[num_cols])
        if cat_cols:
            sub = pd.get_dummies(sub, columns=cat_cols)
        group_dict[gname] = list(sub.columns)
        df_proc = pd.concat([df_proc, sub], axis=1)

    df_proc.replace([np.inf, -np.inf], np.nan, inplace=True)
    if df_proc.isna().any().any():
        num_cols_all = df_proc.select_dtypes(include="number").columns
        if len(num_cols_all):
            df_proc[num_cols_all] = df_proc[num_cols_all].fillna(0)
        df_proc = df_proc.fillna(0)

    tmp = prince.MFA(n_components=df_proc.shape[1]).fit(df_proc, groups=group_dict)
    eigenvalues = getattr(tmp, "eigenvalues_", None)
    if eigenvalues is None:
        eigenvalues = (tmp.percentage_of_variance_ / 100) * df_proc.shape[1]
    n_comp = _auto_components(eigenvalues)

    mfa = prince.MFA(n_components=n_comp)
    mfa = mfa.fit(df_proc, groups=group_dict)
    rows = mfa.row_coordinates(df_proc)
    rows.index = df_active.index
    inertia = pd.Series(
        mfa.explained_inertia_, index=[f"F{i+1}" for i in range(n_comp)]
    )

    runtime = time() - start
    return {
        "model": mfa,
        "inertia": inertia,
        "embeddings": rows,
        "runtime_s": runtime,
    }


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

def run_all_factor_methods(
    df_active: pd.DataFrame,
    quant_vars: List[str],
    qual_vars: List[str],
    groups: Optional[Sequence[Sequence[str]]] = None,
) -> Dict[str, Dict[str, object]]:
    """Run all factorial methods and return their results in a dictionary."""
    results = {
        "PCA": run_pca(df_active, quant_vars),
        "MCA": run_mca(df_active, qual_vars),
        "FAMD": run_famd(df_active, quant_vars, qual_vars),
    }
    if groups is not None:
        results["MFA"] = run_mfa(df_active, groups)
    return results


