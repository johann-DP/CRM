"""Utility functions for non-linear dimensionality reduction.

This module provides thin wrappers around UMAP, PHATE and PaCMAP with
basic preprocessing and timing. Each function returns a dictionary with
the fitted model/operator, the resulting embeddings and runtime.
"""

from __future__ import annotations

import time
from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    import umap
except Exception:  # pragma: no cover - optional dependency
    umap = None  # type: ignore

try:
    import phate
except Exception:  # pragma: no cover - optional dependency
    phate = None  # type: ignore

try:
    import pacmap
except Exception:  # pragma: no cover - optional dependency
    pacmap = None  # type: ignore


# ---------------------------------------------------------------------------
# Helpers

def _to_numeric_matrix(df: pd.DataFrame) -> np.ndarray:
    """Scale numeric columns and one-hot encode categoricals."""
    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = [c for c in df.columns if c not in num_cols]

    X_num = StandardScaler().fit_transform(df[num_cols]) if num_cols else np.empty((len(df), 0))
    try:
        enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # pragma: no cover - older scikit-learn
        enc = OneHotEncoder(handle_unknown="ignore", sparse=False)
    X_cat = enc.fit_transform(df[cat_cols]) if cat_cols else np.empty((len(df), 0))

    return np.hstack([X_num, X_cat]) if cat_cols or num_cols else np.empty((len(df), 0))


# ---------------------------------------------------------------------------
# Main functions

def run_umap(
    df_active: pd.DataFrame,
    n_components: int = 2,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int | None = 42,
) -> Dict[str, Any]:
    """Run UMAP on ``df_active`` and return embeddings with runtime."""

    if umap is None:
        raise ImportError("umap-learn is required for run_umap")

    X = _to_numeric_matrix(df_active)

    start = time.time()
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state,
    )
    embedding = reducer.fit_transform(X)
    runtime = time.time() - start

    emb_df = pd.DataFrame(
        embedding,
        index=df_active.index,
        columns=[f"UMAP{i + 1}" for i in range(n_components)],
    )

    return {
        "model": reducer,
        "embeddings": emb_df,
        "runtime": runtime,
        "params": {
            "n_components": n_components,
            "n_neighbors": n_neighbors,
            "min_dist": min_dist,
        },
    }


def run_phate(
    df_active: pd.DataFrame,
    n_components: int = 2,
    k: int = 15,
    a: int = 40,
    random_state: int | None = 42,
) -> Dict[str, Any]:
    """Run PHATE on ``df_active`` and return embeddings with runtime."""

    if phate is None:
        raise ImportError("phate is required for run_phate")

    X = _to_numeric_matrix(df_active)

    start = time.time()
    op = phate.PHATE(
        n_components=n_components,
        k=k,
        a=a,
        random_state=random_state,
        n_jobs=-1,
    )
    embedding = op.fit_transform(X)
    runtime = time.time() - start

    emb_df = pd.DataFrame(
        embedding,
        index=df_active.index,
        columns=[f"PHATE{i + 1}" for i in range(n_components)],
    )

    return {
        "model": op,
        "embeddings": emb_df,
        "runtime": runtime,
        "params": {"n_components": n_components, "k": k, "a": a},
    }


def run_pacmap(
    df_active: pd.DataFrame,
    n_components: int = 2,
    n_neighbors: int = 10,
    random_state: int | None = 42,
) -> Dict[str, Any]:
    """Run PaCMAP on ``df_active`` and return embeddings with runtime."""

    if pacmap is None:
        raise ImportError("pacmap is required for run_pacmap")

    X = _to_numeric_matrix(df_active)

    start = time.time()
    model = pacmap.PaCMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        random_state=random_state,
        init="pca",
    )
    embedding = model.fit_transform(X)
    runtime = time.time() - start

    emb_df = pd.DataFrame(
        embedding,
        index=df_active.index,
        columns=[f"PACMAP{i + 1}" for i in range(n_components)],
    )

    return {
        "model": model,
        "embeddings": emb_df,
        "runtime": runtime,
        "params": {"n_components": n_components, "n_neighbors": n_neighbors},
    }


# ---------------------------------------------------------------------------
# Convenience aggregator

def run_all_nonlin(df_active: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """Run UMAP, PHATE and PaCMAP in sequence.

    Each method is executed with default parameters. Errors due to missing
    optional dependencies are caught and reported.
    """

    results: Dict[str, Dict[str, Any]] = {}

    for name, func in ("umap", run_umap), ("phate", run_phate), ("pacmap", run_pacmap):
        try:
            results[name] = func(df_active)
        except Exception as exc:  # pragma: no cover - missing deps
            results[name] = {"error": str(exc)}

    return results

