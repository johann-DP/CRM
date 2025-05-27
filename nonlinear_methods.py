"""Non-linear dimensionality reduction utilities.

This module re-implements the UMAP, PHATE and PaCMAP wrappers that were
previously scattered across several scripts. The functions are self-contained
and do not depend on ``phase4v2.py`` or the fine-tuning scripts so that those
files can be removed without breaking the pipeline.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:  # optional dependency
    import umap  # type: ignore
except Exception:  # pragma: no cover - optional dependency may not be present
    umap = None

try:  # optional dependencies
    import phate  # type: ignore
except Exception:  # pragma: no cover - optional dependency may not be present
    phate = None

try:
    import pacmap  # type: ignore
except Exception:  # pragma: no cover - optional dependency may not be present
    pacmap = None


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _encode_mixed(df: pd.DataFrame) -> np.ndarray:
    """Return a numeric matrix from ``df`` with scaling and one-hot encoding."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(exclude="number").columns.tolist()

    X_num = (
        StandardScaler().fit_transform(df[numeric_cols]) if numeric_cols else np.empty((len(df), 0))
    )

    if cat_cols:
        try:
            enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        except TypeError:  # pragma: no cover - older scikit-learn
            enc = OneHotEncoder(sparse=False, handle_unknown="ignore")
        X_cat = enc.fit_transform(df[cat_cols])
    else:
        X_cat = np.empty((len(df), 0))

    if X_num.size and X_cat.size:
        X = np.hstack([X_num, X_cat])
    elif X_num.size:
        X = X_num
    else:
        X = X_cat

    # ensure no NaN values
    if np.isnan(X).any():  # pragma: no cover - should not happen
        X = np.nan_to_num(X)

    return X


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_umap(
    df_active: pd.DataFrame,
    n_components: int = 2,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    *,
    metric: str = "euclidean",
    random_state: Optional[int] = None,
) -> Dict[str, Any]:
    """Run UMAP on ``df_active`` and return model and embeddings."""
    if umap is None:  # pragma: no cover - optional dependency may be absent
        logger.warning("UMAP is not installed; skipping")
        return {
            "model": None,
            "embeddings": pd.DataFrame(index=df_active.index),
            "params": {},
        }

    start = time.perf_counter()
    X = _encode_mixed(df_active)

    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
    )
    embedding = reducer.fit_transform(X)
    runtime = time.perf_counter() - start

    cols = [f"Dim{i + 1}" for i in range(n_components)]
    emb_df = pd.DataFrame(embedding, index=df_active.index, columns=cols)

    params = {
        "n_components": n_components,
        "n_neighbors": n_neighbors,
        "min_dist": min_dist,
        "metric": metric,
    }
    return {
        "model": reducer,
        "embeddings": emb_df,
        "params": params,
        "runtime_s": runtime,
    }


def run_phate(
    df_active: pd.DataFrame,
    n_components: int = 2,
    k: int = 15,
    a: int = 40,
    *,
    t: str | int = "auto",
    random_state: Optional[int] = None,
) -> Dict[str, Any]:
    """Run PHATE on ``df_active``.

    Returns an empty result if PHATE is not installed.
    """
    if phate is None:
        logger.warning("PHATE is not installed; skipping")
        return {"model": None, "embeddings": pd.DataFrame(index=df_active.index), "params": {}}

    start = time.perf_counter()
    X = _encode_mixed(df_active)

    op = phate.PHATE(
        n_components=n_components,
        knn=k,
        decay=a,
        t=t,
        n_jobs=-1,
        random_state=random_state,
        verbose=False,
    )
    embedding = op.fit_transform(X)
    runtime = time.perf_counter() - start

    cols = [f"Dim{i + 1}" for i in range(n_components)]
    emb_df = pd.DataFrame(embedding, index=df_active.index, columns=cols)

    params = {"n_components": n_components, "k": k, "a": a, "t": t}
    return {"model": op, "embeddings": emb_df, "params": params, "runtime_s": runtime}


def run_pacmap(
    df_active: pd.DataFrame,
    n_components: int = 2,
    n_neighbors: int = 10,
    *,
    MN_ratio: float = 0.5,
    FP_ratio: float = 2.0,
    num_iters: Tuple[int, int, int] = (10, 10, 10),
    random_state: Optional[int] = None,
) -> Dict[str, Any]:
    """Run PaCMAP on ``df_active``.

    If PaCMAP is unavailable or fails, ``model`` is ``None`` and the embeddings
    DataFrame is empty.
    """
    if pacmap is None:
        logger.warning("PaCMAP is not installed; skipping")
        return {"model": None, "embeddings": pd.DataFrame(index=df_active.index), "params": {}}

    start = time.perf_counter()
    X = _encode_mixed(df_active)

    try:
        params = dict(
            n_components=n_components,
            n_neighbors=n_neighbors,
            MN_ratio=MN_ratio,
            FP_ratio=FP_ratio,
            num_iters=num_iters,
            random_state=random_state,
            verbose=False,
            apply_pca=True,
        )
        model = pacmap.PaCMAP(**params)
        embedding = model.fit_transform(X)
    except Exception as exc:  # pragma: no cover - rare runtime error
        logger.warning("PaCMAP failed: %s", exc)
        return {"model": None, "embeddings": pd.DataFrame(index=df_active.index), "params": {}}

    runtime = time.perf_counter() - start
    cols = [f"Dim{i + 1}" for i in range(n_components)]
    emb_df = pd.DataFrame(embedding, index=df_active.index, columns=cols)
    params.pop("verbose")
    params.pop("apply_pca")
    return {
        "model": model,
        "embeddings": emb_df,
        "params": params,
        "runtime_s": runtime,
    }


def run_all_nonlinear(df_active: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """Execute all non-linear methods on ``df_active`` and collect the results."""
    results: Dict[str, Dict[str, Any]] = {}

    try:
        results["umap"] = run_umap(df_active)
    except Exception as exc:  # pragma: no cover - unexpected runtime failure
        logger.warning("UMAP failed: %s", exc)
        results["umap"] = {"error": str(exc)}

    try:
        results["phate"] = run_phate(df_active)
    except Exception as exc:  # pragma: no cover - unexpected runtime failure
        logger.warning("PHATE failed: %s", exc)
        results["phate"] = {"error": str(exc)}

    try:
        results["pacmap"] = run_pacmap(df_active)
    except Exception as exc:  # pragma: no cover - unexpected runtime failure
        logger.warning("PaCMAP failed: %s", exc)
        results["pacmap"] = {"error": str(exc)}

    return results

