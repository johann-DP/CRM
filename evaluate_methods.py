"""Metrics to compare dimensionality reduction methods.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.manifold import trustworthiness
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler



def dunn_index(X: np.ndarray, labels: np.ndarray) -> float:
    """Compute the Dunn index of a clustering.

    Parameters
    ----------
    X:
        Coordinates of the points.
    labels:
        Cluster labels for each point.

    Returns
    -------
    float
        Dunn index (higher is better). ``NaN`` if undefined.
    """
    from scipy.spatial.distance import pdist, squareform

    if len(np.unique(labels)) < 2:
        return float("nan")

    dist = squareform(pdist(X))
    unique = np.unique(labels)

    intra_diam = []
    min_inter = np.inf

    for i, ci in enumerate(unique):
        idx_i = np.where(labels == ci)[0]
        if len(idx_i) > 1:
            intra = dist[np.ix_(idx_i, idx_i)].max()
        else:
            intra = 0.0
        intra_diam.append(intra)

        for cj in unique[i + 1:]:
            idx_j = np.where(labels == cj)[0]
            inter = dist[np.ix_(idx_i, idx_j)].min()
            if inter < min_inter:
                min_inter = inter

    max_intra = max(intra_diam)
    if max_intra == 0:
        return float("nan")
    return float(min_inter / max_intra)


def evaluate_methods(
    results_dict: Dict[str, Dict[str, Any]],
    df_active: pd.DataFrame,
    quant_vars: Sequence[str],
    qual_vars: Sequence[str],
    *,
    n_clusters: int = 3,
) -> pd.DataFrame:
    """Compute comparison metrics for each dimensionality reduction method.

    Parameters
    ----------
    results_dict:
        Mapping of method name to a dictionary containing at least ``embeddings``
        as a DataFrame. Optionally ``inertia`` (list or Series) and
        ``runtime_s`` or ``runtime``.
    df_active:
        Original high dimensional dataframe.
    quant_vars:
        Names of quantitative variables in ``df_active``.
    qual_vars:
        Names of qualitative variables in ``df_active``.
    n_clusters:
        Number of clusters for the silhouette and Dunn metrics.

    Returns
    -------
    pandas.DataFrame
        Metrics table indexed by method name.
    """
    rows = []
    n_features = len(quant_vars) + len(qual_vars)

    for method, info in results_dict.items():
        inertias = info.get("inertia")
        if inertias is None:
            inertias = []
        if isinstance(inertias, pd.Series):
            inertias = inertias.tolist()
        inertias = list(inertias)

        kaiser = int(sum(1 for eig in np.array(inertias) * n_features if eig > 1))
        cum_inertia = float(sum(inertias) * 100) if inertias else np.nan

        X_low = info["embeddings"].values
        labels = KMeans(n_clusters=n_clusters, random_state=None).fit_predict(X_low)
        if len(labels) <= n_clusters or len(set(labels)) < 2:
            sil = float("nan")
            dunn = float("nan")
        else:
            sil = float(silhouette_score(X_low, labels))
            dunn = dunn_index(X_low, labels)

        try:
            enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        except TypeError:  # pragma: no cover - older scikit-learn
            enc = OneHotEncoder(sparse=False, handle_unknown="ignore")
        X_num = StandardScaler().fit_transform(
            df_active.loc[info["embeddings"].index, quant_vars]
        )
        X_cat = (
            enc.fit_transform(df_active.loc[info["embeddings"].index, qual_vars])
            if qual_vars
            else np.empty((len(X_low), 0))
        )
        X_high = np.hstack([X_num, X_cat])
        k_nn = min(10, max(1, len(X_high) // 2))
        if k_nn >= len(X_high) / 2:
            T = float("nan")
            C = float("nan")
        else:
            T = float(trustworthiness(X_high, X_low, n_neighbors=k_nn))
            C = float(trustworthiness(X_low, X_high, n_neighbors=k_nn))

        runtime = info.get("runtime_seconds") or info.get("runtime_s") or info.get("runtime")

        rows.append(
            {
                "method": method,
                "variance_cumulee_%": cum_inertia,
                "nb_axes_kaiser": kaiser,
                "silhouette": sil,
                "dunn_index": dunn,
                "trustworthiness": T,
                "continuity": C,
                "runtime_seconds": runtime,
            }
        )

    df_metrics = pd.DataFrame(rows).set_index("method")
    return df_metrics


def plot_methods_heatmap(df_metrics: pd.DataFrame, output_path: str | Path) -> None:
    """Plot a normalized heatmap of ``df_metrics``.

    Parameters
    ----------
    df_metrics:
        DataFrame as returned by :func:`evaluate_methods`.
    output_path:
        Directory where ``methods_heatmap.png`` will be saved.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    output = Path(output_path)
    output.mkdir(parents=True, exist_ok=True)

    df_norm = df_metrics.copy()
    for col in df_norm.columns:
        cmin, cmax = df_norm[col].min(), df_norm[col].max()
        if pd.isna(cmin) or cmax == cmin:
            df_norm[col] = 0.0
        else:
            df_norm[col] = (df_norm[col] - cmin) / (cmax - cmin)

    plt.figure(figsize=(8, 0.4 * len(df_norm) + 2), dpi=200)
    ax = sns.heatmap(
        df_norm,
        annot=df_metrics,
        fmt=".2f",
        cmap="viridis",
        vmin=0,
        vmax=1,
    )
    ax.set_title("Comparaison des méthodes")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output / "methods_heatmap.png")
    plt.close()

