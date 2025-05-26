"""Evaluation et comparaison des méthodes de réduction de dimension (Bloc 7)."""

from __future__ import annotations

from typing import Dict, Sequence, Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.manifold import trustworthiness
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path


def dunn_index(X: np.ndarray, labels: np.ndarray) -> float:
    """Calcule l'indice de Dunn pour un partitionnement."""
    from scipy.spatial.distance import pdist, squareform

    if len(np.unique(labels)) < 2:
        return float("nan")

    dist = squareform(pdist(X))
    unique = np.unique(labels)

    intra_diam = []
    min_inter = np.inf

    for i, ci in enumerate(unique):
        idx_i = np.where(labels == ci)[0]
        intra = dist[np.ix_(idx_i, idx_i)].max() if len(idx_i) > 1 else 0.0
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


def _prepare_matrix(df: pd.DataFrame, quant: Sequence[str], qual: Sequence[str]) -> np.ndarray:
    """Encode et centre les colonnes quantitatives et qualitatives."""
    X_num = StandardScaler().fit_transform(df[quant]) if quant else np.empty((len(df), 0))
    try:
        enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # pragma: no cover - older scikit-learn
        enc = OneHotEncoder(handle_unknown="ignore", sparse=False)
    X_cat = enc.fit_transform(df[qual]) if qual else np.empty((len(df), 0))
    return np.hstack([X_num, X_cat]) if quant or qual else np.empty((len(df), 0))


def evaluate_methods(
    results_dict: Dict[str, Dict[str, Any]],
    df_active: pd.DataFrame,
    quant_vars: Sequence[str],
    qual_vars: Sequence[str],
    *,
    n_clusters: int = 3,
) -> pd.DataFrame:
    """Calcule différentes métriques pour comparer les méthodes."""

    rows = []
    n_features = len(quant_vars) + len(qual_vars)

    X_high = _prepare_matrix(df_active, quant_vars, qual_vars)

    for method, res in results_dict.items():
        emb = res.get("embeddings")
        if not isinstance(emb, pd.DataFrame) or emb.empty:
            continue

        inertias = res.get("inertia")
        if inertias is None:
            inert_list: Sequence[float] = []
        elif isinstance(inertias, pd.Series):
            inert_list = inertias.tolist()
        else:
            inert_list = list(inertias)

        kaiser = sum(1 for eig in np.array(inert_list) * n_features if eig > 1)
        cum_var = sum(inert_list) * 100 if inert_list else np.nan

        X = emb.values
        labels = KMeans(n_clusters=n_clusters, random_state=0).fit_predict(X)
        sil = silhouette_score(X, labels)
        dunn = dunn_index(X, labels)

        rt = res.get("runtime_s") if res.get("runtime_s") is not None else res.get("runtime")

        idx = df_active.index.intersection(emb.index)
        Xh = X_high[df_active.index.get_indexer(idx)]
        Xl = emb.loc[idx].values
        T = trustworthiness(Xh, Xl, n_neighbors=10)
        C = trustworthiness(Xl, Xh, n_neighbors=10)

        rows.append(
            {
                "method": method,
                "variance_cumulee_%": cum_var,
                "nb_axes_kaiser": kaiser,
                "silhouette": sil,
                "dunn_index": dunn,
                "trustworthiness": T,
                "continuity": C,
                "runtime_seconds": rt,
            }
        )

    comp_df = pd.DataFrame(rows).set_index("method")
    return comp_df


def plot_methods_heatmap(df_metrics: pd.DataFrame, output_path: str | Path) -> None:
    """Génère une heatmap normalisée des métriques et la sauvegarde."""

    df_norm = df_metrics.copy()
    for col in df_norm.columns:
        cmin = df_norm[col].min()
        cmax = df_norm[col].max()
        if pd.notna(cmin) and pd.notna(cmax) and cmax > cmin:
            df_norm[col] = (df_norm[col] - cmin) / (cmax - cmin)

    fig, ax = plt.subplots(figsize=(12, 6), dpi=200)
    sns.heatmap(
        df_norm,
        annot=df_metrics.round(2),
        cmap="coolwarm",
        vmin=0,
        vmax=1,
        ax=ax,
    )
    ax.set_title("Comparaison des méthodes")
    plt.yticks(rotation=0)
    fig.tight_layout()

    out = Path(output_path)
    out.mkdir(parents=True, exist_ok=True)
    plt.savefig(out / "methods_heatmap.png")
    plt.close(fig)

