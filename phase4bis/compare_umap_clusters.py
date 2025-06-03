#!/usr/bin/env python3
"""UMAP and clustering comparison between raw and prepared datasets.

This script loads two datasets (raw and cleaned/enriched), applies the same
UMAP parameters and agglomerative clustering, then visualises the results
side by side. It also prints silhouette scores for both segmentations.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score

from phase4.functions import run_umap, silhouette_score_safe


_DEF_NEIGHBORS = 15
_DEF_MIN_DIST = 0.1
_DEF_K = 5


def _read(path: Path) -> pd.DataFrame:
    """Read a CSV or Excel file into a DataFrame."""
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        df = pd.read_excel(path)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _basic_prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Minimal preparation: drop Code Analytique and cast categories."""
    df = df.drop(columns=["Code Analytique"], errors="ignore").copy()
    for col in df.select_dtypes(include="object"):
        df[col] = df[col].astype("category")
    return df.dropna()


def _umap_clusters(df: pd.DataFrame, *, k: int, n_neighbors: int, min_dist: float) -> tuple[pd.DataFrame, pd.Series]:
    """Return UMAP embedding and agglomerative labels for ``df``."""
    res = run_umap(df, n_components=2, n_neighbors=n_neighbors, min_dist=min_dist)
    emb = res["embeddings"]
    labels = AgglomerativeClustering(n_clusters=k).fit_predict(emb.values)
    return emb, pd.Series(labels, index=emb.index, name="cluster")


def compare_datasets(
    df_raw: pd.DataFrame,
    df_prep: pd.DataFrame,
    *,
    k: int = _DEF_K,
    n_neighbors: int = _DEF_NEIGHBORS,
    min_dist: float = _DEF_MIN_DIST,
    output: Path | None = None,
) -> plt.Figure:
    """Create a side-by-side UMAP + clustering comparison figure."""
    df_raw_p = _basic_prepare(df_raw)
    df_prep_p = _basic_prepare(df_prep)

    emb_raw, lab_raw = _umap_clusters(df_raw_p, k=k, n_neighbors=n_neighbors, min_dist=min_dist)
    emb_prep, lab_prep = _umap_clusters(df_prep_p, k=k, n_neighbors=n_neighbors, min_dist=min_dist)

    sil_raw = silhouette_score_safe(emb_raw.values, lab_raw.values)
    sil_prep = silhouette_score_safe(emb_prep.values, lab_prep.values)
    rand = adjusted_rand_score(lab_raw.values, lab_prep.values)

    print(f"Silhouette brut: {sil_raw:.3f}")
    print(f"Silhouette préparé: {sil_prep:.3f}")
    print(f"ARI brut vs préparé: {rand:.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 6), dpi=200)
    palette_raw = sns.color_palette("deep", len(lab_raw.unique()))
    palette_prep = sns.color_palette("deep", len(lab_prep.unique()))

    sns.scatterplot(x=emb_raw.iloc[:, 0], y=emb_raw.iloc[:, 1], hue=lab_raw.astype(str),
                    palette=palette_raw, s=10, ax=axes[0], legend=True)
    axes[0].set_title("Données brutes – UMAP + Clusters")
    axes[0].set_xlabel(emb_raw.columns[0])
    axes[0].set_ylabel(emb_raw.columns[1])
    axes[0].legend(title="cluster")

    sns.scatterplot(x=emb_prep.iloc[:, 0], y=emb_prep.iloc[:, 1], hue=lab_prep.astype(str),
                    palette=palette_prep, s=10, ax=axes[1], legend=True)
    axes[1].set_title("Données enrichies – UMAP + Clusters")
    axes[1].set_xlabel(emb_prep.columns[0])
    axes[1].set_ylabel(emb_prep.columns[1])
    axes[1].legend(title="cluster")

    fig.suptitle("Comparaison de la segmentation avant/après préparation des données")
    fig.tight_layout()

    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output)
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description="UMAP clustering comparison")
    parser.add_argument("--raw", required=True, help="Path to raw dataset (CSV or Excel)")
    parser.add_argument("--prepared", required=True, help="Path to prepared dataset (CSV or Excel)")
    parser.add_argument("--k", type=int, default=_DEF_K, help="Number of clusters")
    parser.add_argument("--n_neighbors", type=int, default=_DEF_NEIGHBORS, help="UMAP n_neighbors")
    parser.add_argument("--min_dist", type=float, default=_DEF_MIN_DIST, help="UMAP min_dist")
    parser.add_argument("--output", default="umap_comparison.png", help="Output PNG path")
    args = parser.parse_args()

    df_raw = _read(Path(args.raw))
    df_prep = _read(Path(args.prepared))

    compare_datasets(df_raw, df_prep, k=args.k, n_neighbors=args.n_neighbors,
                     min_dist=args.min_dist, output=Path(args.output))


if __name__ == "__main__":  # pragma: no cover
    main()
