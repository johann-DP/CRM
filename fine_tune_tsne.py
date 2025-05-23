#!/usr/bin/env python3
"""Fine-tune t-SNE embeddings for the CRM dataset.

The script loads the cleaned multivariate CSV, performs preprocessing,
optionally reduces the dimensionality with PCA and runs a grid search
over several t-SNE parameters. The best configuration is selected using
the silhouette score after clustering with KMeans. The final model and
embeddings are exported along with basic scatter plots.
"""

from __future__ import annotations

import argparse
import logging
import pickle
from itertools import product
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DEFAULT_INPUT = "/mnt/data/phase3_cleaned_multivariate.csv"
DEFAULT_OUTPUT = "/mnt/data/phase4_output/fine_tuning_tsne"


def load_preprocess(csv_path: str) -> tuple[pd.DataFrame, np.ndarray]:
    """Load the dataset and return the preprocessed matrix."""
    df = pd.read_csv(csv_path)
    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(exclude="number").columns.tolist()

    # Imputation
    df_num = df[num_cols].fillna(df[num_cols].mean())
    df_cat = df[cat_cols].fillna("unknown")

    # Encoding / scaling
    scaler = StandardScaler()
    X_num = scaler.fit_transform(df_num)

    encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
    X_cat = encoder.fit_transform(df_cat)

    X_all = np.hstack([X_num, X_cat])
    return df, X_all


def run_grid_search(X: np.ndarray) -> tuple[Dict[str, int | float | str], np.ndarray, pd.DataFrame]:
    """Evaluate several t-SNE parameter combinations."""
    grid = {
        "perplexity": [5, 10, 20, 30, 50],
        "learning_rate": [10, 100, 200, 500],
        "n_iter": [500, 1000, 2000],
        "metric": ["euclidean", "cosine"],
    }

    results: List[Dict[str, float]] = []
    best_score = -1.0
    best_params: Dict[str, int | float | str] | None = None
    best_emb: np.ndarray | None = None

    for perp, lr, n_it, metric in product(
        grid["perplexity"], grid["learning_rate"], grid["n_iter"], grid["metric"]
    ):
        tsne = TSNE(
            random_state=42,
            init="pca",
            perplexity=perp,
            learning_rate=lr,
            n_iter=n_it,
            metric=metric,
        )
        emb = tsne.fit_transform(X)
        labels = KMeans(n_clusters=6, random_state=42).fit_predict(emb)
        sil = silhouette_score(emb, labels)
        results.append(
            {
                "perplexity": perp,
                "learning_rate": lr,
                "n_iter": n_it,
                "metric": metric,
                "silhouette": sil,
            }
        )
        if sil > best_score:
            best_score = sil
            best_params = {
                "perplexity": perp,
                "learning_rate": lr,
                "n_iter": n_it,
                "metric": metric,
            }
            best_emb = emb

    assert best_params is not None and best_emb is not None
    metrics_df = pd.DataFrame(results)
    return best_params, best_emb, metrics_df


def export_results(
    df: pd.DataFrame,
    embedding: np.ndarray,
    params: Dict[str, int | float | str],
    metrics_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Save model, embeddings and scatter plots."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save model
    tsne_model = TSNE(random_state=42, init="pca", **params)
    tsne_model.fit(embedding)
    with open(output_dir / "tsne_model.pkl", "wb") as fh:
        pickle.dump(tsne_model, fh)

    # Save embeddings with labels
    label_cols_all = [
        "Catégories",
        "Entité opérationnelle",
        "Pilier",
        "Sous-catégorie",
        "Statut commercial",
        "Statut production",
        "Type opportunité",
    ]
    label_cols = [c for c in label_cols_all if c in df.columns]

    emb_df = pd.DataFrame(embedding[:, :2], columns=["TSNE1", "TSNE2"])
    for col in label_cols:
        emb_df[col] = df[col].values
    emb_df.to_csv(output_dir / "tsne_embeddings.csv", index=False)

    # Scatter plots
    for col in label_cols:
        plt.figure(figsize=(8, 6), dpi=200)
        sns.scatterplot(
            data=emb_df,
            x="TSNE1",
            y="TSNE2",
            hue=col,
            s=20,
            alpha=0.7,
            palette="tab10",
        )
        plt.title(f"t-SNE colored by {col}")
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", title=col)
        plt.tight_layout()
        name = col.replace(" ", "_").replace("/", "_").lower()
        plt.savefig(output_dir / f"tsne_{name}.png")
        plt.close()

    with open(output_dir / "tsne_metrics.txt", "w", encoding="utf-8") as fh:
        fh.write(metrics_df.sort_values("silhouette", ascending=False).to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune t-SNE on CRM data")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="CSV file")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output directory")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    out_dir = Path(args.output)

    logging.info("Loading and preprocessing data")
    df, X_all = load_preprocess(args.input)

    logging.info("Running PCA for initialization")
    n_components = min(50, X_all.shape[1])
    X_pca = PCA(n_components=n_components, random_state=42).fit_transform(X_all)

    logging.info("Starting t-SNE grid search")
    best_params, best_emb, metrics_df = run_grid_search(X_pca)
    logging.info("Best params: %s", best_params)

    logging.info("Refitting t-SNE with best parameters")
    final_tsne = TSNE(random_state=42, init="pca", **best_params)
    final_emb = final_tsne.fit_transform(X_pca)

    export_results(df, final_emb, best_params, metrics_df, out_dir)
    logging.info("t-SNE fine-tuning complete")


if __name__ == "__main__":
    main()
