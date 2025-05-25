#!/usr/bin/env python3
"""Fine-tuning UMAP on the cleaned CRM dataset."""

from __future__ import annotations

import argparse
import logging
import pickle
from pathlib import Path
import json

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import umap

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine tune UMAP")
    parser.add_argument("--input", required=True, help="Cleaned multivariate CSV")
    parser.add_argument("--output", required=True, help="Output directory")
    return parser.parse_args()

DATA_PATH = Path()
OUTPUT_DIR = Path()
RANDOM_STATE = None


def setup_logger() -> logging.Logger:
    logger = logging.getLogger("fine_tuning_umap")
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = "%(asctime)s - %(levelname)s - %(message)s"
        handler.setFormatter(logging.Formatter(fmt))
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


def load_and_preprocess(df: pd.DataFrame) -> np.ndarray:
    """Impute, encode and scale the dataframe."""
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = df.select_dtypes(exclude="number").columns.tolist()

    num_imputer = SimpleImputer(strategy="mean")
    cat_imputer = SimpleImputer(strategy="most_frequent")
    X_num = num_imputer.fit_transform(df[numeric_cols])
    X_cat = cat_imputer.fit_transform(df[categorical_cols])
    for i, col in enumerate(categorical_cols):
        if df[col].dtype == "bool":
            X_cat[:, i] = X_cat[:, i].astype(str)

    try:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    except TypeError:  # for older scikit-learn
        encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
    X_cat = encoder.fit_transform(X_cat)

    scaler = StandardScaler()
    X_num = scaler.fit_transform(X_num)

    X = np.hstack([X_num, X_cat])
    return X


def grid_search_umap(X: np.ndarray, logger: logging.Logger):
    param_grid = {
        "n_neighbors": [5, 10, 15, 30],
        "min_dist": [0.1, 0.3, 0.5],
        "metric": ["euclidean", "cosine"],
    }
    results = []
    best = None
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for nn in param_grid["n_neighbors"]:
        for md in param_grid["min_dist"]:
            for metric in param_grid["metric"]:
                logger.info(
                    "Training UMAP: n_neighbors=%d, min_dist=%.2f, metric=%s",
                    nn,
                    md,
                    metric,
                )
                kwargs = dict(n_neighbors=nn, min_dist=md, metric=metric)
                if RANDOM_STATE is not None:
                    kwargs["random_state"] = RANDOM_STATE
                reducer = umap.UMAP(**kwargs)
                emb = reducer.fit_transform(X)
                km_rs = RANDOM_STATE if RANDOM_STATE is not None else 0
                labels = KMeans(n_clusters=5, random_state=km_rs).fit_predict(emb)
                score = silhouette_score(emb, labels)
                logger.info("Silhouette: %.3f", score)
                results.append({
                    "n_neighbors": nn,
                    "min_dist": md,
                    "metric": metric,
                    "silhouette": score,
                })
                emb_path = OUTPUT_DIR / f"embeddings_nn{nn}_dist{md}_metric{metric}.csv"
                pd.DataFrame(emb, columns=["UMAP1", "UMAP2"]).to_csv(emb_path, index=False)
                if best is None or score > best[0]:
                    best = (score, nn, md, metric)

    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_DIR / "grid_search_scores.csv", index=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x="silhouette", y=results_df.index, data=results_df, orient="h")
    plt.yticks(
        results_df.index,
        [f"nn={r['n_neighbors']}, md={r['min_dist']}, {r['metric']}" for r in results],
    )
    plt.xlabel("Silhouette score")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "silhouette_scores.png")
    plt.close()

    return best


def export_scatter(embedding: np.ndarray, df: pd.DataFrame, column: str):
    if column not in df.columns:
        return
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=embedding[:, 0], y=embedding[:, 1], hue=df[column], s=10, alpha=0.7)
    plt.title(f"UMAP colored by {column}")
    plt.tight_layout()
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.savefig(OUTPUT_DIR / f"umap_{column}.png")
    plt.close()


def main() -> None:
    args = parse_args()
    global DATA_PATH, OUTPUT_DIR
    DATA_PATH = Path(args.input)
    OUTPUT_DIR = Path(args.output)

    logger = setup_logger()
    if not DATA_PATH.exists():
        logger.error("File not found: %s", DATA_PATH)
        return
    df = pd.read_csv(DATA_PATH)
    logger.info("Dataset loaded: %d rows x %d columns", df.shape[0], df.shape[1])
    X = load_and_preprocess(df)
    best = grid_search_umap(X, logger)
    if best is None:
        logger.error("No UMAP result")
        return
    score, nn, md, metric = best
    logger.info(
        "Best parameters: n_neighbors=%d, min_dist=%.2f, metric=%s (silhouette=%.3f)",
        nn,
        md,
        metric,
        score,
    )
    kwargs = dict(n_neighbors=nn, min_dist=md, metric=metric)
    if RANDOM_STATE is not None:
        kwargs["random_state"] = RANDOM_STATE
    final_model = umap.UMAP(**kwargs)
    embedding = final_model.fit_transform(X)
    pd.DataFrame(embedding, columns=["UMAP1", "UMAP2"]).to_csv(
        OUTPUT_DIR / "umap_embeddings.csv", index=False
    )
    with open(OUTPUT_DIR / "umap_model.pkl", "wb") as f:
        pickle.dump(final_model, f)

    best = {
        "method": "UMAP",
        "params": {
            "n_neighbors": int(nn),
            "min_dist": float(md),
            "metric": metric,
            "n_components": 2,
        },
    }
    with open(OUTPUT_DIR / "best_params.json", "w", encoding="utf-8") as fh:
        json.dump(best, fh, indent=2)

    for col in ["Pilier", "Sous-catégorie", "Catégorie", "Statut commercial"]:
        export_scatter(embedding, df, col)

    logger.info("UMAP fine-tuning complete")


if __name__ == "__main__":
    main()
