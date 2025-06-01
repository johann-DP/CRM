import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

from phase4_functions import dunn_index


ALGOS = {
    "kmeans": KMeans,
    "agglomerative": AgglomerativeClustering,
}


def evaluate_clustering(X: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    """Return clustering quality metrics for ``labels``."""
    if len(np.unique(labels)) < 2:
        return {
            "dunn_index": float("nan"),
            "calinski_harabasz": float("nan"),
            "davies_bouldin": float("nan"),
            "silhouette": float("nan"),
        }
    sil = float(silhouette_score(X, labels))
    dunn = dunn_index(X, labels, sample_size=1000)
    ch = float(calinski_harabasz_score(X, labels))
    db = float(davies_bouldin_score(X, labels))
    return {
        "dunn_index": dunn,
        "calinski_harabasz": ch,
        "davies_bouldin": db,
        "silhouette": sil,
    }


def run_algorithms(X: np.ndarray, k_values: list[int]) -> pd.DataFrame:
    records: list[dict[str, float]] = []
    for name, cls in ALGOS.items():
        for k in k_values:
            if k < 2 or k >= len(X):
                continue
            labels = cls(n_clusters=k).fit_predict(X)
            metrics = evaluate_clustering(X, labels)
            metrics["solution"] = f"{name}_k{k}"
            records.append(metrics)
    return pd.DataFrame.from_records(records).set_index("solution")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute clustering quality indices")
    parser.add_argument("input", help="Path to CSV file with numeric data")
    parser.add_argument("--k", type=int, nargs="*", default=[3, 4, 5], help="Values of k to test")
    parser.add_argument("--output", default="clustering_quality_indices.csv", help="Output CSV file")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    X = df.to_numpy(float)

    table = run_algorithms(X, args.k)
    table.to_csv(args.output)


if __name__ == "__main__":  # pragma: no cover
    main()
