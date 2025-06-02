#!/usr/bin/env python3
"""Compare trustworthiness and continuity across non-linear methods.

This script runs t-SNE, UMAP, PaCMAP and PHATE on the Iris dataset using
predefined optimal parameters from :mod:`phase4.functions`. It computes
trustworthiness and continuity for each method and saves a grouped bar
chart as PNG.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.manifold import trustworthiness

import phase4.functions as pf

_METHODS = {
    "tsne": pf.run_tsne,
    "umap": pf.run_umap,
    "pacmap": pf.run_pacmap,
    "phate": pf.run_phate,
}


def compute_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Return trustworthiness and continuity scores for ``df``."""
    X_high = pf._encode_mixed(df)
    metrics = []
    for name, func in _METHODS.items():
        params = pf.BEST_PARAMS.get(name.upper(), {}).copy()
        res = func(df, **params)
        X_low = res["embeddings"].to_numpy()
        k = min(10, max(1, len(df) // 2))
        T = float(trustworthiness(X_high, X_low, n_neighbors=k))
        C = float(trustworthiness(X_low, X_high, n_neighbors=k))
        metrics.append({"method": name, "trustworthiness": T, "continuity": C})
    return pd.DataFrame(metrics)


def plot_metrics(metrics: pd.DataFrame, output: Path) -> None:
    """Create a grouped bar plot of ``metrics`` and save to ``output``."""
    sns.set(style="whitegrid")
    long_df = metrics.melt("method", var_name="metric", value_name="score")
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    sns.barplot(data=long_df, x="method", y="score", hue="metric",
                palette="deep", ax=ax)
    ax.set_ylabel("Score")
    ax.set_xlabel("Méthode")
    ax.set_title("Comparaison trustworthiness / continuity")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute and plot trustworthiness/continuity comparison"
    )
    parser.add_argument(
        "--output",
        default="trust_continuity_comparison.png",
        help="Path to save the PNG figure",
    )
    args = parser.parse_args()

    iris = load_iris(as_frame=True)
    df = iris.data

    metrics = compute_metrics(df)
    plot_metrics(metrics, Path(args.output))


if __name__ == "__main__":
    main()

