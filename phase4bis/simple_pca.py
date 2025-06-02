#!/usr/bin/env python3
"""Basic PCA analysis of the CRM dataset.

This script loads the CRM data from a CSV file, removes the
``Code Analytique`` column when present and performs a Principal
Component Analysis on all numerical variables. The resulting scree
plot and the variance table are saved next to the script.

Usage:
    python simple_pca.py [path/to/CRM_data.csv]
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def load_data(path: Path) -> pd.DataFrame:
    """Read the CSV file and drop the ``Code Analytique`` column."""
    df = pd.read_csv(path)
    if "Code Analytique" in df.columns:
        df = df.drop(columns=["Code Analytique"])
    return df


def run_pca(df: pd.DataFrame) -> pd.DataFrame:
    """Standardise quantitative columns and fit PCA."""
    num_cols = df.select_dtypes(include="number").columns.tolist()
    X = StandardScaler().fit_transform(df[num_cols])
    pca = PCA().fit(X)
    eig_vals = pca.explained_variance_
    var_ratio = pca.explained_variance_ratio_
    cum_var = var_ratio.cumsum()

    table = pd.DataFrame(
        {
            "valeur_propre": eig_vals,
            "variance_expliquee_pct": var_ratio * 100,
            "variance_cumulee_pct": cum_var * 100,
        },
        index=[f"F{i+1}" for i in range(len(eig_vals))],
    )
    return table


def plot_scree(var_ratio: pd.Series, cum_var: pd.Series, output: Path) -> None:
    """Generate and save the scree plot."""
    sns.set_palette("deep")
    fig, ax = plt.subplots(figsize=(12, 6), dpi=200)
    comps = range(1, len(var_ratio) + 1)
    ax.bar(comps, var_ratio * 100, color=sns.color_palette("deep")[0], edgecolor="black")
    ax.plot(comps, cum_var * 100, "-o", color=sns.color_palette("deep")[1])

    for x, y in zip(comps, var_ratio * 100):
        ax.text(x, y + 0.5, f"{y:.1f}%", ha="center", va="bottom", fontsize=8)

    ax.set_xlabel("Composante principale")
    ax.set_ylabel("Variance expliquée (%)")
    ax.set_title("ACP – Éboulis des valeurs propres")
    ax.set_xticks(list(comps))

    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def main(argv: list[str]) -> None:
    data_path = Path(argv[1]) if len(argv) > 1 else Path("CRM_data.csv")
    df = load_data(data_path)
    table = run_pca(df)

    table.to_csv("ACP_variance_expliquee.csv", index_label="composante")
    plot_scree(
        table["variance_expliquee_pct"] / 100,
        table["variance_cumulee_pct"] / 100,
        Path("pca_scree_plot.png"),
    )


if __name__ == "__main__":
    main(sys.argv)
