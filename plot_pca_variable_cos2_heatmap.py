#!/usr/bin/env python3
"""Generate a heatmap of PCA variable cos² values.

This script loads the dataset defined in ``config.yaml`` (or another
configuration file) and performs a Principal Component Analysis on the
quantitative variables. It computes the cos² of each variable on the
first ``n_axes`` components and saves a heatmap ``pca_cos2_heatmap.png``
in the current directory.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import yaml
import numpy as np

import phase4_functions as pf

logger = logging.getLogger(__name__)


def load_config(path: Path) -> Mapping[str, Any]:
    """Return the configuration mapping from ``path``."""
    with open(path, "r", encoding="utf-8") as fh:
        if path.suffix.lower() in {".yaml", ".yml"}:
            return yaml.safe_load(fh)
        return json.load(fh)


def variable_cos2(pca: Any, variables: Sequence[str], n_axes: int) -> pd.DataFrame:
    """Return the cos² of ``variables`` for the first ``n_axes`` components."""
    comps = np.asarray(pca.components_[:n_axes], dtype=float)
    eig = np.asarray(pca.explained_variance_[:n_axes], dtype=float)
    loadings = comps.T * np.sqrt(eig)
    cos2 = loadings**2 / eig
    df = pd.DataFrame(cos2, index=variables, columns=[f"Dim{i+1}" for i in range(n_axes)])
    return df


def plot_heatmap(df: pd.DataFrame, output: Path) -> Path:
    """Save ``df`` as a heatmap to ``output``."""
    fig, ax = plt.subplots(figsize=(12, 6), dpi=200)
    sns.heatmap(df, cmap="coolwarm", vmin=0, vmax=1, ax=ax, cbar=True)
    ax.set_xlabel("Axe factoriel")
    ax.set_ylabel("Variable")
    ax.set_title("ACP – cos² des variables")
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)
    return output


def run(config: Mapping[str, Any], *, dataset: str, n_axes: int = 4) -> Path:
    """Compute cos² and generate the heatmap for ``dataset``."""
    datasets = pf.load_datasets(config, ignore_schema=bool(config.get("ignore_schema", False)))
    if dataset not in datasets:
        raise KeyError(f"dataset '{dataset}' not found")

    df_prep = pf.prepare_data(datasets[dataset], exclude_lost=bool(config.get("exclude_lost", True)))
    df_active, quant, qual = pf.select_variables(df_prep)
    df_active = pf.handle_missing_values(df_active, quant, qual)

    if not quant:
        raise ValueError("No quantitative variables available for PCA")

    res = pf.run_pca(df_active, quant, n_components=max(n_axes, len(quant)))
    pca = res["model"]
    cos2_df = variable_cos2(pca, quant, n_axes)
    return plot_heatmap(cos2_df, Path("pca_cos2_heatmap.png"))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="PCA variable cos² heatmap")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    parser.add_argument("--dataset", default="raw", help="Dataset to process")
    parser.add_argument("--axes", type=int, default=4, help="Number of axes to display")
    args = parser.parse_args(argv)

    cfg = load_config(Path(args.config))
    path = run(cfg, dataset=args.dataset, n_axes=args.axes)
    print(path)


if __name__ == "__main__":  # pragma: no cover - CLI
    logging.basicConfig(level=logging.INFO)
    main()
