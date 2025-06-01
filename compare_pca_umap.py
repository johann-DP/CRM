#!/usr/bin/env python3
"""Compare PCA and UMAP projections on the CRM dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import yaml

from phase4_functions import (
    load_datasets,
    prepare_data,
    select_variables,
    handle_missing_values,
    run_pca,
    run_umap,
    optimize_clusters,
)


_DEF_NEIGHBORS = 15


def _load_config(path: Path) -> Mapping[str, Any]:
    """Return configuration mapping from ``path``."""
    with open(path, "r", encoding="utf-8") as fh:
        if path.suffix.lower() in {".yaml", ".yml"}:
            return yaml.safe_load(fh)
        return json.load(fh)


def main() -> None:
    parser = argparse.ArgumentParser(description="Scatter plot of PCA vs UMAP")
    parser.add_argument("--config", required=True, help="Path to config file")
    parser.add_argument("--dataset", default="raw", help="Dataset name")
    parser.add_argument("--n_neighbors", type=int, default=_DEF_NEIGHBORS,
                        help="UMAP n_neighbors")
    parser.add_argument("--output", default="pca_vs_umap.png", help="Output PNG")
    args = parser.parse_args()

    cfg = _load_config(Path(args.config))
    data = load_datasets(cfg, ignore_schema=True)
    df = data.get(args.dataset)
    if df is None:
        raise KeyError(f"dataset '{args.dataset}' not found")

    df_prep = prepare_data(df, exclude_lost=True)
    df_active, quant, qual = select_variables(df_prep, min_modalite_freq=5)
    df_active = handle_missing_values(df_active, quant, qual)

    pca_res = run_pca(df_active, quant_vars=quant, n_components=2)
    umap_res = run_umap(
        df_active,
        n_components=2,
        n_neighbors=args.n_neighbors,
    )

    emb_pca = pca_res["embeddings"]
    emb_umap = umap_res["embeddings"]

    labels, _best_k, _ = optimize_clusters("kmeans", emb_pca.values)
    labels = pd.Series(labels, index=emb_pca.index, name="cluster")
    palette = sns.color_palette("deep", len(labels.unique()))

    fig, axes = plt.subplots(1, 2, figsize=(12, 6), dpi=200)
    for ax, emb, title in zip(
        axes,
        [emb_pca, emb_umap],
        ["ACP", f"UMAP (n_neighbors={args.n_neighbors})"],
    ):
        for col, lab in zip(palette, sorted(labels.unique())):
            mask = labels == lab
            ax.scatter(
                emb.loc[mask, emb.columns[0]],
                emb.loc[mask, emb.columns[1]],
                s=10,
                alpha=0.6,
                color=col,
                label=str(lab),
            )
        ax.set_xlabel(emb.columns[0])
        ax.set_ylabel(emb.columns[1])
        ax.set_title(title)
    axes[0].legend(title="cluster", bbox_to_anchor=(1.05, 1), loc="upper left")
    fig.suptitle("Comparaison des projections – ACP vs UMAP")
    fig.tight_layout()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.show()


if __name__ == "__main__":  # pragma: no cover
    main()
