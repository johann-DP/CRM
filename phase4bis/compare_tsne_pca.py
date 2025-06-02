#!/usr/bin/env python3
"""Compare PCA and t-SNE projections on the CRM dataset."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Mapping

import matplotlib.pyplot as plt
import seaborn as sns
import yaml

from phase4.functions import (
    load_datasets,
    prepare_data,
    select_variables,
    handle_missing_values,
    run_pca,
    run_tsne,
)


def _load_config(path: Path) -> Mapping[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        if path.suffix.lower() in {".yaml", ".yml"}:
            return yaml.safe_load(fh)
        return json.load(fh)


def main() -> None:
    parser = argparse.ArgumentParser(description="Scatter plot of PCA vs t-SNE")
    parser.add_argument("--config", required=True, help="Path to config file")
    parser.add_argument("--dataset", default="cleaned_3_univ", help="Dataset name")
    parser.add_argument("--output", default="tsne_vs_pca.png", help="Output PNG")
    parser.add_argument("--csv", help="Optional CSV export of t-SNE coordinates")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    cfg = _load_config(Path(args.config))
    data = load_datasets(cfg, ignore_schema=True)
    df = data.get(args.dataset)
    if df is None:
        raise KeyError(f"dataset '{args.dataset}' not found")

    df_prep = prepare_data(df, exclude_lost=True)
    df_active, quant, qual = select_variables(df_prep, min_modalite_freq=5)
    df_active = handle_missing_values(df_active, quant, qual)

    pca_res = run_pca(df_active, quant_vars=quant, n_components=2)
    tsne_res = run_tsne(df_active, n_components=2)

    color_var = None
    if qual:
        color_var = qual[0]

    fig, axes = plt.subplots(1, 2, figsize=(12, 6), dpi=200)

    emb_pca = pca_res["embeddings"]
    emb_tsne = tsne_res["embeddings"]

    if color_var and color_var in df_active.columns:
        cats = df_active[color_var].astype("category")
        palette = sns.color_palette("deep", len(cats.cat.categories))
        for color, cat in zip(palette, cats.cat.categories):
            mask = cats == cat
            axes[0].scatter(
                emb_pca.loc[mask, emb_pca.columns[0]],
                emb_pca.loc[mask, emb_pca.columns[1]],
                s=10,
                alpha=0.6,
                color=color,
                label=str(cat),
            )
        for color, cat in zip(palette, cats.cat.categories):
            mask = cats == cat
            axes[1].scatter(
                emb_tsne.loc[mask, emb_tsne.columns[0]],
                emb_tsne.loc[mask, emb_tsne.columns[1]],
                s=10,
                alpha=0.6,
                color=color,
                label=str(cat),
            )
        axes[0].legend(title=color_var, bbox_to_anchor=(1.05, 1), loc="upper left")
    else:
        axes[0].scatter(emb_pca.iloc[:, 0], emb_pca.iloc[:, 1], s=10, alpha=0.6)
        axes[1].scatter(emb_tsne.iloc[:, 0], emb_tsne.iloc[:, 1], s=10, alpha=0.6)

    var_exp = pca_res["model"].explained_variance_ratio_[:2] * 100
    axes[0].set_title(f"ACP (lin\xe9aire) ({var_exp[0]:.1f}% + {var_exp[1]:.1f}%)")
    axes[1].set_title("t-SNE (non-lin\xe9aire)")

    axes[0].set_xlabel(emb_pca.columns[0])
    axes[0].set_ylabel(emb_pca.columns[1])
    axes[1].set_xlabel(emb_tsne.columns[0])
    axes[1].set_ylabel(emb_tsne.columns[1])

    fig.suptitle("Comparaison des projections \u2013 ACP vs t-SNE")
    fig.tight_layout()
    fig.savefig(args.output, dpi=200)
    plt.show()

    if args.csv:
        out = emb_tsne.rename(columns={
            emb_tsne.columns[0]: "Dim1_tSNE",
            emb_tsne.columns[1]: "Dim2_tSNE",
        })
        out.to_csv(args.csv, index_label="index")


if __name__ == "__main__":  # pragma: no cover
    main()
