#!/usr/bin/env python3
"""Compare PCA projections across multiple dataset versions."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Mapping

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import yaml

from phase4.functions import (
    load_datasets,
    prepare_data,
    select_variables,
    handle_missing_values,
    run_pca,
)


def _load_config(path: Path) -> Mapping[str, Any]:
    """Return configuration mapping from ``path``."""
    with open(path, "r", encoding="utf-8") as fh:
        if path.suffix.lower() in {".yaml", ".yml"}:
            return yaml.safe_load(fh)
        return json.load(fh)


def main() -> None:
    parser = argparse.ArgumentParser(description="PCA comparison across dataset versions")
    parser.add_argument("--config", required=True, help="Path to YAML/JSON config")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["raw", "cleaned_1", "cleaned_3_univ", "cleaned_3_multi"],
        help="Dataset versions to process",
    )
    parser.add_argument(
        "--color", default="Statut commercial", help="Variable used for colouring"
    )
    parser.add_argument("--output", default="pca_versions.png", help="Output PNG path")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    cfg = _load_config(Path(args.config))
    data = load_datasets(cfg, ignore_schema=True)

    results: dict[str, tuple[pd.DataFrame, pd.Series | None]] = {}
    categories: set[str] = set()

    for name in args.datasets:
        df = data.get(name)
        if df is None:
            logging.warning("Dataset %s missing", name)
            continue

        df_prep = prepare_data(df, exclude_lost=True)
        df_active, quant_vars, qual_vars = select_variables(df_prep, min_modalite_freq=5)
        df_active = handle_missing_values(df_active, quant_vars, qual_vars)

        if not quant_vars:
            logging.warning("Dataset %s has no quantitative variables", name)
            continue

        res = run_pca(df_active, quant_vars, n_components=2)
        emb = res["embeddings"].iloc[:, :2]
        color_var = args.color if args.color in df_active.columns else None
        colors: pd.Series | None = None
        if color_var is not None:
            colors = df_active.loc[emb.index, color_var].astype("category")
            categories.update(colors.cat.categories.astype(str))
        results[name] = (emb, colors)

    if not results:
        raise RuntimeError("No datasets processed")

    all_emb = pd.concat([emb for emb, _ in results.values()])
    xlims = (all_emb.iloc[:, 0].min(), all_emb.iloc[:, 0].max())
    ylims = (all_emb.iloc[:, 1].min(), all_emb.iloc[:, 1].max())

    cat_list = sorted(categories)
    palette = sns.color_palette("tab10", len(cat_list)) if cat_list else None
    color_map = dict(zip(cat_list, palette)) if palette else {}

    fig, axes = plt.subplots(2, 2, figsize=(12, 12), dpi=200)
    order = ["raw", "cleaned_1", "cleaned_3_univ", "cleaned_3_multi"]
    titles = {
        "raw": "Raw",
        "cleaned_1": "Cleaned_1",
        "cleaned_3_univ": "Cleaned_3_univ",
        "cleaned_3_multi": "Cleaned_3_multi",
    }

    for ax, name in zip(axes.ravel(), order):
        emb_color = results.get(name)
        if emb_color is None:
            ax.set_visible(False)
            continue
        emb, col = emb_color
        if col is None:
            ax.scatter(emb.iloc[:, 0], emb.iloc[:, 1], s=10, alpha=0.6, color="tab:blue")
        else:
            cats = col.astype("category")
            for cat in cat_list:
                mask = cats == cat
                if not mask.any():
                    continue
                ax.scatter(
                    emb.loc[mask, emb.columns[0]],
                    emb.loc[mask, emb.columns[1]],
                    s=10,
                    alpha=0.6,
                    color=color_map.get(cat, "tab:blue"),
                    label=str(cat),
                )
        ax.axhline(0, color="gray", lw=0.5)
        ax.axvline(0, color="gray", lw=0.5)
        ax.set_xlim(xlims)
        ax.set_ylim(ylims)
        ax.set_xlabel(emb.columns[0])
        ax.set_ylabel(emb.columns[1])
        ax.set_title(titles.get(name, name))

    if cat_list:
        handles = [
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=color_map[c],
                label=str(c),
                markersize=6,
                alpha=0.6,
            )
            for c in cat_list
        ]
        fig.legend(handles=handles, title=args.color, loc="upper right")

    fig.suptitle("Comparaison PCA des différentes étapes", y=0.92)
    fig.tight_layout()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.show()


if __name__ == "__main__":  # pragma: no cover - manual execution
    main()
