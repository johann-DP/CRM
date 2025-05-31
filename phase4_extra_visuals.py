#!/usr/bin/env python3
"""Generate additional Phase 4 visualisations.

This script creates density maps on PCA and UMAP projections and
scatter plots with centroids and covariance ellipses for FAMD, PCA,
UMAP and PaCMAP. It mirrors the data preparation steps of
``phase4.py`` but only runs the methods needed for these extra
figures.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
import pandas as pd
import seaborn as sns
import yaml

from phase4_functions import (
    load_datasets,
    prepare_data,
    select_variables,
    handle_missing_values,
    run_pca,
    run_famd,
    run_umap,
    run_pacmap,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_config(path: Path) -> Mapping[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        if path.suffix.lower() in {".yaml", ".yml"}:
            return yaml.safe_load(fh)
        return json.load(fh)


def _choose_color_var(df: pd.DataFrame, qual_vars: Sequence[str]) -> str | None:
    preferred = [
        "Statut production",
        "Statut commercial",
        "Type opportunité",
    ]
    for col in preferred:
        if col in df.columns:
            return col
    for col in qual_vars:
        if col in df.columns:
            return col
    return None


def _plot_density(df: pd.DataFrame, x: str, y: str, title: str, out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(6, 5), dpi=200)
    sns.kdeplot(data=df, x=x, y=y, fill=True, cmap="mako", thresh=0, ax=ax)
    ax.scatter(df[x], df[y], s=5, color="white", alpha=0.4)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(title)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return out


def _cov_ellipse(ax: plt.Axes, data: np.ndarray, color: str, n_std: float = 2.0) -> None:
    if data.shape[0] <= 2:
        return
    cov = np.cov(data, rowvar=False)
    if not np.all(np.isfinite(cov)):
        return
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    width, height = 2 * n_std * np.sqrt(vals)
    angle = math.degrees(math.atan2(*vecs[:, 0][::-1]))
    mean = data.mean(axis=0)
    ell = Ellipse(xy=mean, width=width, height=height, angle=angle,
                  edgecolor=color, facecolor="none", lw=2)
    ax.add_patch(ell)
    ax.scatter(*mean, marker="x", color=color, s=50)


def _plot_ellipses(
    df: pd.DataFrame, x: str, y: str, group: str | None, title: str, out: Path
) -> Path:
    fig, ax = plt.subplots(figsize=(6, 5), dpi=200)
    if group is None or group not in df.columns:
        ax.scatter(df[x], df[y], s=10, alpha=0.6)
    else:
        cats = df[group].astype("category")
        palette = sns.color_palette("tab10", len(cats.cat.categories))
        for color, cat in zip(palette, cats.cat.categories):
            sub = df.loc[cats == cat, [x, y]].values
            ax.scatter(sub[:, 0], sub[:, 1], s=10, alpha=0.6, color=color, label=str(cat))
            _cov_ellipse(ax, sub, color)
        ax.legend(title=group, bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(title)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------

def process_dataset(name: str, df: pd.DataFrame, out_dir: Path) -> None:
    logging.info("Processing %s", name)
    df_prep = prepare_data(df, exclude_lost=True)
    df_active, quant, qual = select_variables(df_prep, min_modalite_freq=5)
    df_active = handle_missing_values(df_active, quant, qual)
    color_var = _choose_color_var(df_active, qual)

    results = {
        "pca": run_pca(df_active, quant_vars=quant, n_components=2),
        "famd": None,
        "umap": run_umap(df_active, n_components=2),
        "pacmap": None,
    }
    if quant and qual:
        results["famd"] = run_famd(df_active, quant, qual, n_components=2)
    try:
        results["pacmap"] = run_pacmap(df_active, n_components=2)
    except Exception as exc:  # pragma: no cover - optional dependency
        logging.warning("PaCMAP failed: %s", exc)

    sub = out_dir / name
    for key in ["density", "ellipses"]:
        (sub / key).mkdir(parents=True, exist_ok=True)

    emb = results["pca"]["embeddings"]
    _plot_density(
        emb,
        emb.columns[0],
        emb.columns[1],
        f"{name} PCA – densité",
        sub / "density" / "pca_density.png",
    )

    emb = results["umap"]["embeddings"]
    if not emb.empty:
        _plot_density(
            emb,
            emb.columns[0],
            emb.columns[1],
            f"{name} UMAP – densité",
            sub / "density" / "umap_density.png",
        )

    if results["famd"] is not None:
        emb = results["famd"]["embeddings"]
        _plot_ellipses(
            pd.concat([emb.iloc[:, :2], df_active[color_var]], axis=1),
            emb.columns[0],
            emb.columns[1],
            color_var,
            f"{name} FAMD – ellipses",
            sub / "ellipses" / "famd_ellipses.png",
        )

    emb = results["pca"]["embeddings"]
    _plot_ellipses(
        pd.concat([emb.iloc[:, :2], df_active[color_var]], axis=1),
        emb.columns[0],
        emb.columns[1],
        color_var,
        f"{name} PCA – ellipses",
        sub / "ellipses" / "pca_ellipses.png",
    )

    emb = results["umap"]["embeddings"]
    if not emb.empty:
        _plot_ellipses(
            pd.concat([emb.iloc[:, :2], df_active[color_var]], axis=1),
            emb.columns[0],
            emb.columns[1],
            color_var,
            f"{name} UMAP – ellipses",
            sub / "ellipses" / "umap_ellipses.png",
        )

    emb = results.get("pacmap", {}).get("embeddings")
    if emb is not None and not emb.empty:
        _plot_ellipses(
            pd.concat([emb.iloc[:, :2], df_active[color_var]], axis=1),
            emb.columns[0],
            emb.columns[1],
            color_var,
            f"{name} PaCMAP – ellipses",
            sub / "ellipses" / "pacmap_ellipses.png",
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Extra Phase 4 visualisations")
    parser.add_argument("--config", required=True, help="Path to config YAML/JSON")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["raw", "cleaned_1", "cleaned_3_univ", "cleaned_3_multi"],
        help="Datasets to process",
    )
    args = parser.parse_args()

    cfg = _load_config(Path(args.config))
    output_dir = Path(cfg.get("output_dir", "phase4_output")) / "extra_plots"
    logging.basicConfig(level="INFO")

    data = load_datasets(cfg, ignore_schema=True)
    for name in args.datasets:
        df = data.get(name)
        if df is None:
            logging.warning("Dataset %s missing", name)
            continue
        process_dataset(name, df, output_dir)


if __name__ == "__main__":  # pragma: no cover
    main()
