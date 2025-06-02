# -*- coding: utf-8 -*-
"""Compare UMAP and PaCMAP projections on the CRM dataset.

This script mirrors the data preparation steps used in the repository
and creates a side by side figure of the two non-linear methods.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import yaml

from phase4.functions import (
    load_datasets,
    prepare_data,
    select_variables,
    handle_missing_values,
    run_umap,
    run_pacmap,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_config(path: Path) -> Mapping[str, Any]:
    """Return configuration mapping from ``path``."""
    with open(path, "r", encoding="utf-8") as fh:
        if path.suffix.lower() in {".yaml", ".yml"}:
            return yaml.safe_load(fh)
        return json.load(fh)


def _cov_ellipse(ax: plt.Axes, data: Sequence[Sequence[float]], color: str, n_std: float = 2.0) -> None:
    """Draw a covariance ellipse covering roughly ``n_std`` standard deviations."""
    import numpy as np
    from matplotlib.patches import Ellipse

    data = np.asarray(data)
    if data.shape[0] <= 2:
        return
    cov = np.cov(data, rowvar=False)
    if not np.all(np.isfinite(cov)):
        return
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    width, height = 2 * n_std * np.sqrt(vals)
    angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    mean = data.mean(axis=0)
    ell = Ellipse(xy=mean, width=width, height=height, angle=angle,
                  edgecolor=color, facecolor="none", lw=2)
    ax.add_patch(ell)
    ax.scatter(*mean, marker="x", color=color, s=50)


# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="UMAP vs PaCMAP comparison")
    parser.add_argument("--config", required=True, help="Path to YAML/JSON config")
    parser.add_argument("--dataset", default="raw", help="Dataset key to use")
    parser.add_argument("--color", default="", help="Column used for colouring")
    parser.add_argument("--output", default="umap_pacmap.png", help="Output image path")
    parser.add_argument("--csv", action="store_true", help="Export coordinates as CSV")
    args = parser.parse_args()

    cfg = _load_config(Path(args.config))
    data = load_datasets(cfg, ignore_schema=True)
    if args.dataset not in data:
        raise KeyError(f"dataset '{args.dataset}' not found")

    df = data[args.dataset]
    df_prep = prepare_data(df, exclude_lost=True)
    df_active, quant, qual = select_variables(df_prep, min_modalite_freq=5)
    df_active = handle_missing_values(df_active, quant, qual)

    color_var = args.color if args.color in df_prep.columns else None
    colors = None
    if color_var is not None:
        colors = df_prep.loc[df_active.index, color_var].astype("category")
        palette = sns.color_palette("deep", len(colors.cat.categories))
    else:
        palette = None

    umap_res = run_umap(df_active, n_components=2, random_state=42)
    pacmap_res = run_pacmap(df_active, n_components=2)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6), dpi=200)
    for ax, res, title in zip(axes, [umap_res, pacmap_res], ["UMAP", "PaCMAP"]):
        emb = res.get("embeddings")
        if not isinstance(emb, pd.DataFrame) or emb.empty:
            ax.set_visible(False)
            continue
        x, y = emb.columns[:2]
        if colors is None:
            ax.scatter(emb[x], emb[y], s=10, alpha=0.6, color="tab:blue")
        else:
            for col, cat in zip(palette, colors.cat.categories):
                mask = colors == cat
                ax.scatter(emb.loc[mask, x], emb.loc[mask, y], s=10, alpha=0.6, color=col, label=str(cat))
                _cov_ellipse(ax, emb.loc[mask, [x, y]].values, col)
            ax.legend(title=color_var, bbox_to_anchor=(1.05, 1), loc="upper left")
        ax.axhline(0, color="gray", lw=0.5)
        ax.axvline(0, color="gray", lw=0.5)
        ax.set_xlabel("Dim1")
        ax.set_ylabel("Dim2")
        ax.set_title(title)

    plt.suptitle("Comparaison UMAP vs PaCMAP")
    fig.tight_layout()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.show()

    if args.csv:
        if isinstance(umap_res.get("embeddings"), pd.DataFrame) and not umap_res["embeddings"].empty:
            umap_res["embeddings"].to_csv("UMAP_coordonnees.csv", index=False)
        if isinstance(pacmap_res.get("embeddings"), pd.DataFrame) and not pacmap_res["embeddings"].empty:
            pacmap_res["embeddings"].to_csv("PaCMAP_coordonnees.csv", index=False)


if __name__ == "__main__":  # pragma: no cover - manual execution
    main()
