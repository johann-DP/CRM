#!/usr/bin/env python3
# flake8: noqa: E402
"""Plot heatmap of PCA cos² for individuals."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import yaml

from phase4_functions import (
    load_datasets,
    prepare_data,
    select_variables,
    handle_missing_values,
    run_pca,
    pca_individual_contributions,
)


logger = logging.getLogger(__name__)


def load_config(path: Path) -> Mapping[str, Any]:
    """Load YAML configuration file."""
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def pca_cos2_heatmap(
    config: Mapping[str, Any],
    *,
    dataset: str = "raw",
    n_components: int = 2,
    max_obs: int = 50,
    output: Path = Path("ACP_cos2_individus.png"),
) -> Path:
    """Compute cos² for PCA individuals and save a heatmap."""

    datasets = load_datasets(config, ignore_schema=bool(config.get("ignore_schema", False)))
    if dataset not in datasets:
        raise KeyError(f"dataset '{dataset}' not found")

    df = datasets[dataset]
    df_prep = prepare_data(df, exclude_lost=bool(config.get("exclude_lost", True)))

    df_active, quant_vars, qual_vars = select_variables(df_prep)
    df_active = handle_missing_values(df_active, quant_vars, qual_vars)

    if not quant_vars:
        raise ValueError("No quantitative variables available for PCA")

    res = run_pca(df_active, quant_vars, n_components=n_components)
    coords = res["embeddings"].iloc[:, :n_components]
    cos2 = pca_individual_contributions(coords)

    cos2["total"] = cos2.sum(axis=1)
    cos2 = cos2.sort_values("total", ascending=False).drop(columns="total")
    if cos2.shape[0] > max_obs:
        cos2 = cos2.iloc[:max_obs]

    fig, ax = plt.subplots(figsize=(12, 6), dpi=200)
    sns.heatmap(cos2, cmap="coolwarm", vmin=0, vmax=100, ax=ax)
    ax.set_title("ACP – cos² des individus")
    ax.set_xlabel("Axe factoriel")
    ax.set_ylabel("Individu")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved heatmap to %s", output)
    return output


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="PCA cos² heatmap for individuals")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    parser.add_argument("--dataset", default="raw", help="Dataset to process")
    parser.add_argument("--components", type=int, default=2, help="Number of PCA components")
    parser.add_argument("--max-obs", type=int, default=50, help="Maximum individuals to display")
    parser.add_argument("--output", default="ACP_cos2_individus.png", help="Output PNG file")
    args = parser.parse_args(argv)

    cfg = load_config(Path(args.config))
    pca_cos2_heatmap(
        cfg,
        dataset=args.dataset,
        n_components=args.components,
        max_obs=args.max_obs,
        output=Path(args.output),
    )


if __name__ == "__main__":  # pragma: no cover - CLI
    logging.basicConfig(format="%(message)s", level=logging.INFO)
    main()
