#!/usr/bin/env python3
"""Compute and plot correlation matrix for the cleaned_3_univ dataset.

The script reads ``config.yaml`` to locate the ``input_file_cleaned_3_univ``
dataset and the ``output_dir``. It saves a CSV file with the correlation
matrix and a heatmap titled "Matrice de corrélation des variables
quantitatives (cleaned_3_univ)" in the output directory.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import yaml


logger = logging.getLogger(__name__)


def _read_dataset(path: Path) -> pd.DataFrame:
    """Read a CSV or Excel file with basic type handling."""
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        df = pd.read_excel(path)
    df.columns = [str(c).strip() for c in df.columns]
    for col in df.columns:
        if "date" in col.lower():
            df[col] = pd.to_datetime(df[col], errors="coerce")
    for col in df.select_dtypes(include="object"):
        if any(k in col.lower() for k in ["montant", "recette", "budget", "total"]):
            series = df[col].astype(str).str.replace("\xa0", "", regex=False)
            series = series.str.replace(" ", "", regex=False)
            series = series.str.replace(",", ".", regex=False)
            df[col] = pd.to_numeric(series, errors="coerce")
    return df


def load_config(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Correlation matrix for cleaned_3_univ")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    args = parser.parse_args(argv)

    cfg = load_config(Path(args.config))
    data_file = Path(cfg.get("input_file_cleaned_3_univ", ""))
    out_dir = Path(cfg.get("output_dir", "."))
    if not data_file.exists():
        parser.error(f"Dataset not found: {data_file}")
    out_dir.mkdir(parents=True, exist_ok=True)

    df = _read_dataset(data_file)

    numeric_cols = [c for c in df.select_dtypes(include="number").columns if df[c].nunique(dropna=True) > 1]
    if not numeric_cols:
        parser.error("No quantitative variables found in dataset")

    corr = df[numeric_cols].corr()

    csv_path = out_dir / "cleaned_3_univ_correlation_matrix.csv"
    corr.to_csv(csv_path, index=True)
    logger.info("Correlation matrix saved to %s", csv_path)

    n_vars = len(corr.columns)
    cell_size = 0.4
    fig_width = cell_size * n_vars + 3
    fig_height = cell_size * n_vars + 2
    plt.figure(figsize=(fig_width, fig_height))
    plt.figure(figsize=(12, 6))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="YlGnBu",
        vmin=0,
        vmax=1,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8, "label": "Coefficient de corrélation"},
        square=True,
    )
    plt.title("Matrice de corrélation des variables quantitatives (cleaned_3_univ)")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    png_path = out_dir / "cleaned_3_univ_correlation_heatmap.png"
    plt.savefig(png_path)
    plt.close()
    logger.info("Heatmap saved to %s", png_path)


if __name__ == "__main__":  # pragma: no cover - manual execution
    logging.basicConfig(level=logging.INFO)
    main()
