#!/usr/bin/env python3
"""Export FAMD individual factor scores to a CSV file."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import yaml

import phase4
from phase4.functions import (
    load_datasets,
    prepare_data,
    select_variables,
    handle_missing_values,
    run_famd,
)


def load_config(path: Path) -> Mapping[str, Any]:
    """Load YAML or JSON configuration file."""
    with open(path, "r", encoding="utf-8") as fh:
        if path.suffix.lower() in {".yaml", ".yml"}:
            return yaml.safe_load(fh)
        return json.load(fh)


def export_famd_scores(
    config: Mapping[str, Any],
    *,
    dataset: str | None = None,
    output: str | Path = "FAMD_coordonnees_individus.csv",
) -> pd.DataFrame:
    """Compute FAMD coordinates and save them to ``output``.

    Parameters
    ----------
    config:
        Configuration mapping used to load the dataset and parameters.
    dataset:
        Optional dataset name overriding the value from ``config``.
    output:
        Path of the CSV file written by the function.

    Returns
    -------
    pandas.DataFrame
        DataFrame of individual coordinates (index named ``ID``).
    """
    datasets = load_datasets(config, ignore_schema=bool(config.get("ignore_schema", False)))
    name = dataset or config.get("dataset", config.get("main_dataset", "raw"))
    if name not in datasets:
        raise KeyError(f"dataset '{name}' not found")

    df_prep = prepare_data(datasets[name], exclude_lost=bool(config.get("exclude_lost", True)))
    df_active, quant_vars, qual_vars = select_variables(
        df_prep, min_modalite_freq=int(config.get("min_modalite_freq", 5))
    )
    df_active = handle_missing_values(df_active, quant_vars, qual_vars)

    if not quant_vars or not qual_vars:
        raise ValueError("FAMD requires at least one quantitative and one qualitative variable")

    params = phase4._method_params("famd", config)  # type: ignore[attr-defined]
    res = run_famd(df_active, quant_vars, qual_vars, optimize=False, **params)
    coords = res["embeddings"].copy()
    coords.index.name = "ID"
    coords.columns = [f"Dim{i+1}_FAMD" for i in range(coords.shape[1])]
    coords.reset_index().to_csv(output, index=False)
    return coords


def main() -> None:
    parser = argparse.ArgumentParser(description="Export FAMD factor scores")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    parser.add_argument("--dataset", help="Dataset to process (override config)")
    parser.add_argument(
        "--output",
        default="FAMD_coordonnees_individus.csv",
        help="Output CSV file",
    )
    args = parser.parse_args()

    logging.basicConfig(format="%(message)s", level=logging.INFO)

    cfg = load_config(Path(args.config))
    if args.dataset:
        cfg["dataset"] = args.dataset

    df = export_famd_scores(cfg, dataset=args.dataset, output=args.output)
    logging.info("Saved %s with %d rows", args.output, len(df))


if __name__ == "__main__":
    main()
