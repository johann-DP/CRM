#!/usr/bin/env python3
"""Export FAMD inertia table covering a cumulative variance threshold."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import yaml

import phase4
from phase4_functions import (
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


def export_famd_inertias(
    config: Mapping[str, Any],
    *,
    dataset: str | None = None,
    threshold: float = 0.95,
    output: str | Path = "FAMD_inerties.csv",
) -> pd.DataFrame:
    """Compute FAMD inertias and save them to ``output``.

    Parameters
    ----------
    config:
        Configuration mapping used to load the dataset and parameters.
    dataset:
        Optional dataset name overriding the value from ``config``.
    threshold:
        Cumulative variance threshold used to select the number of axes.
    output:
        Path of the CSV file written by the function.

    Returns
    -------
    pandas.DataFrame
        DataFrame of inertias per axis.
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
    params.pop("n_components", None)
    params.setdefault("variance_threshold", threshold)

    res = run_famd(
        df_active,
        quant_vars,
        qual_vars,
        optimize=True,
        **params,
    )
    inertia = res["inertia"]
    n_vars = len(quant_vars) + len(qual_vars)
    eigvals = inertia * n_vars
    explained_pct = inertia * 100
    cumulative = explained_pct.cumsum()

    table = pd.DataFrame(
        {
            "dimension": inertia.index,
            "valeur_propre": eigvals.values,
            "variance_expliquee_pct": explained_pct.values,
            "variance_cumulee_pct": cumulative.values,
        }
    )
    pd.DataFrame(table).to_csv(output, index=False)
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description="Export FAMD inertias")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    parser.add_argument("--dataset", help="Dataset to process (override config)")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.95,
        help="Cumulative variance threshold",
    )
    parser.add_argument(
        "--output",
        default="FAMD_inerties.csv",
        help="Output CSV file",
    )
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    if args.dataset:
        cfg["dataset"] = args.dataset

    df = export_famd_inertias(cfg, dataset=args.dataset, threshold=args.threshold, output=args.output)
    print(f"Saved {args.output} with {len(df)} rows")


if __name__ == "__main__":
    main()
