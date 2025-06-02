#!/usr/bin/env python3
"""Compute FAMD variable contributions for the first two axes.

This script loads the main dataset defined in ``config.yaml`` and exports a
CSV file ``FAMD_contributions_variables.csv`` with the contribution percentage
of each variable to axes F1 and F2. Categorical variables are aggregated across
modalities.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import yaml

import phase4.functions as pf


def load_config(path: Path) -> Mapping[str, Any]:
    """Return configuration dictionary from ``path``."""
    with open(path, "r", encoding="utf-8") as fh:
        if path.suffix.lower() in {".yaml", ".yml"}:
            return yaml.safe_load(fh)
        return json.load(fh)


def aggregate_contributions(contrib: pd.DataFrame) -> pd.DataFrame:
    """Return contributions aggregated by variable for F1 and F2."""
    if not {"F1", "F2"}.issubset(contrib.columns):
        cols = list(contrib.columns[:2])
        rename: dict[str, str] = {}
        if cols:
            rename[cols[0]] = "F1"
        if len(cols) > 1:
            rename[cols[1]] = "F2"
        contrib = contrib.rename(columns=rename)
    if "F2" not in contrib.columns:
        contrib["F2"] = 0.0

    grouped: dict[str, pd.Series] = {}
    for idx in contrib.index:
        var = idx.split("__", 1)[0]
        grouped.setdefault(var, pd.Series(dtype=float))
        grouped[var] = grouped[var].add(contrib.loc[idx, ["F1", "F2"]], fill_value=0)
    df = pd.DataFrame(grouped).T.fillna(0)
    df = df.loc[sorted(df.index)]
    return df


def run(config: Mapping[str, Any]) -> Path:
    """Compute FAMD contributions according to ``config``."""
    datasets = pf.load_datasets(config, ignore_schema=bool(config.get("ignore_schema", False)))
    name = config.get("dataset", config.get("main_dataset", "raw"))
    if name not in datasets:
        raise KeyError(f"dataset '{name}' not found")

    df_prep = pf.prepare_data(datasets[name], exclude_lost=bool(config.get("exclude_lost", True)))
    df_active, quant, qual = pf.select_variables(
        df_prep, min_modalite_freq=int(config.get("min_modalite_freq", 5))
    )
    df_active = pf.handle_missing_values(df_active, quant, qual)

    res = pf.run_famd(df_active, quant, qual, n_components=2)
    contrib = res.get("contributions")
    if not isinstance(contrib, pd.DataFrame):
        raise RuntimeError("FAMD contributions missing")

    df = aggregate_contributions(contrib)
    out_path = Path("FAMD_contributions_variables.csv")
    df.to_csv(out_path, index_label="variable", float_format="%.6f")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Export FAMD variable contributions")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    args = parser.parse_args()

    config = load_config(Path(args.config))
    path = run(config)
    print(f"Saved contributions to {path}")


if __name__ == "__main__":
    main()

