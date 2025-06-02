import argparse
import json
import logging
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import yaml
from scipy.spatial.distance import pdist, squareform

import phase4
from phase4.functions import (
    load_datasets,
    prepare_data,
    select_variables,
    handle_missing_values,
)


def _load_config(path: Path) -> Mapping[str, Any]:
    """Return configuration mapping from YAML or JSON file."""
    with open(path, "r", encoding="utf-8") as fh:
        if path.suffix.lower() in {".yaml", ".yml"}:
            return yaml.safe_load(fh)
        return json.load(fh)


def _encode_data(
    df: pd.DataFrame, quant_vars: list[str], qual_vars: list[str]
) -> np.ndarray:
    """Return numeric matrix with one-hot encoded categoricals."""
    parts = []
    if quant_vars:
        parts.append(df[quant_vars].to_numpy(float))
    if qual_vars:
        dummies = pd.get_dummies(df[qual_vars], dummy_na=False)
        parts.append(dummies.to_numpy(float))
    if parts:
        return np.hstack(parts)
    return np.empty((len(df), 0))


def compute_distance_matrix(
    df: pd.DataFrame, quant_vars: list[str], qual_vars: list[str]
) -> pd.DataFrame:
    """Compute Euclidean distance matrix on ``df``."""
    X = _encode_data(df, quant_vars, qual_vars)
    dist = squareform(pdist(X, metric="euclidean"))
    return pd.DataFrame(dist, index=df.index, columns=df.index)


def run(config: Mapping[str, Any], *, output: Path) -> None:
    """Generate distance matrix for configured dataset."""
    phase4.set_blas_threads(int(config.get("n_jobs", -1)))

    datasets = load_datasets(config, ignore_schema=bool(config.get("ignore_schema", False)))
    name = config.get("dataset", config.get("main_dataset", "raw"))
    if name not in datasets:
        raise KeyError(f"dataset '{name}' not found")

    df_prep = prepare_data(datasets[name], exclude_lost=bool(config.get("exclude_lost", True)))
    df_active, quant_vars, qual_vars = select_variables(
        df_prep, min_modalite_freq=int(config.get("min_modalite_freq", 5))
    )
    df_active = handle_missing_values(df_active, quant_vars, qual_vars)

    matrix = compute_distance_matrix(df_active, quant_vars, qual_vars)
    matrix.to_csv(output, sep=";", encoding="utf-8")
    logging.info("Saved distance matrix to %s", output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export distance matrix for CRM dataset")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    parser.add_argument("--dataset", help="Dataset to process (override config)")
    parser.add_argument("--output", default="Distances_inter_individus.csv", help="Output CSV path")
    args = parser.parse_args()

    logging.basicConfig(format="%(message)s", level=logging.INFO)

    cfg = _load_config(Path(args.config))
    if args.dataset:
        cfg["dataset"] = args.dataset
    run(cfg, output=Path(args.output))
