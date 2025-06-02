from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd
import yaml

import phase4
from phase4.functions import (
    load_datasets,
    prepare_data,
    select_variables,
    handle_missing_values,
    run_mfa,
)


def load_config(path: Path) -> Mapping[str, Any]:
    """Return configuration mapping from ``path``."""
    with open(path, "r", encoding="utf-8") as fh:
        if path.suffix.lower() in {".yaml", ".yml"}:
            return yaml.safe_load(fh)
        return json.load(fh)


def export_mfa_inertias(
    config: Mapping[str, Any],
    *,
    dataset: str | None = None,
    output: str | Path = "AFM_inerties.csv",
) -> pd.DataFrame:
    """Compute MFA inertias and save them to ``output``."""
    phase4.set_blas_threads(int(config.get("n_jobs", -1)))

    datasets = load_datasets(config, ignore_schema=bool(config.get("ignore_schema", False)))
    name = dataset or config.get("dataset", config.get("main_dataset", "raw"))
    if name not in datasets:
        raise KeyError(f"dataset '{name}' not found")

    df_prep = prepare_data(datasets[name], exclude_lost=bool(config.get("exclude_lost", True)))
    df_active, quant_vars, qual_vars = select_variables(
        df_prep, min_modalite_freq=int(config.get("min_modalite_freq", 5))
    )
    df_active = handle_missing_values(df_active, quant_vars, qual_vars)

    if not (quant_vars or qual_vars):
        raise ValueError("No variables available for MFA")

    groups: list[Sequence[str]] = []
    if quant_vars:
        groups.append(quant_vars)
    if qual_vars:
        groups.append(qual_vars)

    params = phase4._method_params("mfa", config)  # type: ignore[attr-defined]
    grp = params.pop("groups", None)
    if grp:
        groups = grp

    res = run_mfa(df_active, groups, optimize=False, **params)
    inertia = res["inertia"]
    eigvals = inertia.values * df_active.shape[1]
    explained_pct = inertia.values * 100
    cumulative = explained_pct.cumsum()

    table = pd.DataFrame(
        {
            "axe": inertia.index,
            "valeur_propre": eigvals,
            "variance_expliquee_pct": explained_pct,
            "variance_cumulee_pct": cumulative,
        }
    )
    table.to_csv(output, index=False)
    logging.info("Saved MFA inertias to %s", output)
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description="Export MFA inertias")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    parser.add_argument("--dataset", help="Dataset to process (override config)")
    parser.add_argument("--output", default="AFM_inerties.csv", help="Output CSV file")
    args = parser.parse_args()

    config = load_config(Path(args.config))
    if args.dataset:
        config["dataset"] = args.dataset

    export_mfa_inertias(config, dataset=args.dataset, output=args.output)


if __name__ == "__main__":
    logging.basicConfig(format="%(message)s", level=logging.INFO)
    main()
