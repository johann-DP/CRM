#!/usr/bin/env python3
"""Export MFA group contributions for each axis to a CSV file."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import yaml

import phase4_functions as pf


def load_config(path: Path) -> Mapping[str, Any]:
    """Load YAML or JSON configuration file."""
    with open(path, "r", encoding="utf-8") as fh:
        if path.suffix.lower() in {".yaml", ".yml"}:
            return yaml.safe_load(fh)
        return json.load(fh)


def run(config: Mapping[str, Any]) -> Path:
    """Compute MFA group contributions using ``config`` settings."""
    datasets = pf.load_datasets(config, ignore_schema=bool(config.get("ignore_schema", False)))
    name = config.get("dataset", config.get("main_dataset", "raw"))
    if name not in datasets:
        raise KeyError(f"dataset '{name}' not found")

    df_prep = pf.prepare_data(datasets[name], exclude_lost=bool(config.get("exclude_lost", True)))
    df_active, quant, qual = pf.select_variables(
        df_prep, min_modalite_freq=int(config.get("min_modalite_freq", 5))
    )
    df_active = pf.handle_missing_values(df_active, quant, qual)

    mfa_cfg = config.get("mfa", {})
    groups = mfa_cfg.get("groups")
    if not groups:
        groups = []
        if quant:
            groups.append(quant)
        if qual:
            groups.append(qual)
    res = pf.run_mfa(
        df_active,
        groups,
        n_components=mfa_cfg.get("n_components"),
        optimize=False,
        segment_col=None,
        weights=mfa_cfg.get("weights"),
        n_iter=int(mfa_cfg.get("n_iter", 3)),
    )
    model = res["model"]
    table = pf.mfa_group_contributions(model)
    out_path = Path("AFM_group_contributions.csv")
    table.to_csv(out_path, index_label="axe", float_format="%.6f")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Export MFA group contributions")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    path = run(cfg)
    print(f"Saved contributions to {path}")


if __name__ == "__main__":
    main()
