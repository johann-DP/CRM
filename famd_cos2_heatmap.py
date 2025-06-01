#!/usr/bin/env python3
"""Generate a FAMD cos² heatmap for all variables."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import yaml

import phase4_functions as pf


def load_config(path: Path) -> Mapping[str, Any]:
    """Load YAML or JSON configuration file."""
    with open(path, "r", encoding="utf-8") as fh:
        if path.suffix.lower() in {".yaml", ".yml"}:
            return yaml.safe_load(fh)
        return json.load(fh)


def compute_cos2(coords: pd.DataFrame) -> pd.DataFrame:
    """Return cos² (%) for each variable from ``coords``."""
    if coords.empty:
        return pd.DataFrame()
    cols = list(coords.columns[:2])
    rename: dict[str, str] = {}
    if cols:
        rename[cols[0]] = "F1"
    if len(cols) > 1:
        rename[cols[1]] = "F2"
    coords = coords.rename(columns=rename)
    if "F2" not in coords.columns:
        coords["F2"] = 0.0
    sq = coords ** 2
    return sq.div(sq.sum(axis=1), axis=0) * 100


def run(config: Mapping[str, Any]) -> pd.DataFrame:
    """Compute FAMD variable cos² according to ``config``."""
    data = pf.load_datasets(config, ignore_schema=bool(config.get("ignore_schema", False)))
    name = config.get("dataset", config.get("main_dataset", "raw"))
    if name not in data:
        raise KeyError(f"dataset '{name}' not found")
    df_prep = pf.prepare_data(data[name], exclude_lost=bool(config.get("exclude_lost", True)))
    df_active, quant, qual = pf.select_variables(
        df_prep, min_modalite_freq=int(config.get("min_modalite_freq", 5))
    )
    df_active = pf.handle_missing_values(df_active, quant, qual)
    res = pf.run_famd(df_active, quant, qual, n_components=2)
    coords = res.get("column_coords")
    if not isinstance(coords, pd.DataFrame):
        raise RuntimeError("FAMD column coordinates missing")
    cos2 = compute_cos2(coords)
    cos2 = cos2.loc[sorted(cos2.index)]
    return cos2


def plot_heatmap(cos2: pd.DataFrame, output: Path) -> None:
    """Plot ``cos2`` as a heatmap and save to ``output``."""
    fig, ax = plt.subplots(figsize=(12, 6), dpi=200)
    sns.heatmap(cos2, annot=True, fmt=".1f", cmap="coolwarm", vmin=0, vmax=100, ax=ax)
    ax.set_xlabel("Dimension factorielle")
    ax.set_ylabel("Variable")
    ax.set_title("FAMD – cos² des variables")
    plt.yticks(rotation=0)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Heatmap des cos² des variables en FAMD")
    parser.add_argument("--config", default="config.yaml", help="Fichier de configuration")
    parser.add_argument("--output", default="FAMD_cos2_heatmap.png", help="Fichier PNG de sortie")
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    cos2 = run(cfg)
    plot_heatmap(cos2, Path(args.output))


if __name__ == "__main__":  # pragma: no cover
    main()
