#!/usr/bin/env python3
"""Run a full Factor Analysis of Mixed Data on the CRM dataset.

This standalone script loads the raw CRM file, prepares the data and
runs FAMD on all quantitative and qualitative variables selected by the
standard helper functions. The "Code Analytique" column is excluded
through :func:`phase4.functions.select_variables`.

The script outputs an "FAMD_scree.png" figure displaying the percentage
of variance explained by each dimension and optionally exports a
``FAMD_variance_expliquee.csv`` summary.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from phase4.functions import (
    _read_dataset,
    prepare_data,
    select_variables,
    handle_missing_values,
    run_famd,
)


def perform_famd(input_path: Path, out_dir: Path, export_csv: bool = True) -> Path:
    """Run FAMD on ``input_path`` and save the scree plot under ``out_dir``."""
    if not input_path.is_file():
        raise FileNotFoundError(input_path)

    logging.info("Reading %s", input_path)
    df_raw = _read_dataset(input_path)
    df_prep = prepare_data(df_raw, exclude_lost=True)
    df_active, quant_vars, qual_vars = select_variables(df_prep, min_modalite_freq=5)
    df_active = handle_missing_values(df_active, quant_vars, qual_vars)

    if not quant_vars or not qual_vars:
        raise ValueError("FAMD requires at least one quantitative and one qualitative variable")

    res = run_famd(df_active, quant_vars, qual_vars, optimize=True)
    inertia = res["inertia"]  # Explained variance ratio
    n_vars = len(quant_vars) + len(qual_vars)

    eigvals = inertia * n_vars
    explained_pct = inertia * 100
    cumulative = explained_pct.cumsum()

    sns.set_palette("deep")
    fig, ax = plt.subplots(figsize=(12, 6), dpi=200)
    bars = ax.bar(range(1, len(explained_pct) + 1), explained_pct)
    ax.plot(range(1, len(cumulative) + 1), cumulative, color="black", marker="o", label="Variance cumulée")
    ax.set_xticks(range(1, len(explained_pct) + 1))
    ax.set_xlabel("Dimension factorielle")
    ax.set_ylabel("% variance expliquée")
    ax.set_title("FAMD – Éboulis des valeurs propres")
    ax.legend()
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)

    out_dir.mkdir(parents=True, exist_ok=True)
    fig_path = out_dir / "FAMD_scree.png"
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)

    if export_csv:
        table = pd.DataFrame({
            "dimension": inertia.index,
            "valeur_propre": eigvals.values,
            "variance_expliquee_pct": explained_pct.values,
            "variance_cumulee_pct": cumulative.values,
        })
        table.to_csv(out_dir / "FAMD_variance_expliquee.csv", index=False)

    logging.info("FAMD results saved to %s", out_dir)
    return fig_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyse FAMD complète du CRM")
    parser.add_argument("input_file", help="CSV/XLSX file containing the CRM data")
    parser.add_argument("--out-dir", default="famd_output", help="Directory for outputs")
    parser.add_argument("--no-csv", action="store_true", help="Do not export the CSV summary")
    args = parser.parse_args()

    logging.basicConfig(format="%(message)s", level=logging.INFO)

    perform_famd(Path(args.input_file), Path(args.out_dir), export_csv=not args.no_csv)


if __name__ == "__main__":
    main()
