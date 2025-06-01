#!/usr/bin/env python3
"""Generate a heatmap of cos² for FAMD individuals."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import phase4_functions as pf


def load_dataset(path: Path) -> pd.DataFrame:
    """Load CSV or Excel dataset according to file extension."""
    if path.suffix.lower() in {".xls", ".xlsx"}:
        return pd.read_excel(path)
    return pd.read_csv(path)


def plot_cos2_heatmap(df: pd.DataFrame, output: Path, max_rows: int = 200) -> None:
    """Compute FAMD cos² and save a heatmap."""
    quant_vars = df.select_dtypes(include=[np.number]).columns.tolist()
    qual_vars = [c for c in df.columns if c not in quant_vars]
    res = pf.run_famd(df, quant_vars, qual_vars)
    cos2 = pf.famd_individual_cos2(res["embeddings"])
    if len(cos2) > max_rows:
        cos2 = cos2.sample(n=max_rows, random_state=0)
    fig, ax = plt.subplots(figsize=(12, 6), dpi=200)
    sns.heatmap(cos2, cmap="coolwarm", ax=ax)
    ax.set_title("FAMD – cos² des individus")
    ax.set_xlabel("Axe")
    ax.set_ylabel("Individu")
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Heatmap cos² des individus FAMD")
    parser.add_argument("input_file", type=Path, help="CSV or Excel file")
    parser.add_argument("--output", type=Path, default=Path("FAMD_cos2_heatmap.png"))
    parser.add_argument("--max-rows", type=int, default=200, help="Maximum individuals displayed")
    args = parser.parse_args()

    df = load_dataset(args.input_file)
    plot_cos2_heatmap(df, args.output, args.max_rows)
    

if __name__ == "__main__":  # pragma: no cover
    main()
