"""Exploratory plots of cleaning flags impact."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd

from .config import INPUT_CSV, OUTPUT_DIR


def _load(csv_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, encoding="utf-8")
    df["Date de fin actualisée"] = pd.to_datetime(
        df["Date de fin actualisée"], errors="coerce", dayfirst=False
    )
    return df


def _plot_flag_effect(df: pd.DataFrame, flags: Iterable[str], prefix: str, out: Path) -> None:
    flags = [f for f in flags if f in df.columns]
    if not flags:
        return

    out.mkdir(parents=True, exist_ok=True)
    mask = df[flags].astype(bool).any(axis=1)

    before = df.copy()
    after = df[~mask].copy()

    # Scatter before filtering (highlight flagged points)
    plt.figure(figsize=(10, 5))
    plt.scatter(before["Date de fin actualisée"], before["Total recette réalisé"], s=10, label="All")
    if mask.any():
        plt.scatter(
            before.loc[mask, "Date de fin actualisée"],
            before.loc[mask, "Total recette réalisé"],
            s=10,
            color="red",
            label="Flagged",
        )
    plt.legend()
    plt.xlabel("Date de fin actualisée")
    plt.ylabel("Total recette réalisé")
    plt.tight_layout()
    plt.savefig(out / f"{prefix}_scatter_before.png", dpi=150)
    plt.close()

    # Scatter after filtering
    plt.figure(figsize=(10, 5))
    plt.scatter(after["Date de fin actualisée"], after["Total recette réalisé"], s=10)
    plt.xlabel("Date de fin actualisée")
    plt.ylabel("Total recette réalisé")
    plt.tight_layout()
    plt.savefig(out / f"{prefix}_scatter_after.png", dpi=150)
    plt.close()

    # Aggregated monthly time series
    ts_before = (
        before.set_index("Date de fin actualisée")["Total recette réalisé"].resample("ME").sum()
    )
    ts_after = (
        after.set_index("Date de fin actualisée")["Total recette réalisé"].resample("ME").sum()
    )
    plt.figure(figsize=(12, 6))
    plt.plot(ts_before.index, ts_before.values, label="Before")
    plt.plot(ts_after.index, ts_after.values, label="After")
    plt.legend()
    plt.ylabel("Total recette réalisé (mensuel)")
    plt.tight_layout()
    plt.savefig(out / f"{prefix}_monthly_agg.png", dpi=150)
    plt.close()


def main(csv_path: str | Path = INPUT_CSV, output_dir: str | Path = OUTPUT_DIR) -> None:
    df = _load(csv_path)
    analysis = Path(output_dir) / "analysis"

    flag_cols = [c for c in df.columns if c.startswith("flag_")]
    for col in flag_cols:
        _plot_flag_effect(df, [col], col, analysis)

    # Grouped flags
    univ_flags = [c for c in flag_cols if c.startswith("flag_univ_")]
    if univ_flags:
        _plot_flag_effect(df, univ_flags, "all_univ_flags", analysis)

    combo = flag_cols
    if combo:
        _plot_flag_effect(df, combo, "all_flags", analysis)


if __name__ == "__main__":  # pragma: no cover - CLI helper
    import argparse

    parser = argparse.ArgumentParser(description="Analyse des effets des indicateurs de nettoyage")
    parser.add_argument("--csv", default=str(INPUT_CSV), help="Jeu de données nettoyé phase 3")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="Dossier de sortie des figures")
    args = parser.parse_args()

    main(args.csv, args.output_dir)
