"""Aggregate won opportunity revenue by time periods."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Tuple

import pandas as pd


# ---------------------------------------------------------------------------
# Core functionality
# ---------------------------------------------------------------------------

def load_won_opportunities(
    path: str | Path,
    *,
    date_col: str = "Date de fin réelle",
    status_col: str = "Statut commercial",
    won_values: Iterable[str] | None = None,
    amount_col: str = "Total recette réalisé",
) -> pd.DataFrame:
    """Return DataFrame of won opportunities from ``path``.

    Parameters
    ----------
    path:
        CSV file containing the opportunities.
    date_col:
        Name of the column with the closing date.
    status_col:
        Column indicating whether the opportunity is won.
    won_values:
        Iterable of values considered as "won". If ``None`` defaults to
        ``{"Won", "Gagnée"}``.
    amount_col:
        Column containing the revenue amount.
    """
    if won_values is None:
        won_values = {"Won", "Gagnée"}

    df = pd.read_csv(path)
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df[df[status_col].isin(set(won_values))].copy()
    df = df.sort_values(date_col)
    df = df.set_index(date_col)
    return df[[amount_col]]


def aggregate_revenue(df: pd.DataFrame, amount_col: str) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Aggregate revenue by month, quarter and year.

    Missing periods are filled with 0 so that the returned series have
    continuous indices.
    """
    monthly = df[amount_col].resample("M").sum().fillna(0)
    quarterly = df[amount_col].resample("Q").sum().fillna(0)
    yearly = df[amount_col].resample("A").sum().fillna(0)
    return monthly, quarterly, yearly


# ---------------------------------------------------------------------------
# CLI / helper
# ---------------------------------------------------------------------------


def build_timeseries(
    csv_path: str | Path,
    *,
    date_col: str = "Date de fin réelle",
    status_col: str = "Statut commercial",
    won_values: Iterable[str] | None = None,
    amount_col: str = "Total recette réalisé",
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Load data and return the aggregated revenue time series."""
    df = load_won_opportunities(
        csv_path,
        date_col=date_col,
        status_col=status_col,
        won_values=won_values,
        amount_col=amount_col,
    )
    return aggregate_revenue(df, amount_col)


if __name__ == "__main__":  # pragma: no cover - CLI helper
    import argparse

    parser = argparse.ArgumentParser(description="Aggregate revenue per period")
    parser.add_argument("csv", help="Path to cleaned CRM CSV file")
    parser.add_argument("--date-col", default="Date de fin réelle", help="Closing date column")
    parser.add_argument("--status-col", default="Statut commercial", help="Status column")
    parser.add_argument("--won", nargs="*", default=["Won", "Gagnée"], help="Values considered as won")
    parser.add_argument("--amount-col", default="Total recette réalisé", help="Amount column")
    args = parser.parse_args()

    monthly, quarterly, yearly = build_timeseries(
        args.csv,
        date_col=args.date_col,
        status_col=args.status_col,
        won_values=args.won,
        amount_col=args.amount_col,
    )

    print("Monthly revenue:\n", monthly.head(), "\n...\n", monthly.tail())
    print("Quarterly revenue:\n", quarterly.head(), "\n...\n", quarterly.tail())
    print("Yearly revenue:\n", yearly.head(), "\n...\n", yearly.tail())
