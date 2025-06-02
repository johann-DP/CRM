"""Clean and preprocess aggregated revenue time series."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd


def load_and_aggregate(cfg: Dict[str, str]) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Return aggregated revenue series filtered on won opportunities."""

    df = pd.read_csv(
        cfg["csv_path"],
        parse_dates=[cfg["date_col"]],
        dayfirst=True,
        dtype={cfg["amount_col"]: float},
    )

    df = df[df[cfg["status_col"]] == cfg["won_value"]].copy()
    df[cfg["date_col"]] = pd.to_datetime(
        df[cfg["date_col"]], dayfirst=True, errors="coerce"
    )
    df = df.dropna(subset=[cfg["date_col"]])
    df = df.set_index(cfg["date_col"])

    # Remove any observations beyond 1 March 2025 to avoid using future data
    cutoff = pd.Timestamp("2025-03-01")
    df = df.loc[df.index <= cutoff]

    ts_monthly = df[cfg["amount_col"]].resample("M").sum().fillna(0)
    ts_quarterly = df[cfg["amount_col"]].resample("Q").sum().fillna(0)
    ts_yearly = df[cfg["amount_col"]].resample("A").sum().fillna(0)

    return ts_monthly, ts_quarterly, ts_yearly


# ---------------------------------------------------------------------------
# Core functionality
# ---------------------------------------------------------------------------

def preprocess_series(series: pd.Series, *, freq: str) -> pd.Series:
    """Return cleaned version of ``series`` resampled to ``freq``.

    Steps performed:
    - ensure a continuous ``DatetimeIndex`` with ``series.asfreq``;
    - handle missing values via linear interpolation (then fill with 0);
    - detect potential outliers using the IQR method;
    - cap values beyond the outlier thresholds.

    Optionally, a log or differencing transformation could be applied for
    models requiring stationarity (e.g. ARIMA).  This is left commented out
    for now as it depends on the chosen model.
    """
    s = series.copy()
    if not isinstance(s.index, pd.DatetimeIndex):
        s.index = pd.to_datetime(s.index)

    # Ensure the desired frequency and fill missing periods
    s = s.asfreq(freq)

    # ------------------------------------------------------------------
    # Missing values
    # ------------------------------------------------------------------
    if s.isna().any():
        # Linear interpolation handles small gaps; remaining NaNs become 0
        s = s.interpolate(method="linear").fillna(0)

    # ------------------------------------------------------------------
    # Outlier detection (IQR)
    # ------------------------------------------------------------------
    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    # Indices of observations considered outliers
    outliers = (s < lower) | (s > upper)

    if outliers.any():
        # Replace outliers by the threshold (capping)
        s = s.clip(lower=lower, upper=upper)

    # ------------------------------------------------------------------
    # Optional stationarity transform
    # ------------------------------------------------------------------
    # Example: uncomment if log + diff is desired
    # s = np.log1p(s).diff().dropna()

    return s


def preprocess_all(
    monthly: pd.Series,
    quarterly: pd.Series,
    yearly: pd.Series,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Clean monthly, quarterly and yearly time series."""
    monthly_clean = preprocess_series(monthly, freq="ME")
    quarterly_clean = preprocess_series(quarterly, freq="QE")
    yearly_clean = preprocess_series(yearly, freq="YE")
    return monthly_clean, quarterly_clean, yearly_clean


# ---------------------------------------------------------------------------
# Simple CLI / validation helper
# ---------------------------------------------------------------------------

def _summarize(original: pd.Series, cleaned: pd.Series, name: str) -> None:
    print(f"{name} -- missing before: {original.isna().sum()}, after: {cleaned.isna().sum()}")
    print(f"{name} -- mean before: {original.mean():.2f}, after: {cleaned.mean():.2f}")
    print(f"{name} -- std  before: {original.std():.2f}, after: {cleaned.std():.2f}\n")


def main() -> None:  # pragma: no cover - CLI helper
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Preprocess aggregated time series")
    parser.add_argument("csv", help="Path to cleaned CRM CSV file")
    parser.add_argument("--date-col", default="Date de fin actualisée")
    parser.add_argument("--status-col", default="Statut commercial")
    parser.add_argument("--won-value", default="Gagné")
    parser.add_argument("--amount-col", default="Total recette réalisé")
    args = parser.parse_args()

    cfg = {
        "csv_path": Path(args.csv),
        "date_col": args.date_col,
        "status_col": args.status_col,
        "won_value": args.won_value,
        "amount_col": args.amount_col,
    }
    monthly, quarterly, yearly = load_and_aggregate(cfg)
    m_c, q_c, y_c = preprocess_all(monthly, quarterly, yearly)

    _summarize(monthly, m_c, "monthly")
    _summarize(quarterly, q_c, "quarterly")
    _summarize(yearly, y_c, "yearly")


__all__ = ["load_and_aggregate", "preprocess_series", "preprocess_all"]


if __name__ == "__main__":
    main()
