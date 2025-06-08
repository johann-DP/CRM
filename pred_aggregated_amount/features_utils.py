"""Helper utilities for supervised time series features."""

from __future__ import annotations

import pandas as pd


def make_lag_features(
    series: pd.Series,
    n_lags: int,
    freq: str,
    add_time_cat: bool,
) -> pd.DataFrame:
    """Return dataframe with target, lag features and optional time category.

    Parameters
    ----------
    series : pd.Series
        Time series indexed by ``DatetimeIndex``.
    n_lags : int
        Number of past observations to use as predictors.
    freq : str
        Frequency (``"M"``, ``"Q"`` or ``"A"``) used for the categorical time
        feature.
    add_time_cat : bool
        If ``True`` add the month/quarter/year categorical feature.
    """
    df = series.to_frame(name="y")
    for lag in range(1, n_lags + 1):
        df[f"lag{lag}"] = series.shift(lag)

    if add_time_cat:
        if freq == "M":
            df["month"] = series.index.month
        elif freq == "Q":
            df["quarter"] = series.index.quarter
        elif freq == "A":
            df["year"] = series.index.year
        else:  # pragma: no cover - invalid frequency
            raise ValueError("freq must be 'M', 'Q' or 'A'")

    df = df.dropna().copy()
    return df


__all__ = ["make_lag_features"]
