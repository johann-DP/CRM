"""Utilities to load and align external open data series."""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd


def load_external_series(
    path: str | Path,
    *,
    date_col: str,
    value_col: str,
    freq: str = "M",
) -> pd.Series:
    """Return a monthly series from the CSV at ``path``.

    Parameters
    ----------
    path:
        Local path or URL to the CSV file.
    date_col:
        Name of the column containing the dates.
    value_col:
        Column with the numeric values to use.
    freq:
        Resampling frequency (default: monthly ``"M"``).
    """
    df = pd.read_csv(path)
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])
    series = df.set_index(date_col)[value_col].astype(float)
    series = series.sort_index().asfreq(freq)
    return series


def align_exogenous(
    target: pd.Series, exog: Dict[str, pd.Series], *, method: str = "linear"
) -> pd.DataFrame:
    """Return DataFrame of exogenous variables aligned on ``target`` index."""
    df = pd.DataFrame(index=target.index)
    for name, ser in exog.items():
        aligned = ser.reindex(target.index)
        if aligned.isna().any():
            aligned = aligned.interpolate(method=method).fillna(method="bfill").fillna(method="ffill")
        df[name] = aligned
    return df


def merge_target_exog(target: pd.Series, exog: Dict[str, pd.Series]) -> pd.DataFrame:
    """Return DataFrame with target and aligned exogenous variables."""
    exog_df = align_exogenous(target, exog)
    exog_df.insert(0, "y", target)
    return exog_df


__all__ = ["load_external_series", "align_exogenous", "merge_target_exog"]
