"""Fit Prophet models on aggregated revenue time series."""

from __future__ import annotations

from typing import Tuple

import pandas as pd
from prophet import Prophet


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_prophet_df(series: pd.Series) -> pd.DataFrame:
    """Return DataFrame with ``ds`` and ``y`` columns for Prophet."""
    return pd.DataFrame({"ds": series.index, "y": series.values})


# ---------------------------------------------------------------------------
# Core functionality
# ---------------------------------------------------------------------------

def fit_prophet_models(
    monthly: pd.Series,
    quarterly: pd.Series,
    yearly: pd.Series,
) -> Tuple[Prophet, Prophet, Prophet]:
    """Return Prophet models fitted on the three aggregated series.

    Parameters
    ----------
    monthly, quarterly, yearly:
        Preprocessed revenue series resampled to month, quarter and year.

    Notes
    -----
    - Yearly seasonality is enabled for the monthly and quarterly models to
      capture within-year variations.
    - Weekly and daily components are disabled as the data is aggregated at
      higher frequencies.
    - The yearly model only fits a trend (``yearly_seasonality=False``) since a
      single point per year does not allow internal annual cycles.
    """
    # Prepare dataframes for Prophet
    df_month = _to_prophet_df(monthly)
    df_quarter = _to_prophet_df(quarterly)
    df_year = _to_prophet_df(yearly)

    # Instantiate Prophet models with appropriate seasonalities
    model_month = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
    )
    model_quarter = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
    )
    model_year = Prophet(
        yearly_seasonality=False,
        weekly_seasonality=False,
        daily_seasonality=False,
    )

    # Fit the models
    model_month.fit(df_month)
    model_quarter.fit(df_quarter)
    model_year.fit(df_year)

    # The trained models contain a trend component and optional yearly
    # seasonality. Prophet automatically handles changepoints in the trend.
    return model_month, model_quarter, model_year
