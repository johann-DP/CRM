"""Train ARIMA/SARIMA models on preprocessed revenue time series."""

from __future__ import annotations

from typing import Tuple

import pandas as pd

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - type hints
    from statsforecast.models import AutoARIMA


# ---------------------------------------------------------------------------
# Core functionality
# ---------------------------------------------------------------------------


def _fit_series(series: pd.Series, *, seasonal: bool, m: int) -> "AutoARIMA":
    """Return the best ARIMA/SARIMA model for ``series``.

    Parameters
    ----------
    series : pd.Series
        Time series to model. It should be preprocessed and indexed by a
        ``DatetimeIndex`` with the desired frequency.
    seasonal : bool
        Whether to include a seasonal component (SARIMA).  When ``True`` the
        seasonal period ``m`` is also considered during the search.
    m : int
        Number of observations per cycle for the seasonal component.
    """
    from statsforecast.models import AutoARIMA

    season_length = m if seasonal else 1
    model = AutoARIMA(season_length=season_length)
    model.fit(series.values)
    return model


def fit_all_arima(
    monthly: pd.Series,
    quarterly: pd.Series,
    yearly: pd.Series,
) -> Tuple["AutoARIMA", "AutoARIMA", "AutoARIMA"]:
    """Fit ARIMA/SARIMA models for monthly, quarterly and yearly series."""
    # Monthly data has an obvious yearly cycle -> SARIMA with m=12
    model_monthly = _fit_series(monthly, seasonal=True, m=12)

    # Quarterly data repeats every 4 quarters -> SARIMA with m=4
    model_quarterly = _fit_series(quarterly, seasonal=True, m=4)

    # Yearly data has too few points for a seasonal component -> plain ARIMA
    model_yearly = _fit_series(yearly, seasonal=False, m=1)

    # Display a basic summary of each fitted model
    print("Monthly model:", model_monthly)
    print("Quarterly model:", model_quarterly)
    print("Yearly model:", model_yearly)

    return model_monthly, model_quarterly, model_yearly


# ---------------------------------------------------------------------------
# Notes
# ---------------------------------------------------------------------------
# ARIMA models are defined by the non-seasonal orders (p, d, q) controlling
# the autoregressive, differencing and moving-average parts.  SARIMA extends
# this with seasonal orders (P, D, Q, m) where ``m`` is the length of the
# seasonal cycle.  ``AutoARIMA`` explores combinations of these parameters and
# selects the best model according to the Akaike Information Criterion (AIC).
#
# The fitted models returned here are trained on the entire history of each
# time series.  They can be saved (for instance with ``pickle``) and reused
# later for evaluation and forecasting.
