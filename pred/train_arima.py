"""Train ARIMA/SARIMA models on preprocessed revenue time series."""

from __future__ import annotations

from typing import Tuple

import pandas as pd

# ``auto_arima`` performs a grid-search over different (p, d, q) orders and
# optionally seasonal (P, D, Q, m) orders to minimise the AIC.  It returns an
# already fitted model.
try:  # Optional dependency
    from pmdarima import auto_arima
except Exception as _exc_arima:  # pragma: no cover - optional
    auto_arima = None

# Optionally imported so that ``summary()`` outputs the standard statsmodels
# results table.
try:  # pragma: no cover - optional dependency
    import statsmodels.api as sm  # noqa: F401  # used indirectly by auto_arima
except Exception:  # pragma: no cover - ignore if not installed
    sm = None


# ---------------------------------------------------------------------------
# Core functionality
# ---------------------------------------------------------------------------


def _fit_series(series: pd.Series, *, seasonal: bool, m: int) -> auto_arima:
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
    if auto_arima is None:
        raise ImportError("pmdarima is required for ARIMA models") from _exc_arima

    model = auto_arima(
        series,
        seasonal=seasonal,
        m=m,
        trace=False,
        error_action="ignore",
        suppress_warnings=True,
        stepwise=True,
    )
    return model


def fit_all_arima(
    monthly: pd.Series,
    quarterly: pd.Series,
    yearly: pd.Series,
) -> Tuple[auto_arima, auto_arima, auto_arima]:
    """Fit ARIMA/SARIMA models for monthly, quarterly and yearly series."""
    # Monthly data has an obvious yearly cycle -> SARIMA with m=12
    model_monthly = _fit_series(monthly, seasonal=True, m=12)

    # Quarterly data repeats every 4 quarters -> SARIMA with m=4
    model_quarterly = _fit_series(quarterly, seasonal=True, m=4)

    # Yearly data has too few points for a seasonal component -> plain ARIMA
    model_yearly = _fit_series(yearly, seasonal=False, m=1)

    # Display a summary of each fitted model (orders and AIC)
    print(
        "Monthly model:",
        f"ARIMA{model_monthly.order}x{model_monthly.seasonal_order}",
        f"AIC={model_monthly.aic():.2f}",
    )
    print(model_monthly.summary())

    print(
        "Quarterly model:",
        f"ARIMA{model_quarterly.order}x{model_quarterly.seasonal_order}",
        f"AIC={model_quarterly.aic():.2f}",
    )
    print(model_quarterly.summary())

    print(
        "Yearly model:",
        f"ARIMA{model_yearly.order}x{model_yearly.seasonal_order}",
        f"AIC={model_yearly.aic():.2f}",
    )
    print(model_yearly.summary())

    return model_monthly, model_quarterly, model_yearly


# ---------------------------------------------------------------------------
# Notes
# ---------------------------------------------------------------------------
# ARIMA models are defined by the non-seasonal orders (p, d, q) controlling
# the autoregressive, differencing and moving-average parts.  SARIMA extends
# this with seasonal orders (P, D, Q, m) where ``m`` is the length of the
# seasonal cycle.  ``auto_arima`` explores combinations of these parameters and
# selects the best model according to the Akaike Information Criterion (AIC).
#
# The fitted models returned here are trained on the entire history of each
# time series.  They can be saved (for instance with ``pickle``) and reused
# later for evaluation and forecasting.
