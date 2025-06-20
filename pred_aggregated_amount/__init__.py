"""Prediction utilities for revenue time series."""

from .aggregate_revenue import (
    load_won_opportunities,
    aggregate_revenue,
    build_timeseries,
)
from .preprocess_timeseries import (
    preprocess_series,
    preprocess_all,
)
from .preprocess_dates import preprocess_dates
from .lstm_forecast import (
    create_lstm_sequences,
    scale_lstm_data,
    build_lstm_model,
    train_lstm_model,
)
from .train_xgboost import train_xgb_model
from .compare_granularities import build_performance_table
from .future_forecast import (
    forecast_arima,
    forecast_xgb,
    forecast_lstm,
)
from .catboost_forecast import (
    rolling_forecast_catboost,
    forecast_future_catboost,
)
from .features_utils import make_lag_features


def fit_prophet_models(*args, **kwargs):
    from .prophet_models import fit_prophet_models as _fit

    return _fit(*args, **kwargs)


def fit_all_arima(*args, **kwargs):
    from .train_arima import fit_all_arima as _fit

    return _fit(*args, **kwargs)


def forecast_prophet(*args, **kwargs):
    from .future_forecast import forecast_prophet as _forecast

    return _forecast(*args, **kwargs)

__all__ = [
    "load_won_opportunities",
    "aggregate_revenue",
    "build_timeseries",
    "preprocess_series",
    "preprocess_all",
    "preprocess_dates",
    "fit_all_arima",
    "train_xgb_model",
    "create_lstm_sequences",
    "scale_lstm_data",
    "build_lstm_model",
    "train_lstm_model",
    "build_performance_table",
    "forecast_arima",
    "forecast_xgb",
    "forecast_lstm",
    "make_lag_features",
    "rolling_forecast_catboost",
    "forecast_future_catboost",
    "fit_prophet_models",
    "forecast_prophet",
]
