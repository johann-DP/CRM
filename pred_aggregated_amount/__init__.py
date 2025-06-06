"""Prediction utilities for revenue time series."""

from .aggregate_revenue import (
    load_won_opportunities,
    aggregate_revenue,
    build_timeseries,
)
from .preprocess_timeseries import (
    load_and_aggregate,
    preprocess_series,
    preprocess_all,
)
from .preprocess_dates import preprocess_dates
from .lstm_forecast import (
    create_lstm_sequences,
    scale_lstm_data,
    build_lstm_model,
    train_lstm_model,
    quick_predict_check,
)

try:  # Optional dependency
    from .prophet_models import fit_prophet_models  # type: ignore
except Exception as _exc_prophet:  # pragma: no cover - optional

    def fit_prophet_models(*_a, **_k):
        raise ImportError(
            "prophet is required for fit_prophet_models"  # noqa: B904
        ) from _exc_prophet


try:  # Optional dependency
    from .train_arima import fit_all_arima  # type: ignore
except Exception as _exc_arima:  # pragma: no cover - optional

    def fit_all_arima(*_a, **_k):
        raise ImportError("statsforecast is required for fit_all_arima") from _exc_arima


from .train_xgboost import train_xgb_model, train_all_granularities
from .compare_granularities import build_performance_table, plot_metric_comparison

# Prophet-related helpers are optional to avoid hard dependency during tests
try:  # pragma: no cover - import may fail when Prophet is missing
    from .prophet_models import fit_prophet_models
    from .future_forecast import forecast_prophet
except Exception:  # pragma: no cover - keep usable without Prophet
    fit_prophet_models = None
    forecast_prophet = None
from .future_forecast import (
    forecast_arima,
    forecast_xgb,
    forecast_lstm,
)
from .catboost_forecast import (
    prepare_supervised,
    rolling_forecast_catboost,
    forecast_future_catboost,
)

__all__ = [
    "load_won_opportunities",
    "aggregate_revenue",
    "build_timeseries",
    "load_and_aggregate",
    "preprocess_series",
    "preprocess_all",
    "preprocess_dates",
    "fit_all_arima",
    "train_xgb_model",
    "train_all_granularities",
    "create_lstm_sequences",
    "scale_lstm_data",
    "build_lstm_model",
    "train_lstm_model",
    "quick_predict_check",
    "build_performance_table",
    "plot_metric_comparison",
    "forecast_arima",
    "forecast_xgb",
    "forecast_lstm",
    "prepare_supervised",
    "rolling_forecast_catboost",
    "forecast_future_catboost",
]

if fit_prophet_models is not None:
    __all__.insert(6, "fit_prophet_models")

if forecast_prophet is not None:
    __all__.insert(-3, "forecast_prophet")
