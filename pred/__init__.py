"""Prediction utilities for revenue time series."""

from .aggregate_revenue import (
    load_won_opportunities,
    aggregate_revenue,
    build_timeseries,
)
from .preprocess_timeseries import preprocess_series, preprocess_all
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
        raise ImportError("pmdarima is required for fit_all_arima") from _exc_arima


from .train_xgboost import train_xgb_model, train_all_granularities
from .compare_granularities import build_performance_table, plot_metric_comparison
from .future_forecast import (
    forecast_arima,
    forecast_prophet,
    forecast_xgb,
    forecast_lstm,
)

__all__ = [
    "load_won_opportunities",
    "aggregate_revenue",
    "build_timeseries",
    "preprocess_series",
    "preprocess_all",
    "fit_prophet_models",
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
    "forecast_prophet",
    "forecast_xgb",
    "forecast_lstm",
]
