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
from .compare_granularities import build_performance_table, plot_metric_comparison

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
]
