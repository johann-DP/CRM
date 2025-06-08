"""Forecast future aggregated revenue using trained models."""

from __future__ import annotations

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from typing import TYPE_CHECKING

from .features_utils import make_lag_features

if TYPE_CHECKING:  # pragma: no cover - optional type hints
    from prophet import Prophet
    from statsforecast.models import AutoARIMA
    from xgboost import XGBRegressor

AutoARIMA = None
Prophet = None
XGBRegressor = None


# ---------------------------------------------------------------------------
# ARIMA / SARIMA forecast
# ---------------------------------------------------------------------------


def forecast_arima(model: "AutoARIMA", series: pd.Series, periods: int) -> pd.DataFrame:
    """Return ARIMA predictions with confidence intervals."""
    freq = series.index.freq or pd.infer_freq(series.index)
    if freq is None:
        raise ValueError("Series index must have a frequency")
    start = series.index[-1] + pd.tseries.frequencies.to_offset(freq)
    idx = pd.date_range(start=start, periods=periods, freq=freq)

    res = model.predict(h=periods, level=[95])
    if isinstance(res, pd.DataFrame):
        preds = res["mean"].to_numpy()
        lower = res.get("lo-95", pd.Series([float("nan")] * periods)).to_numpy()
        upper = res.get("hi-95", pd.Series([float("nan")] * periods)).to_numpy()
    else:  # pragma: no cover - fallback when predict does not return DataFrame
        preds = res
        lower = [float("nan")] * periods
        upper = [float("nan")] * periods
    return pd.DataFrame(
        {"forecast": preds, "lower_ci": lower, "upper_ci": upper}, index=idx
    )


# ---------------------------------------------------------------------------
# Prophet forecast
# ---------------------------------------------------------------------------


def forecast_prophet(model: "Prophet", series: pd.Series, periods: int) -> pd.DataFrame:
    """Return Prophet predictions with confidence intervals."""
    freq = series.index.freq or pd.infer_freq(series.index) or "M"
    future = model.make_future_dataframe(periods=periods, freq=freq)
    forecast = model.predict(future)
    subset = forecast.tail(periods)[["ds", "yhat", "yhat_lower", "yhat_upper"]]
    subset.columns = ["ds", "forecast", "lower_ci", "upper_ci"]
    subset = subset.set_index("ds")
    return subset


# ---------------------------------------------------------------------------
# XGBoost forecast (approximate interval)
# ---------------------------------------------------------------------------


def forecast_xgb(
    model: "XGBRegressor",
    series: pd.Series,
    periods: int,
    *,
    n_lags: int,
    rmse: float,
    add_time_features: bool = True,
) -> pd.DataFrame:
    """Iteratively forecast with XGBoost and approximate confidence bounds."""
    freq = series.index.freq or pd.infer_freq(series.index)
    if freq is None:
        raise ValueError("Series index must have a frequency")
    start = series.index[-1] + pd.tseries.frequencies.to_offset(freq)
    idx = pd.date_range(start=start, periods=periods, freq=freq)

    history = series.copy()
    freq_str = series.index.freqstr or pd.infer_freq(series.index) or "M"
    if freq_str.startswith("Q"):
        freq_base = "Q"
    elif freq_str.startswith("A"):
        freq_base = "A"
    else:
        freq_base = "M"

    preds, lower, upper = [], [], []
    for i in range(periods):
        df_sup = make_lag_features(history, n_lags, freq_base, add_time_features)
        X_pred = df_sup.drop(columns=["y"]).iloc[-1:]
        pred = float(model.predict(X_pred)[0])
        preds.append(pred)
        lower.append(pred - 1.96 * rmse)
        upper.append(pred + 1.96 * rmse)
        history.loc[idx[i]] = pred

    return pd.DataFrame(
        {"forecast": preds, "lower_ci": lower, "upper_ci": upper}, index=idx
    )


# ---------------------------------------------------------------------------
# LSTM forecast (approximate interval)
# ---------------------------------------------------------------------------


def forecast_lstm(
    model,
    scaler: MinMaxScaler,
    series: pd.Series,
    periods: int,
    *,
    window_size: int,
    rmse: float,
) -> pd.DataFrame:
    """Iteratively forecast with LSTM and approximate confidence bounds."""
    freq = series.index.freq or pd.infer_freq(series.index)
    if freq is None:
        raise ValueError("Series index must have a frequency")
    start = series.index[-1] + pd.tseries.frequencies.to_offset(freq)
    idx = pd.date_range(start=start, periods=periods, freq=freq)

    history = series.copy()
    preds, lower, upper = [], [], []
    for i in range(periods):
        seq = history.values[-window_size:].reshape(1, window_size, 1)
        seq_s = scaler.transform(seq.reshape(-1, 1)).reshape(1, window_size, 1)
        pred_s = model.predict(seq_s, verbose=0)[0, 0]
        pred = scaler.inverse_transform([[pred_s]])[0, 0]
        preds.append(float(pred))
        lower.append(pred - 1.96 * rmse)
        upper.append(pred + 1.96 * rmse)
        history.loc[idx[i]] = pred

    return pd.DataFrame(
        {"forecast": preds, "lower_ci": lower, "upper_ci": upper}, index=idx
    )


__all__ = [
    "forecast_arima",
    "forecast_prophet",
    "forecast_xgb",
    "forecast_lstm",
]
