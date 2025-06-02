"""Forecast future aggregated revenue using trained models."""

from __future__ import annotations

import pandas as pd
try:  # pragma: no cover - optional dependency
    from pmdarima.arima import ARIMA
except Exception:  # pragma: no cover - handle missing lib
    ARIMA = None
try:  # pragma: no cover - optional dependency
    from prophet import Prophet
except Exception:  # pragma: no cover - handle missing lib
    Prophet = None
from xgboost import XGBRegressor
from sklearn.preprocessing import MinMaxScaler

from .train_xgboost import _to_supervised


# ---------------------------------------------------------------------------
# ARIMA / SARIMA forecast
# ---------------------------------------------------------------------------

def forecast_arima(model: ARIMA, series: pd.Series, periods: int) -> pd.DataFrame:
    """Return ARIMA predictions with confidence intervals."""
    if ARIMA is None:
        raise ImportError("pmdarima is required for ARIMA forecasts")
    freq = series.index.freq or pd.infer_freq(series.index)
    if freq is None:
        raise ValueError("Series index must have a frequency")
    start = series.index[-1] + pd.tseries.frequencies.to_offset(freq)
    idx = pd.date_range(start=start, periods=periods, freq=freq)

    preds, conf = model.predict(n_periods=periods, return_conf_int=True)
    return pd.DataFrame(
        {
            "forecast": preds,
            "lower_ci": conf[:, 0],
            "upper_ci": conf[:, 1],
        },
        index=idx,
    )


# ---------------------------------------------------------------------------
# Prophet forecast
# ---------------------------------------------------------------------------

def forecast_prophet(model: Prophet, series: pd.Series, periods: int) -> pd.DataFrame:
    """Return Prophet predictions with confidence intervals."""
    if Prophet is None:
        raise ImportError("prophet is required for Prophet forecasts")
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
    model: XGBRegressor,
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
    preds, lower, upper = [], [], []
    for i in range(periods):
        X_hist, _ = _to_supervised(history, n_lags, add_time_features=add_time_features)
        X_pred = X_hist.iloc[-1:]
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
