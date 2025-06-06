"""Hyperparameter optimisation helpers for forecasting models."""

from __future__ import annotations

from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from xgboost import XGBRegressor

from .train_xgboost import _to_supervised
from .lstm_forecast import (
    create_lstm_sequences,
    scale_lstm_data,
)


# ---------------------------------------------------------------------------
# ARIMA / SARIMA
# ---------------------------------------------------------------------------

def grid_search_arima(
    series: pd.Series,
    p: Iterable[int],
    d: Iterable[int],
    q: Iterable[int],
    *,
    seasonal: bool = False,
    P: Iterable[int] | None = None,
    D: Iterable[int] | None = None,
    Q: Iterable[int] | None = None,
    m: int = 1,
) -> Tuple[Tuple[int, int, int], Tuple[int, int, int, int] | None, float]:
    """Return best ARIMA orders according to AIC."""
    best_aic = float("inf")
    best_order: Tuple[int, int, int] | None = None
    best_seasonal: Tuple[int, int, int, int] | None = None

    for i in p:
        for j in d:
            for k in q:
                if seasonal:
                    for ip in P or [0]:
                        for id_ in D or [0]:
                            for iq in Q or [0]:
                                try:
                                    model = ARIMA(
                                        series,
                                        order=(i, j, k),
                                        seasonal_order=(ip, id_, iq, m),
                                    )
                                    res = model.fit()
                                    if res.aic < best_aic:
                                        best_aic = res.aic
                                        best_order = (i, j, k)
                                        best_seasonal = (ip, id_, iq, m)
                                except Exception:
                                    continue
                else:
                    try:
                        model = ARIMA(series, order=(i, j, k))
                        res = model.fit()
                        if res.aic < best_aic:
                            best_aic = res.aic
                            best_order = (i, j, k)
                            best_seasonal = None
                    except Exception:
                        continue

    assert best_order is not None
    return best_order, best_seasonal, best_aic


# ---------------------------------------------------------------------------
# Prophet
# ---------------------------------------------------------------------------

def tune_prophet(
    series: pd.Series,
    cps_values: Iterable[float],
    *,
    val_size: int = 6,
    seasonality_mode: Iterable[str] = ("additive",),
) -> Tuple[Dict[str, float], float]:
    """Tune Prophet ``changepoint_prior_scale`` and seasonality mode."""
    train = series.iloc[:-val_size]
    val = series.iloc[-val_size:]
    freq = series.index.freqstr or "M"

    best_params: Dict[str, float] = {}
    best_mape = float("inf")

    for cps in cps_values:
        for mode in seasonality_mode:
            model = Prophet(
                changepoint_prior_scale=cps,
                seasonality_mode=mode,
                yearly_seasonality=True,
                weekly_seasonality=False,
                daily_seasonality=False,
            )
            model.fit(pd.DataFrame({"ds": train.index, "y": train.values}))
            future = model.make_future_dataframe(periods=len(val), freq=freq)
            forecast = model.predict(future)
            preds = forecast.iloc[-len(val):]["yhat"].values
            mape = mean_absolute_percentage_error(val.values, preds)
            if mape < best_mape:
                best_mape = mape
                best_params = {
                    "changepoint_prior_scale": cps,
                    "seasonality_mode": mode,
                }

    return best_params, best_mape


# ---------------------------------------------------------------------------
# XGBoost
# ---------------------------------------------------------------------------

def grid_search_xgb(
    series: pd.Series,
    param_grid: Dict[str, Iterable],
    *,
    n_lags: int,
    add_time_features: bool = True,
    n_splits: int = 3,
) -> Tuple[Dict[str, float], float]:
    """Grid search for :class:`XGBRegressor` using ``TimeSeriesSplit``."""
    X, y = _to_supervised(series, n_lags, add_time_features=add_time_features)
    tscv = TimeSeriesSplit(n_splits=n_splits)
    model = XGBRegressor(objective="reg:squarederror", random_state=42)
    grid = GridSearchCV(
        model,
        param_grid,
        cv=tscv,
        scoring="neg_mean_squared_error",
    )
    grid.fit(X, y)
    best_params = grid.best_params_
    best_rmse = (-grid.best_score_) ** 0.5
    return best_params, best_rmse


# ---------------------------------------------------------------------------
# LSTM
# ---------------------------------------------------------------------------

def tune_lstm(
    series: pd.Series,
    units: Iterable[int],
    epochs: Iterable[int],
    *,
    window: int,
    val_size: int = 6,
) -> Tuple[Dict[str, int], float]:
    """Return best number of units and epochs for an LSTM."""
    train = series.iloc[:-val_size]
    val = series.iloc[-val_size:]

    X_train, y_train = create_lstm_sequences(train, window)
    X_train_s, y_train_s, scaler = scale_lstm_data(X_train, y_train)

    # prepare validation sequences from end of train + val
    tail = pd.concat([train.iloc[-window:], val])
    X_val, y_val = create_lstm_sequences(tail, window)
    X_val_s = scaler.transform(X_val.reshape(-1, 1)).reshape(X_val.shape)

    from keras.models import Sequential
    from keras.layers import LSTM, Dense

    best_mape = float("inf")
    best = {}

    for u in units:
        for e in epochs:
            model = Sequential()
            model.add(LSTM(u, input_shape=(window, 1)))
            model.add(Dense(1))
            model.compile(loss="mse", optimizer="adam")
            model.fit(
                X_train_s,
                y_train_s,
                epochs=e,
                batch_size=16,
                validation_split=0.1,
                verbose=0,
            )
            preds = model.predict(X_val_s, verbose=0).reshape(-1, 1)
            preds = scaler.inverse_transform(preds).reshape(-1)
            mape = np.mean(np.abs((y_val - preds) / y_val)) * 100
            if mape < best_mape:
                best_mape = mape
                best = {"units": u, "epochs": e}

    return best, best_mape


__all__ = [
    "grid_search_arima",
    "tune_prophet",
    "grid_search_xgb",
    "tune_lstm",
]
