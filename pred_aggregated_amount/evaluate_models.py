"""Rolling forecast evaluation for time series models."""

from __future__ import annotations

from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit


def safe_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Return MAPE ignoring zero ``y_true`` values."""
    mask = y_true != 0
    if mask.sum() == 0:
        return np.nan
    return np.mean(np.abs((y_pred[mask] - y_true[mask]) / y_true[mask])) * 100



from statsforecast.models import AutoARIMA
from prophet import Prophet
from xgboost import XGBRegressor
from .catboost_forecast import prepare_supervised, rolling_forecast_catboost

from .train_xgboost import _to_supervised
from .lstm_forecast import create_lstm_sequences, scale_lstm_data, build_lstm_model


# ---------------------------------------------------------------------------
# Metric helper
# ---------------------------------------------------------------------------

def _compute_metrics(true: np.ndarray, pred: List[float]) -> Dict[str, float]:
    """Return MAE, RMSE and MAPE between ``true`` and ``pred``."""
    true_a = np.asarray(true, dtype=float)
    pred_a = np.asarray(pred, dtype=float)
    mae = mean_absolute_error(true_a, pred_a)
    rmse = mean_squared_error(true_a, pred_a) ** 0.5
    mape = safe_mape(true_a, pred_a)
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape}


# ---------------------------------------------------------------------------
# ARIMA rolling forecast
# ---------------------------------------------------------------------------

def _evaluate_arima(series: pd.Series, test_size: int, *, seasonal: bool, m: int) -> Dict[str, float]:
    """Evaluate ARIMA with rolling forecast."""
    train = series.iloc[:-test_size]
    test = series.iloc[-test_size:]
    history = train.copy()
    preds: List[float] = []
    for t, val in enumerate(test):
        season_length = m if seasonal else 1
        model = AutoARIMA(season_length=season_length)
        model.fit(history.values)
        res = model.predict(h=1)
        if isinstance(res, dict):
            pred = float(res["mean"][0])
        elif hasattr(res, "__getitem__"):
            try:
                pred = float(res[0])
            except Exception:  # pragma: no cover - unexpected format
                pred = float(res["mean"].iloc[0])
        else:  # pragma: no cover - fallback
            pred = float(res)
        preds.append(pred)
        history.loc[test.index[t]] = val
    return _compute_metrics(test.values, preds)


# ---------------------------------------------------------------------------
# Prophet rolling forecast
# ---------------------------------------------------------------------------

def _prophet_df(series: pd.Series) -> pd.DataFrame:
    return pd.DataFrame({"ds": series.index, "y": series.values})


def _evaluate_prophet(series: pd.Series, test_size: int, *, yearly_seasonality: bool) -> Dict[str, float]:
    """Evaluate Prophet with rolling retraining."""
    train = series.iloc[:-test_size]
    test = series.iloc[-test_size:]
    history = train.copy()
    preds: List[float] = []
    freq = series.index.freqstr or "ME"
    for t, val in enumerate(test):
        model = Prophet(yearly_seasonality=yearly_seasonality, weekly_seasonality=False, daily_seasonality=False)
        model.fit(_prophet_df(history))
        future = model.make_future_dataframe(periods=1, freq=freq)
        forecast = model.predict(future)
        pred = forecast.iloc[-1]["yhat"]
        preds.append(pred)
        history.loc[test.index[t]] = val
    return _compute_metrics(test.values, preds)


# ---------------------------------------------------------------------------
# XGBoost rolling forecast
# ---------------------------------------------------------------------------

def _evaluate_xgb(series: pd.Series, test_size: int, *, n_lags: int, add_time_features: bool) -> Dict[str, float]:
    """Evaluate XGBoost model with rolling forecast."""
    train = series.iloc[:-test_size]
    test = series.iloc[-test_size:]
    history = train.copy()
    preds: List[float] = []

    for t, val in enumerate(test):
        X_train, y_train = _to_supervised(history, n_lags, add_time_features=add_time_features)
        model = XGBRegressor(
            objective="reg:squarederror",
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            random_state=42,
        )
        model.fit(X_train, y_train)
        # Build features for next period
        next_index = test.index[t]
        extended = pd.concat([history, pd.Series([np.nan], index=[next_index])])
        extended = extended.asfreq(series.index.freq)
        X_full, _ = _to_supervised(extended, n_lags, add_time_features=add_time_features)
        X_pred = X_full.iloc[-1:]
        pred = float(model.predict(X_pred)[0])
        preds.append(pred)
        history.loc[next_index] = val
    return _compute_metrics(test.values, preds)


# ---------------------------------------------------------------------------
# LSTM rolling forecast
# ---------------------------------------------------------------------------

def _evaluate_lstm(
    series: pd.Series,
    test_size: int,
    *,
    window: int,
    epochs: int = 50,
    update_epochs: int = 5,
) -> Dict[str, float]:
    """Evaluate LSTM with rolling forecast and iterative fine-tuning."""
    train = series.iloc[:-test_size]
    test = series.iloc[-test_size:]
    X, y = create_lstm_sequences(train, window)
    X_scaled, y_scaled, scaler = scale_lstm_data(X, y)
    model = build_lstm_model(window)
    model.fit(
        X_scaled,
        y_scaled,
        epochs=epochs,
        batch_size=16,
        validation_split=0.1,
        verbose=0,
    )
    history = train.copy()
    preds: List[float] = []
    for t, val in enumerate(test):
        seq = history.values[-window:].reshape(1, window, 1)
        seq_scaled = scaler.transform(seq.reshape(-1, 1)).reshape(1, window, 1)
        pred_scaled = model.predict(seq_scaled, verbose=0)[0, 0]
        pred = scaler.inverse_transform([[pred_scaled]])[0, 0]
        preds.append(float(pred))
        history.loc[test.index[t]] = val

        # Fine-tune the model on the updated history
        X_hist, y_hist = create_lstm_sequences(history, window)
        X_hist_s = scaler.transform(X_hist.reshape(-1, 1)).reshape(X_hist.shape)
        y_hist_s = scaler.transform(y_hist.reshape(-1, 1)).reshape(-1)
        model.fit(
            X_hist_s,
            y_hist_s,
            epochs=update_epochs,
            batch_size=16,
            verbose=0,
        )
    return _compute_metrics(test.values, preds)


# ---------------------------------------------------------------------------
# CatBoost rolling forecast
# ---------------------------------------------------------------------------

def _evaluate_catboost(
    series: pd.Series, freq: str, *, test_size: int | None = None
) -> Dict[str, float]:
    """Evaluate CatBoost model with rolling forecast."""
    if series.nunique() == 1:
        n_test = test_size or (12 if freq == "M" else (4 if freq == "Q" else 2))
        const_val = float(series.iloc[-1])
        preds = [const_val] * n_test
        actuals = [const_val] * n_test
        return _compute_metrics(actuals, preds)

    df_sup = prepare_supervised(series, freq)
    preds, actuals = rolling_forecast_catboost(df_sup, freq, test_size=test_size)
    return _compute_metrics(actuals, preds)


# ---------------------------------------------------------------------------
# Time series cross-validation
# ---------------------------------------------------------------------------

def _ts_cross_val(
    series: pd.Series,
    eval_fn: Callable[..., Dict[str, float]],
    *,
    n_splits: int = 5,
    **kwargs,
) -> Dict[str, float]:
    """Return mean/std metrics over ``n_splits`` time series folds."""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_metrics: List[Dict[str, float]] = []
    for _, test_idx in tscv.split(series):
        end = test_idx[-1] + 1
        subset = series.iloc[:end]
        metrics = eval_fn(subset, len(test_idx), **kwargs)
        fold_metrics.append(metrics)

    df = pd.DataFrame(fold_metrics)
    mean = df.mean()
    std = df.std()
    out = {k: float(mean[k]) for k in df.columns}
    out.update({f"{k}_std": float(std[k]) for k in df.columns})
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------






# ---------------------------------------------------------------------------
# Rolling forecast returning predictions
# ---------------------------------------------------------------------------

def _rolling_preds_arima(series: pd.Series, test_size: int, *, seasonal: bool, m: int) -> Tuple[pd.Series, pd.Series]:
    """Return predicted and true values of a rolling ARIMA forecast."""
    train = series.iloc[:-test_size]
    test = series.iloc[-test_size:]
    history = train.copy()
    preds: List[float] = []
    for t, val in enumerate(test):
        season_length = m if seasonal else 1
        model = AutoARIMA(season_length=season_length)
        model.fit(history.values)
        res = model.predict(h=1)
        if isinstance(res, dict):
            pred = float(res["mean"][0])
        elif hasattr(res, "__getitem__"):
            try:
                pred = float(res[0])
            except Exception:  # pragma: no cover - unexpected format
                pred = float(res["mean"].iloc[0])
        else:  # pragma: no cover - fallback
            pred = float(res)
        preds.append(pred)
        history.loc[test.index[t]] = val
    return pd.Series(preds, index=test.index), test


def _rolling_preds_prophet(series: pd.Series, test_size: int, *, yearly_seasonality: bool) -> Tuple[pd.Series, pd.Series]:
    """Return predicted and true values of a rolling Prophet forecast."""
    train = series.iloc[:-test_size]
    test = series.iloc[-test_size:]
    history = train.copy()
    preds: List[float] = []
    freq = series.index.freqstr or "ME"
    for t, val in enumerate(test):
        model = Prophet(yearly_seasonality=yearly_seasonality, weekly_seasonality=False, daily_seasonality=False)
        model.fit(_prophet_df(history))
        future = model.make_future_dataframe(periods=1, freq=freq)
        forecast = model.predict(future)
        pred = forecast.iloc[-1]["yhat"]
        preds.append(pred)
        history.loc[test.index[t]] = val
    return pd.Series(preds, index=test.index), test


def _rolling_preds_xgb(series: pd.Series, test_size: int, *, n_lags: int, add_time_features: bool) -> Tuple[pd.Series, pd.Series]:
    """Return predicted and true values of a rolling XGBoost forecast."""
    train = series.iloc[:-test_size]
    test = series.iloc[-test_size:]
    history = train.copy()
    preds: List[float] = []
    for t, val in enumerate(test):
        X_train, y_train = _to_supervised(history, n_lags, add_time_features=add_time_features)
        model = XGBRegressor(
            objective="reg:squarederror",
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            random_state=42,
        )
        model.fit(X_train, y_train)
        next_index = test.index[t]
        extended = pd.concat([history, pd.Series([np.nan], index=[next_index])])
        extended = extended.asfreq(series.index.freq)
        X_full, _ = _to_supervised(extended, n_lags, add_time_features=add_time_features)
        X_pred = X_full.iloc[-1:]
        pred = float(model.predict(X_pred)[0])
        preds.append(pred)
        history.loc[next_index] = val
    return pd.Series(preds, index=test.index), test


def _rolling_preds_lstm(
    series: pd.Series,
    test_size: int,
    *,
    window: int,
    epochs: int = 50,
    update_epochs: int = 5,
) -> Tuple[pd.Series, pd.Series]:
    """Return predicted and true values of a rolling LSTM forecast with fine-tuning."""
    train = series.iloc[:-test_size]
    test = series.iloc[-test_size:]
    X, y = create_lstm_sequences(train, window)
    X_scaled, y_scaled, scaler = scale_lstm_data(X, y)
    model = build_lstm_model(window)
    model.fit(
        X_scaled,
        y_scaled,
        epochs=epochs,
        batch_size=16,
        validation_split=0.1,
        verbose=0,
    )
    history = train.copy()
    preds: List[float] = []
    for t, val in enumerate(test):
        seq = history.values[-window:].reshape(1, window, 1)
        seq_scaled = scaler.transform(seq.reshape(-1, 1)).reshape(1, window, 1)
        pred_scaled = model.predict(seq_scaled, verbose=0)[0, 0]
        pred = scaler.inverse_transform([[pred_scaled]])[0, 0]
        preds.append(float(pred))
        history.loc[test.index[t]] = val

        # Fine-tune the model on the updated history
        X_hist, y_hist = create_lstm_sequences(history, window)
        X_hist_s = scaler.transform(X_hist.reshape(-1, 1)).reshape(X_hist.shape)
        y_hist_s = scaler.transform(y_hist.reshape(-1, 1)).reshape(-1)
        model.fit(
            X_hist_s,
            y_hist_s,
            epochs=update_epochs,
            batch_size=16,
            verbose=0,
        )
    return pd.Series(preds, index=test.index), test


def _rolling_preds_catboost(series: pd.Series, freq: str) -> Tuple[pd.Series, pd.Series]:
    """Return predicted and true values for CatBoost rolling forecast."""
    df_sup = prepare_supervised(series, freq)
    preds, actuals = rolling_forecast_catboost(df_sup, freq)
    n = len(preds)
    test_index = df_sup.index[-n:]
    return pd.Series(preds, index=test_index), pd.Series(actuals, index=test_index)




