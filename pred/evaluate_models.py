"""Rolling forecast evaluation for time series models."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


def safe_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Return MAPE ignoring zero ``y_true`` values."""
    mask = y_true != 0
    if mask.sum() == 0:
        return np.nan
    return np.mean(np.abs((y_pred[mask] - y_true[mask]) / y_true[mask])) * 100



from statsforecast.models import AutoARIMA
from prophet import Prophet
from xgboost import XGBRegressor

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

def _evaluate_lstm(series: pd.Series, test_size: int, *, window: int, epochs: int = 50) -> Dict[str, float]:
    """Evaluate LSTM with rolling forecast (no retraining during test)."""
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
    return _compute_metrics(test.values, preds)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def evaluate_all_models(
    monthly: pd.Series,
    quarterly: pd.Series,
    yearly: pd.Series,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Return MAE, RMSE and MAPE for all models and granularities."""
    results: Dict[str, Dict[str, Dict[str, float]]] = {
        "ARIMA": {"monthly": {}, "quarterly": {}, "yearly": {}},
        "Prophet": {"monthly": {}, "quarterly": {}, "yearly": {}},
        "XGBoost": {"monthly": {}, "quarterly": {}, "yearly": {}},
        "LSTM": {"monthly": {}, "quarterly": {}, "yearly": {}},
    }

    # Monthly: last 12 months for testing
    results["ARIMA"]["monthly"] = _evaluate_arima(monthly, 12, seasonal=True, m=12)
    results["Prophet"]["monthly"] = _evaluate_prophet(monthly, 12, yearly_seasonality=True)
    results["XGBoost"]["monthly"] = _evaluate_xgb(monthly, 12, n_lags=12, add_time_features=True)
    results["LSTM"]["monthly"] = _evaluate_lstm(monthly, 12, window=12)

    # Quarterly: last 4 quarters for testing
    results["ARIMA"]["quarterly"] = _evaluate_arima(quarterly, 4, seasonal=True, m=4)
    results["Prophet"]["quarterly"] = _evaluate_prophet(quarterly, 4, yearly_seasonality=True)
    results["XGBoost"]["quarterly"] = _evaluate_xgb(quarterly, 4, n_lags=4, add_time_features=True)
    results["LSTM"]["quarterly"] = _evaluate_lstm(quarterly, 4, window=4)

    # Yearly: last 3 years for testing
    results["ARIMA"]["yearly"] = _evaluate_arima(yearly, 3, seasonal=False, m=1)
    results["Prophet"]["yearly"] = _evaluate_prophet(yearly, 3, yearly_seasonality=False)
    results["XGBoost"]["yearly"] = _evaluate_xgb(yearly, 3, n_lags=3, add_time_features=False)
    results["LSTM"]["yearly"] = _evaluate_lstm(yearly, 3, window=3)

    return results


__all__ = ["evaluate_all_models"]

