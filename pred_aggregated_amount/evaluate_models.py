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
# CatBoost rolling forecast
# ---------------------------------------------------------------------------

def _evaluate_catboost(
    series: pd.Series, freq: str, *, test_size: int | None = None
) -> Dict[str, float]:
    """Evaluate CatBoost model with rolling forecast."""
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

def evaluate_all_models(
    monthly: pd.Series,
    quarterly: pd.Series,
    yearly: pd.Series,
    *,
    cross_val: bool = False,
    n_splits: int = 5,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Return MAE, RMSE and MAPE for all models and granularities."""
    results: Dict[str, Dict[str, Dict[str, float]]] = {
        "ARIMA": {"monthly": {}, "quarterly": {}, "yearly": {}},
        "Prophet": {"monthly": {}, "quarterly": {}, "yearly": {}},
        "XGBoost": {"monthly": {}, "quarterly": {}, "yearly": {}},
        "LSTM": {"monthly": {}, "quarterly": {}, "yearly": {}},
        "CatBoost": {"monthly": {}, "quarterly": {}, "yearly": {}},
    }

    if cross_val:
        cv = lambda s, fn, **kw: _ts_cross_val(s, fn, n_splits=n_splits, **kw)
        results["ARIMA"]["monthly"] = cv(monthly, _evaluate_arima, seasonal=True, m=12)
        results["Prophet"]["monthly"] = cv(monthly, _evaluate_prophet, yearly_seasonality=True)
        results["XGBoost"]["monthly"] = cv(monthly, _evaluate_xgb, n_lags=12, add_time_features=True)
        results["LSTM"]["monthly"] = cv(monthly, _evaluate_lstm, window=12)
        results["CatBoost"]["monthly"] = cv(monthly, _evaluate_catboost, freq="M")

        results["ARIMA"]["quarterly"] = cv(quarterly, _evaluate_arima, seasonal=True, m=4)
        results["Prophet"]["quarterly"] = cv(quarterly, _evaluate_prophet, yearly_seasonality=True)
        results["XGBoost"]["quarterly"] = cv(quarterly, _evaluate_xgb, n_lags=4, add_time_features=True)
        results["LSTM"]["quarterly"] = cv(quarterly, _evaluate_lstm, window=4)
        results["CatBoost"]["quarterly"] = cv(quarterly, _evaluate_catboost, freq="Q")

        results["ARIMA"]["yearly"] = cv(yearly, _evaluate_arima, seasonal=False, m=1)
        results["Prophet"]["yearly"] = cv(yearly, _evaluate_prophet, yearly_seasonality=False)
        results["XGBoost"]["yearly"] = cv(yearly, _evaluate_xgb, n_lags=3, add_time_features=False)
        results["LSTM"]["yearly"] = cv(yearly, _evaluate_lstm, window=3)
        results["CatBoost"]["yearly"] = cv(yearly, _evaluate_catboost, freq="A")
    else:
        # Monthly: last 12 months for testing
        results["ARIMA"]["monthly"] = _evaluate_arima(monthly, 12, seasonal=True, m=12)
        results["Prophet"]["monthly"] = _evaluate_prophet(monthly, 12, yearly_seasonality=True)
        results["XGBoost"]["monthly"] = _evaluate_xgb(monthly, 12, n_lags=12, add_time_features=True)
        results["LSTM"]["monthly"] = _evaluate_lstm(monthly, 12, window=12)
        results["CatBoost"]["monthly"] = _evaluate_catboost(monthly, "M")

        # Quarterly: last 4 quarters for testing
        results["ARIMA"]["quarterly"] = _evaluate_arima(quarterly, 4, seasonal=True, m=4)
        results["Prophet"]["quarterly"] = _evaluate_prophet(quarterly, 4, yearly_seasonality=True)
        results["XGBoost"]["quarterly"] = _evaluate_xgb(quarterly, 4, n_lags=4, add_time_features=True)
        results["LSTM"]["quarterly"] = _evaluate_lstm(quarterly, 4, window=4)
        results["CatBoost"]["quarterly"] = _evaluate_catboost(quarterly, "Q")

        # Yearly: last 3 years for testing
        results["ARIMA"]["yearly"] = _evaluate_arima(yearly, 3, seasonal=False, m=1)
        results["Prophet"]["yearly"] = _evaluate_prophet(yearly, 3, yearly_seasonality=False)
        results["XGBoost"]["yearly"] = _evaluate_xgb(yearly, 3, n_lags=3, add_time_features=False)
        results["LSTM"]["yearly"] = _evaluate_lstm(yearly, 3, window=3)
        results["CatBoost"]["yearly"] = _evaluate_catboost(yearly, "A")

    return results





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


def _rolling_preds_lstm(series: pd.Series, test_size: int, *, window: int, epochs: int = 50) -> Tuple[pd.Series, pd.Series]:
    """Return predicted and true values of a rolling LSTM forecast."""
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
    return pd.Series(preds, index=test.index), test


def _rolling_preds_catboost(series: pd.Series, freq: str) -> Tuple[pd.Series, pd.Series]:
    """Return predicted and true values for CatBoost rolling forecast."""
    df_sup = prepare_supervised(series, freq)
    preds, actuals = rolling_forecast_catboost(df_sup, freq)
    n = len(preds)
    test_index = df_sup.index[-n:]
    return pd.Series(preds, index=test_index), pd.Series(actuals, index=test_index)


def evaluate_and_predict_all_models(
    monthly: pd.Series,
    quarterly: pd.Series,
    yearly: pd.Series,
) -> Tuple[Dict[str, Dict[str, Dict[str, float]]], Dict[str, Dict[str, pd.Series]]]:
    """Return metrics and rolling predictions for all models."""
    metrics: Dict[str, Dict[str, Dict[str, float]]] = {
        "ARIMA": {"monthly": {}, "quarterly": {}, "yearly": {}},
        "Prophet": {"monthly": {}, "quarterly": {}, "yearly": {}},
        "XGBoost": {"monthly": {}, "quarterly": {}, "yearly": {}},
        "LSTM": {"monthly": {}, "quarterly": {}, "yearly": {}},
        "CatBoost": {"monthly": {}, "quarterly": {}, "yearly": {}},
    }
    preds: Dict[str, Dict[str, pd.Series]] = {
        "ARIMA": {},
        "Prophet": {},
        "XGBoost": {},
        "LSTM": {},
        "CatBoost": {},
    }

    p, t = _rolling_preds_arima(monthly, 12, seasonal=True, m=12)
    metrics["ARIMA"]["monthly"] = _compute_metrics(t.values, list(p))
    preds["ARIMA"]["monthly"] = p

    p, t = _rolling_preds_prophet(monthly, 12, yearly_seasonality=True)
    metrics["Prophet"]["monthly"] = _compute_metrics(t.values, list(p))
    preds["Prophet"]["monthly"] = p

    p, t = _rolling_preds_xgb(monthly, 12, n_lags=12, add_time_features=True)
    metrics["XGBoost"]["monthly"] = _compute_metrics(t.values, list(p))
    preds["XGBoost"]["monthly"] = p

    p, t = _rolling_preds_lstm(monthly, 12, window=12)
    metrics["LSTM"]["monthly"] = _compute_metrics(t.values, list(p))
    preds["LSTM"]["monthly"] = p

    p, t = _rolling_preds_catboost(monthly, "M")
    metrics["CatBoost"]["monthly"] = _compute_metrics(t.values, list(p))
    preds["CatBoost"]["monthly"] = p

    p, t = _rolling_preds_arima(quarterly, 4, seasonal=True, m=4)
    metrics["ARIMA"]["quarterly"] = _compute_metrics(t.values, list(p))
    preds["ARIMA"]["quarterly"] = p

    p, t = _rolling_preds_prophet(quarterly, 4, yearly_seasonality=True)
    metrics["Prophet"]["quarterly"] = _compute_metrics(t.values, list(p))
    preds["Prophet"]["quarterly"] = p

    p, t = _rolling_preds_xgb(quarterly, 4, n_lags=4, add_time_features=True)
    metrics["XGBoost"]["quarterly"] = _compute_metrics(t.values, list(p))
    preds["XGBoost"]["quarterly"] = p

    p, t = _rolling_preds_lstm(quarterly, 4, window=4)
    metrics["LSTM"]["quarterly"] = _compute_metrics(t.values, list(p))
    preds["LSTM"]["quarterly"] = p

    p, t = _rolling_preds_catboost(quarterly, "Q")
    metrics["CatBoost"]["quarterly"] = _compute_metrics(t.values, list(p))
    preds["CatBoost"]["quarterly"] = p

    p, t = _rolling_preds_arima(yearly, 3, seasonal=False, m=1)
    metrics["ARIMA"]["yearly"] = _compute_metrics(t.values, list(p))
    preds["ARIMA"]["yearly"] = p

    p, t = _rolling_preds_prophet(yearly, 3, yearly_seasonality=False)
    metrics["Prophet"]["yearly"] = _compute_metrics(t.values, list(p))
    preds["Prophet"]["yearly"] = p

    p, t = _rolling_preds_xgb(yearly, 3, n_lags=3, add_time_features=False)
    metrics["XGBoost"]["yearly"] = _compute_metrics(t.values, list(p))
    preds["XGBoost"]["yearly"] = p

    p, t = _rolling_preds_lstm(yearly, 3, window=3)
    metrics["LSTM"]["yearly"] = _compute_metrics(t.values, list(p))
    preds["LSTM"]["yearly"] = p

    p, t = _rolling_preds_catboost(yearly, "A")
    metrics["CatBoost"]["yearly"] = _compute_metrics(t.values, list(p))
    preds["CatBoost"]["yearly"] = p

    return metrics, preds


__all__ = [
    "evaluate_all_models",
    "evaluate_and_predict_all_models",
]

