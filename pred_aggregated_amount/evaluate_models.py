"""Rolling forecast evaluation for time series models."""

from __future__ import annotations

from typing import Callable, Dict, List, Tuple
import concurrent.futures

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
from .catboost_forecast import rolling_forecast_catboost
from .features_utils import make_lag_features
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

    freq_str = series.index.freqstr or pd.infer_freq(series.index) or "M"
    if freq_str.startswith("Q"):
        freq = "Q"
    elif freq_str.startswith("A"):
        freq = "A"
    else:
        freq = "M"

    for t, val in enumerate(test):
        df_sup = make_lag_features(history, n_lags, freq, add_time_features)
        X_train = df_sup.drop(columns=["y"])
        y_train = df_sup["y"]
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
        df_full = make_lag_features(extended, n_lags, freq, add_time_features)
        X_pred = df_full.drop(columns=["y"]).iloc[-1:]
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

    if freq == "M":
        n_lags = 12
    elif freq == "Q":
        n_lags = 4
    else:
        n_lags = 3
    df_sup = make_lag_features(series, n_lags, freq, True)
    # Convert categorical column to string for CatBoost
    for col in ("month", "quarter", "year"):
        if col in df_sup.columns:
            df_sup[col] = df_sup[col].astype(str)
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

def _eval_arima(m, q, y, *, cross_val: bool, n_splits: int) -> Dict[str, Dict[str, float]]:
    if cross_val:
        cv = lambda s, **kw: _ts_cross_val(s, _evaluate_arima, n_splits=n_splits, **kw)
        return {
            "monthly": cv(m, seasonal=True, m=12),
            "quarterly": cv(q, seasonal=True, m=4),
            "yearly": cv(y, seasonal=False, m=1),
        }
    return {
        "monthly": _evaluate_arima(m, 12, seasonal=True, m=12),
        "quarterly": _evaluate_arima(q, 4, seasonal=True, m=4),
        "yearly": _evaluate_arima(y, 3, seasonal=False, m=1),
    }


def _eval_prophet(m, q, y, *, cross_val: bool, n_splits: int) -> Dict[str, Dict[str, float]]:
    if cross_val:
        cv = lambda s, **kw: _ts_cross_val(s, _evaluate_prophet, n_splits=n_splits, **kw)
        return {
            "monthly": cv(m, yearly_seasonality=True),
            "quarterly": cv(q, yearly_seasonality=True),
            "yearly": cv(y, yearly_seasonality=False),
        }
    return {
        "monthly": _evaluate_prophet(m, 12, yearly_seasonality=True),
        "quarterly": _evaluate_prophet(q, 4, yearly_seasonality=True),
        "yearly": _evaluate_prophet(y, 3, yearly_seasonality=False),
    }


def _eval_xgb(m, q, y, *, cross_val: bool, n_splits: int) -> Dict[str, Dict[str, float]]:
    if cross_val:
        cv = lambda s, **kw: _ts_cross_val(s, _evaluate_xgb, n_splits=n_splits, **kw)
        return {
            "monthly": cv(m, n_lags=12, add_time_features=True),
            "quarterly": cv(q, n_lags=4, add_time_features=True),
            "yearly": cv(y, n_lags=3, add_time_features=False),
        }
    return {
        "monthly": _evaluate_xgb(m, 12, n_lags=12, add_time_features=True),
        "quarterly": _evaluate_xgb(q, 4, n_lags=4, add_time_features=True),
        "yearly": _evaluate_xgb(y, 3, n_lags=3, add_time_features=False),
    }


def _eval_lstm(m, q, y, *, cross_val: bool, n_splits: int) -> Dict[str, Dict[str, float]]:
    if cross_val:
        cv = lambda s, **kw: _ts_cross_val(s, _evaluate_lstm, n_splits=n_splits, **kw)
        return {
            "monthly": cv(m, window=12),
            "quarterly": cv(q, window=4),
            "yearly": cv(y, window=3),
        }
    return {
        "monthly": _evaluate_lstm(m, 12, window=12),
        "quarterly": _evaluate_lstm(q, 4, window=4),
        "yearly": _evaluate_lstm(y, 3, window=3),
    }


def _eval_catboost(m, q, y, *, cross_val: bool, n_splits: int) -> Dict[str, Dict[str, float]]:
    if cross_val:
        cv = lambda s, f: _ts_cross_val(
            s,
            lambda ser, ts, *, freq=f: _evaluate_catboost(ser, freq, test_size=ts),
            n_splits=n_splits,
        )
        return {
            "monthly": cv(m, "M"),
            "quarterly": cv(q, "Q"),
            "yearly": cv(y, "A"),
        }

    if m.nunique() == 1 and q.nunique() == 1 and y.nunique() == 1:
        zero = {"MAE": 0.0, "RMSE": 0.0, "MAPE": 0.0}
        return {"monthly": zero, "quarterly": zero, "yearly": zero}

    dfm = prepare_supervised(m, freq="M")
    dfq = prepare_supervised(q, freq="Q")
    dfy = prepare_supervised(y, freq="A")

    preds_m, actuals_m = rolling_forecast_catboost(dfm, freq="M")
    preds_q, actuals_q = rolling_forecast_catboost(dfq, freq="Q")
    preds_y, actuals_y = rolling_forecast_catboost(dfy, freq="A")

    return {
        "monthly": _compute_metrics(actuals_m, preds_m),
        "quarterly": _compute_metrics(actuals_q, preds_q),
        "yearly": _compute_metrics(actuals_y, preds_y),
    }


EVAL_FUNCS = {
    "ARIMA": _eval_arima,
    "Prophet": _eval_prophet,
    "XGBoost": _eval_xgb,
    "LSTM": _eval_lstm,
    "CatBoost": _eval_catboost,
}


def evaluate_all_models(
    monthly: pd.Series,
    quarterly: pd.Series,
    yearly: pd.Series,
    *,
    jobs: int = 1,
    cross_val: bool = True,
    n_splits: int = 5,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Return MAE, RMSE and MAPE for all models and granularities."""
    results: Dict[str, Dict[str, Dict[str, float]]] = {}
    if jobs > 1:
        with concurrent.futures.ProcessPoolExecutor(max_workers=jobs) as ex:
            futs = {
                ex.submit(func, monthly, quarterly, yearly, cross_val=cross_val, n_splits=n_splits): name
                for name, func in EVAL_FUNCS.items()
            }
            for fut in concurrent.futures.as_completed(futs):
                name = futs[fut]
                try:
                    results[name] = fut.result()
                except Exception as exc:  # pragma: no cover - passthrough
                    print(f"{name} failed: {exc}")
                    results[name] = {}
    else:
        for name, func in EVAL_FUNCS.items():
            try:
                results[name] = func(
                    monthly,
                    quarterly,
                    yearly,
                    cross_val=cross_val,
                    n_splits=n_splits,
                )
            except Exception as exc:  # pragma: no cover - passthrough
                print(f"{name} failed: {exc}")
                results[name] = {}
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
    freq_str = series.index.freqstr or pd.infer_freq(series.index) or "M"
    if freq_str.startswith("Q"):
        freq = "Q"
    elif freq_str.startswith("A"):
        freq = "A"
    else:
        freq = "M"

    for t, val in enumerate(test):
        df_sup = make_lag_features(history, n_lags, freq, add_time_features)
        X_train = df_sup.drop(columns=["y"])
        y_train = df_sup["y"]
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
        df_full = make_lag_features(extended, n_lags, freq, add_time_features)
        X_pred = df_full.drop(columns=["y"]).iloc[-1:]
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
    if freq == "M":
        n_lags = 12
    elif freq == "Q":
        n_lags = 4
    else:
        n_lags = 3
    df_sup = make_lag_features(series, n_lags, freq, True)
    for col in ("month", "quarter", "year"):
        if col in df_sup.columns:
            df_sup[col] = df_sup[col].astype(str)
    preds, actuals = rolling_forecast_catboost(df_sup, freq)
    n = len(preds)
    test_index = df_sup.index[-n:]
    return pd.Series(preds, index=test_index), pd.Series(actuals, index=test_index)




