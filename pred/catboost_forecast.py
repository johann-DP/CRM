"""CatBoost forecasting utilities for aggregated revenue."""

from __future__ import annotations

from typing import List, Tuple
import os

import numpy as np
import pandas as pd
try:  # pragma: no cover - optional dependency
    from catboost import CatBoostRegressor
except Exception:  # pragma: no cover - handle missing lib
    CatBoostRegressor = None
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
)


# ---------------------------------------------------------------------------
# Supervised dataset creation
# ---------------------------------------------------------------------------

def prepare_supervised(series: pd.Series, freq: str) -> pd.DataFrame:
    """Return supervised dataframe with lags and time feature.

    The function creates ``lag1`` .. ``lagK`` columns from ``series`` and adds a
    categorical time variable (month, quarter or year) depending on ``freq``.
    This feature is passed to CatBoost via ``cat_features`` so that the model
    handles the seasonality without one-hot encoding.
    """

    df = series.to_frame(name="y")

    if freq == "M":
        k = 12
        df["month"] = df.index.month
    elif freq == "Q":
        k = 4
        df["quarter"] = df.index.quarter
    elif freq == "A":
        k = 3
        df["year"] = df.index.year
    else:  # pragma: no cover - invalid frequency
        raise ValueError("freq must be 'M', 'Q' or 'A'")

    for lag in range(1, k + 1):
        df[f"lag{lag}"] = df["y"].shift(lag)

    # Convert categorical columns to string before dropping NaNs
    for col in ("month", "quarter", "year"):
        if col in df.columns:
            df[col] = df[col].astype(str)

    # Drop initial rows with incomplete lag information
    df = df.dropna().copy()

    return df


# ---------------------------------------------------------------------------
# Rolling forecast evaluation
# ---------------------------------------------------------------------------

def rolling_forecast_catboost(
    df_sup: pd.DataFrame,
    freq: str,
    test_size: int | None = None,
) -> Tuple[List[float], List[float]]:
    """Evaluate CatBoost model with a rolling forecast.

    At each step of the test horizon the model is trained on all available
    observations, then used to predict the next period.  The true value is
    appended to the training set for the following iteration, mimicking a real
    deployment scenario.
    """

    if freq == "M":
        default_test = 12
        cat_feat = ["month"]
    elif freq == "Q":
        default_test = 4
        cat_feat = ["quarter"]
    else:
        default_test = 2
        cat_feat = ["year"]

    n_test = test_size or default_test

    df_train = df_sup.iloc[:-n_test].copy()
    df_test = df_sup.iloc[-n_test:].copy()

    preds: List[float] = []
    actuals: List[float] = []

    if CatBoostRegressor is None:
        raise ImportError("catboost is required for CatBoost forecasting")

    for i in range(n_test):
        X_train = df_train.drop(columns=["y"]).copy()
        y_train = df_train["y"]
        for col in cat_feat:
            X_train[col] = X_train[col].astype(str)

        model = CatBoostRegressor(
            iterations=500,
            learning_rate=0.1,
            depth=6,
            random_seed=42,
            verbose=False,
            thread_count=os.cpu_count(),
        )
        model.fit(X_train, y_train, cat_features=cat_feat)

        row_test = df_test.iloc[i]
        X_next = row_test.drop(labels=["y"]).to_frame().T
        for col in cat_feat:
            X_next[col] = X_next[col].astype(str)
        y_pred = float(model.predict(X_next)[0])
        y_true = float(row_test["y"])

        preds.append(y_pred)
        actuals.append(y_true)

        # Update training data with the true observation for the next iteration
        df_train = pd.concat([df_train, df_test.iloc[i : i + 1]])

    mae = mean_absolute_error(actuals, preds)
    rmse = np.sqrt(mean_squared_error(actuals, preds))
    mape = mean_absolute_percentage_error(actuals, preds)
    print(
        f"CatBoost {freq} - MAE: {mae:.2f}, RMSE: {rmse:.2f}, MAPE: {mape*100:.2f}%"
    )

    return preds, actuals


# ---------------------------------------------------------------------------
# Future forecast generation
# ---------------------------------------------------------------------------

def forecast_future_catboost(
    series_clean: pd.Series, freq: str, horizon: int | None = None
) -> pd.DataFrame:
    """Iteratively forecast ``horizon`` future periods with CatBoost."""

    if CatBoostRegressor is None:
        raise ImportError("catboost is required for CatBoost forecasting")

    df_sup = prepare_supervised(series_clean, freq)

    if horizon is None:
        horizon = 12 if freq == "M" else (4 if freq == "Q" else 2)

    X_full = df_sup.drop(columns=["y"])
    y_full = df_sup["y"]

    if freq == "M":
        cat_feat = ["month"]
        k = 12
    elif freq == "Q":
        cat_feat = ["quarter"]
        k = 4
    else:
        cat_feat = ["year"]
        k = 3

    model_full = CatBoostRegressor(
        iterations=500,
        learning_rate=0.1,
        depth=6,
        random_seed=42,
        verbose=False,
        thread_count=os.cpu_count(),
    )
    model_full.fit(X_full, y_full, cat_features=cat_feat)

    # Preserve the training column order so prediction data is consistent
    feature_order = list(X_full.columns)

    last_date = series_clean.index[-1]
    future_dates = pd.date_range(
        start=last_date + pd.tseries.frequencies.to_offset(freq),
        periods=horizon,
        freq=freq,
    )

    history = series_clean.copy()
    forecasts: List[tuple[pd.Timestamp, float]] = []

    for dt in future_dates:
        # Retrieve the latest K observations, forward filling if necessary
        if len(history) >= k:
            last_vals = history.iloc[-k:]
        else:  # pragma: no cover - short series safeguard
            reindexed = history.reindex(
                pd.date_range(history.index[0], history.index[-1], freq=freq)
            )
            last_vals = reindexed.ffill().iloc[-k:]

        # Build feature dictionary in the same column order as training
        if freq == "M":
            features = {"month": str(dt.month)}
        elif freq == "Q":
            features = {"quarter": str(dt.quarter)}
        else:
            features = {"year": str(dt.year)}
        for i in range(1, k + 1):
            features[f"lag{i}"] = last_vals.iloc[-i]

        X_future = pd.DataFrame(features, index=[dt])[feature_order]
        # Ensure categorical columns are strings as required by CatBoost
        for col in cat_feat:
            X_future[col] = X_future[col].astype(str)
        yhat = float(model_full.predict(X_future)[0])
        forecasts.append((dt, yhat))
        history.loc[dt] = yhat

    df_forecast = (
        pd.DataFrame(forecasts, columns=["ds", "yhat_catboost"]).set_index("ds")
    )
    return df_forecast


if __name__ == "__main__":  # pragma: no cover - example usage
    # Dummy example to illustrate how the functions could be chained.
    import pathlib

    data_path = pathlib.Path("data.csv")  # replace with real path
    if data_path.exists():
        s = pd.read_csv(data_path, index_col=0, parse_dates=True).squeeze()
        df_sup = prepare_supervised(s, "M")
        rolling_forecast_catboost(df_sup, "M")
        fut = forecast_future_catboost(s, "M", horizon=12)
        print(fut.head())


__all__ = [
    "prepare_supervised",
    "rolling_forecast_catboost",
    "forecast_future_catboost",
]

