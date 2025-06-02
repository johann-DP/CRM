"""XGBoost regression models for aggregated revenue forecasting."""

from __future__ import annotations

from typing import Tuple

import pandas as pd
from xgboost import XGBRegressor


# ---------------------------------------------------------------------------
# Supervised transformation
# ---------------------------------------------------------------------------

def _to_supervised(series: pd.Series, n_lags: int, *, add_time_features: bool = False) -> Tuple[pd.DataFrame, pd.Series]:
    """Convert ``series`` to a supervised learning dataset.

    Parameters
    ----------
    series:
        Time series indexed by ``DatetimeIndex``.
    n_lags:
        Number of past observations to use as predictors.
    add_time_features:
        If ``True`` add the month/quarter as an additional feature to help
        capture seasonality.
    """
    df = pd.DataFrame({"y": series})
    for i in range(1, n_lags + 1):
        df[f"lag{i}"] = series.shift(i)

    if add_time_features:
        # Add month or quarter depending on the frequency
        if series.index.freqstr and series.index.freqstr.startswith("M"):
            df["month"] = series.index.month
            time_cols = ["month"]
        elif series.index.freqstr and series.index.freqstr.startswith("Q"):
            df["quarter"] = series.index.quarter
            time_cols = ["quarter"]
        else:  # yearly or unknown frequency
            df["year"] = series.index.year
            time_cols = ["year"]
    else:
        time_cols = []

    df = df.dropna()
    feature_cols = [f"lag{i}" for i in range(1, n_lags + 1)] + time_cols
    X = df[feature_cols]
    y = df["y"]
    return X, y


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def train_xgb_model(series: pd.Series, n_lags: int, *, add_time_features: bool = False, **model_params: int) -> Tuple[XGBRegressor, float]:
    """Train an :class:`xgboost.XGBRegressor` on ``series``.

    Returns the fitted model and the training score (R^2).
    """
    X, y = _to_supervised(series, n_lags, add_time_features=add_time_features)
    model = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        random_state=42,
        **model_params,
    )
    model.fit(X, y)
    return model, model.score(X, y)


def train_all_granularities(
    monthly: pd.Series,
    quarterly: pd.Series,
    yearly: pd.Series,
) -> Tuple[Tuple[XGBRegressor, float], Tuple[XGBRegressor, float], Tuple[XGBRegressor, float]]:
    """Return XGBoost models fitted on monthly, quarterly and yearly series."""
    m_model, m_score = train_xgb_model(monthly, 12, add_time_features=True)
    q_model, q_score = train_xgb_model(quarterly, 4, add_time_features=True)
    y_model, y_score = train_xgb_model(yearly, 3)
    return (m_model, m_score), (q_model, q_score), (y_model, y_score)


__all__ = [
    "train_xgb_model",
    "train_all_granularities",
]
