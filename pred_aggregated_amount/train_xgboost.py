"""XGBoost regression models for aggregated revenue forecasting."""

from __future__ import annotations

from typing import Tuple
import os

import pandas as pd
from xgboost import XGBRegressor

from .features_utils import make_lag_features




# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def train_xgb_model(series: pd.Series, n_lags: int, *, add_time_features: bool = False, **model_params: int) -> Tuple[XGBRegressor, float]:
    """Train an :class:`xgboost.XGBRegressor` on ``series``.

    Returns the fitted model and the training score (R^2).
    """
    freq_str = series.index.freqstr or pd.infer_freq(series.index) or "M"
    if freq_str.startswith("Q"):
        freq = "Q"
    elif freq_str.startswith("A"):
        freq = "A"
    else:
        freq = "M"
    df_sup = make_lag_features(series, n_lags, freq, add_time_features)
    X = df_sup.drop(columns=["y"])
    y = df_sup["y"]
    model = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        random_state=42,
        n_jobs=os.cpu_count() or 1,
        **model_params,
    )
    model.fit(X, y)
    return model, model.score(X, y)


__all__ = ["train_xgb_model"]
