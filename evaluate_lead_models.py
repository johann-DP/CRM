"""Evaluate trained lead scoring models and forecast models."""

from __future__ import annotations

from pathlib import Path
from typing import Dict
import pickle

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from prophet import Prophet
from sklearn.metrics import (
    log_loss,
    roc_auc_score,
    precision_recall_fscore_support,
    mean_absolute_error,
    mean_squared_error,
)
import tensorflow as tf


def _safe_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Return MAPE ignoring zero values in ``y_true``."""
    mask = y_true != 0
    if mask.sum() == 0:
        return np.nan
    return np.mean(np.abs((y_pred[mask] - y_true[mask]) / y_true[mask])) * 100


def evaluate_lead_models(cfg: Dict[str, Dict]) -> pd.DataFrame:
    """Return metrics for lead scoring classification and forecast models."""
    lead_cfg = cfg.get("lead_scoring", {})
    out_dir = Path(lead_cfg.get("output_dir", cfg.get("output_dir", ".")))
    models_dir = out_dir / "models"

    # ------------------------------------------------------------------
    # Load datasets
    # ------------------------------------------------------------------
    X_test = pd.read_csv(out_dir / "X_test.csv")
    y_test = pd.read_csv(out_dir / "y_test.csv").squeeze()
    ts_conv_rate_test = (
        pd.read_csv(out_dir / "ts_conv_rate_test.csv", index_col=0, parse_dates=True)[
            "conv_rate"
        ]
    )

    # ------------------------------------------------------------------
    # Load models
    # ------------------------------------------------------------------
    xgb_model = joblib.load(models_dir / "lead_xgb.pkl")

    cat_model = CatBoostClassifier()
    cat_model.load_model(str(models_dir / "lead_catboost.cbm"))

    lstm_model = tf.keras.models.load_model(models_dir / "lead_lstm.h5")

    with open(models_dir / "arima_conv_rate.pkl", "rb") as fh:
        arima_model = pickle.load(fh)

    prophet_model = Prophet()
    prophet_model.load(str(models_dir / "prophet_conv_rate.pkl"))

    metrics = []

    # ------------------------------------------------------------------
    # Classification models
    # ------------------------------------------------------------------
    def _add_clf(name: str, proba: np.ndarray) -> None:
        logloss = log_loss(y_test, proba)
        auc = roc_auc_score(y_test, proba)
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_test, proba > 0.5, average="binary"
        )
        metrics.append(
            {
                "model": name,
                "model_type": "classifier",
                "logloss": logloss,
                "auc": auc,
                "precision": prec,
                "recall": rec,
                "f1": f1,
            }
        )

    _add_clf("xgboost", xgb_model.predict_proba(X_test)[:, 1])
    _add_clf("catboost", cat_model.predict_proba(X_test)[:, 1])
    _add_clf("lstm", lstm_model.predict(X_test).ravel())

    # ------------------------------------------------------------------
    # Forecast models
    # ------------------------------------------------------------------
    h = len(ts_conv_rate_test)
    pred_arima = arima_model.forecast(steps=h)
    metrics.append(
        {
            "model": "arima",
            "model_type": "arima",
            "mae": mean_absolute_error(ts_conv_rate_test, pred_arima),
            "rmse": mean_squared_error(ts_conv_rate_test, pred_arima, squared=False),
            "mape": _safe_mape(ts_conv_rate_test.values, np.asarray(pred_arima)),
        }
    )

    future = prophet_model.make_future_dataframe(periods=h, freq="M")
    forecast = prophet_model.predict(future)
    pred_prophet = forecast.set_index("ds")["yhat"].iloc[-h:]
    metrics.append(
        {
            "model": "prophet",
            "model_type": "prophet",
            "mae": mean_absolute_error(ts_conv_rate_test, pred_prophet),
            "rmse": mean_squared_error(ts_conv_rate_test, pred_prophet, squared=False),
            "mape": _safe_mape(ts_conv_rate_test.values, pred_prophet.values),
        }
    )

    df_metrics = pd.DataFrame(metrics)
    out_file = out_dir / "lead_metrics_summary.csv"
    df_metrics.to_csv(out_file, index=False)
    return df_metrics


if __name__ == "__main__":  # pragma: no cover - simple CLI
    import argparse
    import yaml

    p = argparse.ArgumentParser(description="Evaluate lead scoring models")
    p.add_argument("--config", default="config.yaml", help="Path to YAML config")
    args = p.parse_args()

    with open(args.config, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    df = evaluate_lead_models(cfg)
    print(df.to_string(index=False))
