"""Evaluate trained lead scoring models and forecast models."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional
import pickle
import logging
import concurrent.futures
import multiprocessing as mp

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
    confusion_matrix,
    brier_score_loss,
)
import tensorflow as tf
from .logging_utils import setup_logging

logger = logging.getLogger(__name__)


def _safe_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Return MAPE ignoring zero values in ``y_true``."""
    mask = y_true != 0
    if mask.sum() == 0:
        return np.nan
    return np.mean(np.abs((y_pred[mask] - y_true[mask]) / y_true[mask])) * 100


def _clf_metrics(name: str, proba: np.ndarray, y_true: np.ndarray, out_dir: Path) -> Dict:
    """Compute classification metrics for a single model."""
    pd.Series(proba).to_csv(out_dir / f"proba_{name}.csv", index=False)
    logloss = log_loss(y_true, proba)
    auc = roc_auc_score(y_true, proba)
    pred = proba > 0.5
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, pred, average="binary")
    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
    lost_rec = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    bal_acc = (rec + lost_rec) / 2
    brier = brier_score_loss(y_true, proba)
    return {
        "model": name,
        "model_type": "classifier",
        "logloss": logloss,
        "auc": auc,
        "precision": prec,
        "recall": rec,
        "lost_recall": lost_rec,
        "balanced_accuracy": bal_acc,
        "f1": f1,
        "brier": brier,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def evaluate_lead_models(
    cfg: Dict[str, Dict],
    X_test: Optional[pd.DataFrame] = None,
    y_test: Optional[pd.Series] = None,
    ts_conv_rate_test: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """Return metrics for lead scoring classification and forecast models."""
    lead_cfg = cfg.get("lead_scoring", {})
    out_dir = Path(lead_cfg.get("output_dir", cfg.get("output_dir", ".")))
    data_dir = out_dir / "data_cache"
    models_dir = out_dir / "models"

    # ------------------------------------------------------------------
    # Load datasets
    # ------------------------------------------------------------------
    if X_test is None or y_test is None:
        X_test = pd.read_csv(data_dir / "X_test.csv")
        y_test = pd.read_csv(data_dir / "y_test.csv").squeeze()
    if ts_conv_rate_test is None:
        ts_conv_rate_test = pd.read_csv(
            data_dir / "ts_conv_rate_test.csv", index_col=0, parse_dates=True
        )["conv_rate"]

    # ------------------------------------------------------------------
    # Load models
    # ------------------------------------------------------------------
    xgb_model = joblib.load(models_dir / "lead_xgb.pkl")

    cat_model = CatBoostClassifier()
    cat_model.load_model(str(models_dir / "lead_catboost.cbm"))

    log_model = joblib.load(models_dir / "lead_logistic.pkl")

    mlp_model = tf.keras.models.load_model(models_dir / "lead_mlp.h5")

    with open(models_dir / "arima_conv_rate.pkl", "rb") as fh:
        arima_model = pickle.load(fh)

    prophet_model = Prophet()
    with open(models_dir / "prophet_conv_rate.pkl", "rb") as fh:
        prophet_model = pickle.load(fh)

    metrics = []

    # ------------------------------------------------------------------
    # Classification models
    # ------------------------------------------------------------------
    # Avant de passer X_test à XGBoost, s'assurer que les colonnes object sont en 'category'
    for col in X_test.select_dtypes(include="object").columns:
        X_test[col] = X_test[col].astype("category")

    preds = {
        "xgboost": xgb_model.predict_proba(X_test)[:, 1],
        "catboost": cat_model.predict_proba(X_test)[:, 1],
        "logistic": log_model.predict_proba(X_test)[:, 1],
        "mlp": mlp_model.predict(X_test).ravel(),
    }
    preds["ensemble"] = (preds["xgboost"] + preds["catboost"]) / 2

    with concurrent.futures.ProcessPoolExecutor(max_workers=mp.cpu_count()) as ex:
        futures = [
            ex.submit(_clf_metrics, name, proba, y_test.values, data_dir)
            for name, proba in preds.items()
        ]
        metrics.extend(f.result() for f in futures)

    # ------------------------------------------------------------------
    # Forecast models
    # ------------------------------------------------------------------
    h = len(ts_conv_rate_test)
    pred_arima = arima_model.forecast(steps=h)
    pd.Series(pred_arima, index=ts_conv_rate_test.index).to_csv(
        data_dir / "pred_arima.csv"
    )
    metrics.append(
        {
            "model": "arima",
            "model_type": "arima",
            "mae": mean_absolute_error(ts_conv_rate_test, pred_arima),
            "rmse": mean_squared_error(ts_conv_rate_test, pred_arima),
            "mape": _safe_mape(ts_conv_rate_test.values, np.asarray(pred_arima)),
        }
    )

    future = prophet_model.make_future_dataframe(periods=h, freq="M")
    forecast = prophet_model.predict(future)
    pred_prophet = forecast.set_index("ds")["yhat"].iloc[-h:]
    pred_prophet.to_csv(data_dir / "pred_prophet.csv")
    metrics.append(
        {
            "model": "prophet",
            "model_type": "prophet",
            "mae": mean_absolute_error(ts_conv_rate_test, pred_prophet),
            "rmse": mean_squared_error(ts_conv_rate_test, pred_prophet),
            "mape": _safe_mape(ts_conv_rate_test.values, pred_prophet.values),
        }
    )

    df_metrics = pd.DataFrame(metrics)
    out_file = data_dir / "lead_metrics_summary.csv"
    df_metrics.to_csv(out_file, index=False)
    return df_metrics


if __name__ == "__main__":  # pragma: no cover - simple CLI
    import argparse
    import yaml

    setup_logging()

    p = argparse.ArgumentParser(description="Evaluate lead scoring models")
    p.add_argument("--config", default="config.yaml", help="Path to YAML config")
    args = p.parse_args()

    with open(args.config, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    df = evaluate_lead_models(cfg)
    logger.info("\n%s", df.to_string(index=False))
