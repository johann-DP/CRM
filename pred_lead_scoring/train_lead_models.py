"""Training utilities for lead scoring models."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional
import pickle
import logging
from math import sqrt
import joblib

import numpy as np
import pandas as pd
from sklearn.metrics import (
    log_loss,
    roc_auc_score,
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
)
from sklearn.model_selection import TimeSeriesSplit

import tensorflow as tf
from keras import layers, Model
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from .logging_utils import setup_logging

logger = logging.getLogger(__name__)

try:  # Optional dependency
    from pmdarima import auto_arima as _auto_arima
except Exception as _exc_pmdarima:  # pragma: no cover - optional
    _auto_arima = None


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------


def train_lstm_lead(
    cfg: Dict[str, Dict],
    X_train: Optional[pd.DataFrame] = None,
    y_train: Optional[pd.Series] = None,
    X_val: Optional[pd.DataFrame] = None,
    y_val: Optional[pd.Series] = None,
) -> Model:
    """Train a simple MLP/LSTM model on the preprocessed lead scoring dataset.

    Parameters
    ----------
    cfg: dict
        Configuration dictionary with a ``lead_scoring`` section holding
        ``lstm_params`` and ``output_dir``. ``intra_threads`` and
        ``inter_threads`` control TensorFlow threading and GPU usage is
        detected automatically.
    """

    lead_cfg = cfg.get("lead_scoring", {})
    out_dir = Path(lead_cfg.get("output_dir", cfg.get("output_dir", ".")))
    data_dir = out_dir / "data_cache"
    lstm_params = lead_cfg.get("lstm_params", {})
    cross_val = lead_cfg.get("cross_val", False)

    # ------------------------------------------------------------------
    # Load datasets if not provided
    # ------------------------------------------------------------------
    if X_train is None or y_train is None:
        X_train = pd.read_csv(data_dir / "X_train.csv")
        y_train = pd.read_csv(data_dir / "y_train.csv").squeeze()
    if X_val is None or y_val is None:
        X_val = pd.read_csv(data_dir / "X_val.csv")
        y_val = pd.read_csv(data_dir / "y_val.csv").squeeze()

    X_train = X_train.loc[y_train.index]
    X_val = X_val.loc[y_val.index]

    # Convert to ``np.ndarray`` and ensure correct dtypes
    X_train = np.asarray(X_train, dtype=float)
    X_val = np.asarray(X_val, dtype=float)
    y_train = np.asarray(y_train, dtype=int).ravel()
    y_val = np.asarray(y_val, dtype=int).ravel()

    # Configure TensorFlow threading
    tf.config.threading.set_intra_op_parallelism_threads(
        lstm_params.get("intra_threads", 0)
    )
    tf.config.threading.set_inter_op_parallelism_threads(
        lstm_params.get("inter_threads", 0)
    )

    device = "/GPU:0" if tf.config.list_physical_devices("GPU") else "/CPU:0"

    def _build_model() -> Model:
        with tf.device(device):
            inp = layers.Input(shape=(X_train.shape[1],), name="features")
            x = layers.Dense(256, activation="relu")(inp)
            x = layers.Dropout(0.3)(x)
            x = layers.Dense(128, activation="relu")(x)
            x = layers.Dropout(0.3)(x)
            out = layers.Dense(1, activation="sigmoid")(x)
            model = Model(inputs=inp, outputs=out)
            model.compile(
                optimizer=tf.keras.optimizers.Adam(
                    learning_rate=lstm_params["learning_rate"]
                ),
                loss="binary_crossentropy",
                metrics=["AUC", "accuracy"],
            )
        return model

    if cross_val:
        X_full = np.concatenate([X_train, X_val])
        y_full = np.concatenate([y_train, y_val])
        tscv = TimeSeriesSplit(n_splits=5)
        metrics = []
        for i, (tr, va) in enumerate(tscv.split(X_full)):
            model_cv = _build_model()
            model_cv.fit(
                X_full[tr],
                y_full[tr],
                validation_data=(X_full[va], y_full[va]),
                batch_size=lstm_params["batch_size"],
                epochs=lstm_params["epochs"],
                callbacks=[
                    tf.keras.callbacks.EarlyStopping(
                        monitor="val_loss",
                        patience=lstm_params["patience"],
                        restore_best_weights=True,
                    )
                ],
                verbose=lstm_params.get("verbose", 1),
            )
            preds = model_cv.predict(X_full[va]).ravel()
            fold_logloss = log_loss(y_full[va], preds)
            fold_auc = roc_auc_score(y_full[va], preds)
            metrics.append({"logloss": fold_logloss, "auc": fold_auc})
            logger.info(
                "Fold %d/%d - log loss: %.4f, AUC: %.4f",
                i + 1,
                tscv.n_splits,
                fold_logloss,
                fold_auc,
            )

        mean_logloss = float(np.mean([m["logloss"] for m in metrics]))
        std_logloss = float(np.std([m["logloss"] for m in metrics]))
        mean_auc = float(np.mean([m["auc"] for m in metrics]))
        std_auc = float(np.std([m["auc"] for m in metrics]))
        logger.info(
            "CV log loss: %.4f ± %.4f, AUC: %.4f ± %.4f",
            mean_logloss,
            std_logloss,
            mean_auc,
            std_auc,
        )

        model_lstm = _build_model()
        model_lstm.fit(
            X_full,
            y_full,
            validation_split=0.1,
            batch_size=lstm_params["batch_size"],
            epochs=lstm_params["epochs"],
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=lstm_params["patience"],
                    restore_best_weights=True,
                )
            ],
            verbose=lstm_params.get("verbose", 1),
        )
        metrics_summary = {"logloss": mean_logloss, "auc": mean_auc}
    else:
        model_lstm = _build_model()
        model_lstm.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            batch_size=lstm_params["batch_size"],
            epochs=lstm_params["epochs"],
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=lstm_params["patience"],
                    restore_best_weights=True,
                )
            ],
            verbose=lstm_params.get("verbose", 1),
        )
        val_preds = model_lstm.predict(X_val).ravel()
        metrics_summary = {
            "logloss": log_loss(y_val, val_preds),
            "auc": roc_auc_score(y_val, val_preds),
        }
        pd.Series(val_preds).to_csv(data_dir / "proba_lstm.csv", index=False)
        logger.info(
            "Validation log loss: %.4f, AUC: %.4f",
            metrics_summary["logloss"],
            metrics_summary["auc"],
        )

    model_path = out_dir / "models" / "lead_lstm.h5"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_lstm.save(model_path)

    return model_lstm, metrics_summary


def train_xgboost_lead(
    cfg: Dict[str, Dict],
    X_train: Optional[pd.DataFrame] = None,
    y_train: Optional[pd.Series] = None,
    X_val: Optional[pd.DataFrame] = None,
    y_val: Optional[pd.Series] = None,
) -> tuple[XGBClassifier, Dict[str, float]]:
    """Train an XGBoost classifier on the lead scoring dataset.

    Parameters
    ----------
    cfg : dict
        Configuration with an ``xgb_params`` section. ``n_jobs`` controls
        internal parallelism of the XGBoost training.
    """

    lead_cfg = cfg.get("lead_scoring", {})
    out_dir = Path(lead_cfg.get("output_dir", cfg.get("output_dir", ".")))
    data_dir = out_dir / "data_cache"

    if X_train is None or y_train is None:
        X_train = pd.read_csv(data_dir / "X_train.csv")
        y_train = pd.read_csv(data_dir / "y_train.csv").squeeze()
    if X_val is None or y_val is None:
        X_val = pd.read_csv(data_dir / "X_val.csv")
        y_val = pd.read_csv(data_dir / "y_val.csv").squeeze()

    params = lead_cfg.get("xgb_params", {})
    params.setdefault("n_jobs", -1)
    cross_val = lead_cfg.get("cross_val", False)

    X_train = X_train.loc[y_train.index]
    X_val = X_val.loc[y_val.index]

    def _fit_model(X_t, y_t, X_v=None, y_v=None):
        model = XGBClassifier(use_label_encoder=False, eval_metric="logloss", **params)
        if X_v is not None and y_v is not None:
            model.fit(X_t, y_t, eval_set=[(X_v, y_v)], verbose=params.get("verbose", False))
        else:
            model.fit(X_t, y_t, verbose=params.get("verbose", False))
        return model

    if cross_val:
        X_full = pd.concat([X_train, X_val])
        y_full = pd.concat([y_train, y_val])
        tscv = TimeSeriesSplit(n_splits=5)
        metrics = []
        for i, (tr, va) in enumerate(tscv.split(X_full)):
            model_cv = _fit_model(X_full.iloc[tr], y_full.iloc[tr], X_full.iloc[va], y_full.iloc[va])
            preds = model_cv.predict_proba(X_full.iloc[va])[:, 1]
            fold_logloss = log_loss(y_full.iloc[va], preds)
            fold_auc = roc_auc_score(y_full.iloc[va], preds)
            metrics.append({"logloss": fold_logloss, "auc": fold_auc})
            logger.info(
                "XGB fold %d/%d - log loss: %.4f, AUC: %.4f",
                i + 1,
                tscv.n_splits,
                fold_logloss,
                fold_auc,
            )
        mean_logloss = float(np.mean([m["logloss"] for m in metrics]))
        std_logloss = float(np.std([m["logloss"] for m in metrics]))
        mean_auc = float(np.mean([m["auc"] for m in metrics]))
        std_auc = float(np.std([m["auc"] for m in metrics]))
        logger.info(
            "XGB CV log loss: %.4f ± %.4f, AUC: %.4f ± %.4f",
            mean_logloss,
            std_logloss,
            mean_auc,
            std_auc,
        )
        model_xgb = _fit_model(X_full, y_full)
        metrics_summary = {"logloss": mean_logloss, "auc": mean_auc}
    else:
        model_xgb = _fit_model(X_train, y_train, X_val, y_val)
        val_pred = model_xgb.predict_proba(X_val)[:, 1]
        metrics_summary = {
            "logloss": log_loss(y_val, val_pred),
            "auc": roc_auc_score(y_val, val_pred),
        }
        pd.Series(val_pred).to_csv(data_dir / "proba_xgboost.csv", index=False)

    model_path = out_dir / "models" / "lead_xgb.pkl"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model_xgb, model_path)

    return model_xgb, metrics_summary


def train_catboost_lead(
    cfg: Dict[str, Dict],
    X_train: Optional[pd.DataFrame] = None,
    y_train: Optional[pd.Series] = None,
    X_val: Optional[pd.DataFrame] = None,
    y_val: Optional[pd.Series] = None,
) -> tuple[CatBoostClassifier, Dict[str, float]]:
    """Train a CatBoost classifier on the lead scoring dataset.

    ``thread_count`` in ``catboost_params`` sets the number of threads used
    during training.
    """

    lead_cfg = cfg.get("lead_scoring", {})
    out_dir = Path(lead_cfg.get("output_dir", cfg.get("output_dir", ".")))
    data_dir = out_dir / "data_cache"

    if X_train is None or y_train is None:
        X_train = pd.read_csv(data_dir / "X_train.csv")
        y_train = pd.read_csv(data_dir / "y_train.csv").squeeze()
    if X_val is None or y_val is None:
        X_val = pd.read_csv(data_dir / "X_val.csv")
        y_val = pd.read_csv(data_dir / "y_val.csv").squeeze()

    X_train = X_train.loc[y_train.index]
    X_val = X_val.loc[y_val.index]

    params = lead_cfg.get("catboost_params", {})
    params.setdefault("thread_count", -1)
    cross_val = lead_cfg.get("cross_val", False)

    def _fit_model(X_t, y_t, X_v=None, y_v=None):
        model = CatBoostClassifier(**params)
        if X_v is not None and y_v is not None:
            model.fit(X_t, y_t, eval_set=(X_v, y_v), verbose=params.get("verbose", False))
        else:
            model.fit(X_t, y_t, verbose=params.get("verbose", False))
        return model

    if cross_val:
        X_full = pd.concat([X_train, X_val])
        y_full = pd.concat([y_train, y_val])
        tscv = TimeSeriesSplit(n_splits=5)
        metrics = []
        for i, (tr, va) in enumerate(tscv.split(X_full)):
            model_cv = _fit_model(X_full.iloc[tr], y_full.iloc[tr], X_full.iloc[va], y_full.iloc[va])
            preds = model_cv.predict_proba(X_full.iloc[va])[:, 1]
            fold_logloss = log_loss(y_full.iloc[va], preds)
            fold_auc = roc_auc_score(y_full.iloc[va], preds)
            metrics.append({"logloss": fold_logloss, "auc": fold_auc})
            logger.info(
                "CatBoost fold %d/%d - log loss: %.4f, AUC: %.4f",
                i + 1,
                tscv.n_splits,
                fold_logloss,
                fold_auc,
            )
        mean_logloss = float(np.mean([m["logloss"] for m in metrics]))
        std_logloss = float(np.std([m["logloss"] for m in metrics]))
        mean_auc = float(np.mean([m["auc"] for m in metrics]))
        std_auc = float(np.std([m["auc"] for m in metrics]))
        logger.info(
            "CatBoost CV log loss: %.4f ± %.4f, AUC: %.4f ± %.4f",
            mean_logloss,
            std_logloss,
            mean_auc,
            std_auc,
        )
        model_cat = _fit_model(X_full, y_full)
        metrics_summary = {"logloss": mean_logloss, "auc": mean_auc}
    else:
        model_cat = _fit_model(X_train, y_train, X_val, y_val)
        val_pred = model_cat.predict_proba(X_val)[:, 1]
        metrics_summary = {
            "logloss": log_loss(y_val, val_pred),
            "auc": roc_auc_score(y_val, val_pred),
        }
        pd.Series(val_pred).to_csv(data_dir / "proba_catboost.csv", index=False)

    model_path = out_dir / "models" / "lead_catboost.cbm"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_cat.save_model(str(model_path))

    return model_cat, metrics_summary


def train_arima_conv_rate(
    cfg: Dict[str, Dict],
    ts_conv_rate_train: Optional[pd.Series] = None,
    ts_conv_rate_test: Optional[pd.Series] = None,
) -> tuple[ARIMA, Dict[str, float]]:
    """Train an ARIMA model on the conversion rate time series.

    Parameters
    ----------
    cfg : dict
        Configuration dictionary containing ``lead_scoring`` settings. The
        ``arima_params`` section may define ``n_jobs`` for parallel order search.
    """

    lead_cfg = cfg.get("lead_scoring", {})
    out_dir = Path(lead_cfg.get("output_dir", cfg.get("output_dir", ".")))
    data_dir = out_dir / "data_cache"

    # ------------------------------------------------------------------
    # Load conversion rate series generated by ``preprocess_lead_scoring``
    # ------------------------------------------------------------------
    if ts_conv_rate_train is None or ts_conv_rate_test is None:
        ts_conv_rate_train = pd.read_csv(
            data_dir / "ts_conv_rate_train.csv", index_col=0, parse_dates=True
        )["conv_rate"]
        ts_conv_rate_test = pd.read_csv(
            data_dir / "ts_conv_rate_test.csv", index_col=0, parse_dates=True
        )["conv_rate"]

    ts_conv_rate_train = ts_conv_rate_train.fillna(0.0)
    ts_conv_rate_test = ts_conv_rate_test.fillna(0.0)

    # Determine ARIMA order either from config or via automatic search
    order = lead_cfg.get("arima_order")
    if order is None:
        if _auto_arima is None:
            raise ImportError(
                "pmdarima is required for automatic ARIMA order search"
            ) from _exc_pmdarima
        auto_model = _auto_arima(
            ts_conv_rate_train,
            seasonal=False,
            error_action="ignore",
            suppress_warnings=True,
            n_jobs=lead_cfg.get("arima_params", {}).get("n_jobs", 1),
        )
        order = auto_model.order

    arima_model = ARIMA(ts_conv_rate_train, order=tuple(order))
    fitted_arima = arima_model.fit()

    h = len(ts_conv_rate_test)
    forecast = fitted_arima.get_forecast(steps=h)
    arima_pred = forecast.predicted_mean

    mae = mean_absolute_error(ts_conv_rate_test, arima_pred)
    rmse = sqrt(mean_squared_error(ts_conv_rate_test, arima_pred))
    mape = mean_absolute_percentage_error(ts_conv_rate_test, arima_pred)

    model_path = out_dir / "models" / "arima_conv_rate.pkl"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with open(model_path, "wb") as fh:
        pickle.dump(fitted_arima, fh)

    pd.Series(arima_pred, index=ts_conv_rate_test.index).to_csv(
        data_dir / "pred_arima.csv"
    )

    return fitted_arima, {"mae": mae, "rmse": rmse, "mape": mape}


def train_prophet_conv_rate(
    cfg: Dict[str, Dict],
    df_prophet_train: Optional[pd.DataFrame] = None,
    ts_conv_rate_test: Optional[pd.Series] = None,
) -> tuple[Prophet, Dict[str, float]]:
    """Train a Prophet model on the aggregated conversion rate time series.

    Parameters
    ----------
    cfg : dict
        Configuration dictionary containing ``lead_scoring`` settings.
        Training runs on CPU and is parallelised internally by Prophet when
        possible.
    """

    lead_cfg = cfg.get("lead_scoring", {})
    out_dir = Path(lead_cfg.get("output_dir", cfg.get("output_dir", ".")))
    data_dir = out_dir / "data_cache"

    # ------------------------------------------------------------------
    # Load Prophet training DataFrame and test series generated by
    # ``preprocess_lead_scoring``
    # ------------------------------------------------------------------
    if df_prophet_train is None or ts_conv_rate_test is None:
        df_prophet_train = pd.read_csv(
            data_dir / "df_prophet_train.csv", parse_dates=["ds"]
        )
        ts_conv_rate_test = pd.read_csv(
            data_dir / "ts_conv_rate_test.csv", index_col=0, parse_dates=True
        )["conv_rate"]

    prophet_params = lead_cfg.get("prophet_params", {})
    model_prophet = Prophet(**prophet_params)
    model_prophet.fit(df_prophet_train)

    future = model_prophet.make_future_dataframe(
        periods=lead_cfg.get("prophet_forecast_periods", len(ts_conv_rate_test)),
        freq="M",
    )
    forecast = model_prophet.predict(future)

    forecast_series = forecast.set_index("ds")["yhat"]
    # On réindexe sur l’index de test ; si certaines dates de test ne figurent pas dans 'forecast_series',
    # on prend la dernière valeur connue (méthode "ffill").
    prophet_pred = forecast_series.reindex(ts_conv_rate_test.index, method="ffill")

    mae = mean_absolute_error(ts_conv_rate_test, prophet_pred)
    rmse = sqrt(mean_squared_error(ts_conv_rate_test, prophet_pred))
    mape = mean_absolute_percentage_error(ts_conv_rate_test, prophet_pred)

    model_path = out_dir / "models" / "prophet_conv_rate.pkl"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with open(model_path, "wb") as fh:
        pickle.dump(model_prophet, fh)

    pd.Series(prophet_pred.values, index=prophet_pred.index).to_csv(
        data_dir / "pred_prophet.csv"
    )

    return model_prophet, {"mae": mae, "rmse": rmse, "mape": mape}


__all__ = [
    "train_lstm_lead",
    "train_xgboost_lead",
    "train_catboost_lead",
    "train_arima_conv_rate",
    "train_prophet_conv_rate",
]
