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
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from skopt import BayesSearchCV
from skopt.space import Integer, Real
import concurrent.futures

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


def _run_hyperparameter_search(
    model,
    param_grid: dict,
    bayes_space: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> dict:
    """Run GridSearchCV and BayesSearchCV in parallel and return best params."""
    cv = TimeSeriesSplit(n_splits=5)
    grid = GridSearchCV(
        model,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
    )
    bayes = BayesSearchCV(
        model,
        search_spaces=bayes_space,
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
        n_iter=20,
    )

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
        fut_grid = ex.submit(grid.fit, X_train, y_train)
        fut_bayes = ex.submit(bayes.fit, X_train, y_train)
        fut_grid.result()
        fut_bayes.result()

    if bayes.best_score_ >= grid.best_score_:
        return bayes.best_params_
    return grid.best_params_


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

    # ------------------------------------------------------------------
    # Model definition
    # ------------------------------------------------------------------
    with tf.device(device):
        inp = layers.Input(shape=(X_train.shape[1],), name="features")
        x = layers.Dense(256, activation="relu")(inp)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(128, activation="relu")(x)
        x = layers.Dropout(0.3)(x)
        out = layers.Dense(1, activation="sigmoid")(x)
        model_lstm = Model(inputs=inp, outputs=out)
        model_lstm.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=lstm_params["learning_rate"]
            ),
            loss="binary_crossentropy",
            metrics=["AUC", "accuracy"],
        )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Save trained model
    # ------------------------------------------------------------------
    model_path = out_dir / "models" / "lead_lstm.h5"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_lstm.save(model_path)

    # ------------------------------------------------------------------
    # Validation metrics
    # ------------------------------------------------------------------
    val_preds = model_lstm.predict(X_val).ravel()
    logloss_val = log_loss(y_val, val_preds)
    auc_val = roc_auc_score(y_val, val_preds)
    pd.Series(val_preds).to_csv(data_dir / "proba_lstm.csv", index=False)
    logger.info(
        "Validation log loss: %.4f, AUC: %.4f",
        logloss_val,
        auc_val,
    )

    return model_lstm


def train_logistic_lead(
    cfg: Dict[str, Dict],
    X_train: Optional[pd.DataFrame] = None,
    y_train: Optional[pd.Series] = None,
    X_val: Optional[pd.DataFrame] = None,
    y_val: Optional[pd.Series] = None,
) -> tuple[LogisticRegression, Dict[str, float]]:
    """Train a logistic regression classifier with optional hyperparameter search."""

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

    params = lead_cfg.get("logistic_params", {}).copy()

    if lead_cfg.get("fine_tuning", False):
        grid = {"C": [0.01, 0.1, 1, 10]}
        space = {"C": Real(1e-3, 10, prior="log-uniform")}
        base_model = LogisticRegression(max_iter=1000, **params)
        best_params = _run_hyperparameter_search(base_model, grid, space, X_train, y_train)
        params.update(best_params)

    model_log = LogisticRegression(max_iter=1000, **params)
    model_log.fit(X_train, y_train)

    model_path = out_dir / "models" / "lead_logistic.pkl"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model_log, model_path)

    val_pred = model_log.predict_proba(X_val)[:, 1]
    pd.Series(val_pred).to_csv(data_dir / "proba_logistic.csv", index=False)
    metrics = {
        "logloss": log_loss(y_val, val_pred),
        "auc": roc_auc_score(y_val, val_pred),
    }
    return model_log, metrics


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

    params = lead_cfg.get("xgb_params", {}).copy()
    params.setdefault("n_jobs", lead_cfg.get("xgb_params", {}).get("n_jobs", 1))

    if lead_cfg.get("fine_tuning", False):
        grid = {
            "max_depth": [3, 6, 10],
            "learning_rate": [0.01, 0.1],
            "subsample": [0.5, 1.0],
        }
        space = {
            "max_depth": Integer(3, 10),
            "learning_rate": Real(0.01, 0.3, prior="log-uniform"),
            "subsample": Real(0.5, 1.0),
        }
        base_model = XGBClassifier(use_label_encoder=False, eval_metric="logloss", n_jobs=params["n_jobs"])
        best_params = _run_hyperparameter_search(base_model, grid, space, X_train, y_train)
        params.update(best_params)

    model_xgb = XGBClassifier(use_label_encoder=False, eval_metric="logloss", **params)

    X_train = X_train.loc[y_train.index]
    X_val = X_val.loc[y_val.index]

    logger.debug("DEBUG – XGBoost")
    logger.debug("  X_train.shape: %s", X_train.shape)
    logger.debug("  len(y_train): %d", len(y_train))
    logger.debug("  X_val.shape: %s", X_val.shape)
    logger.debug("  len(y_val): %d", len(y_val))

    model_xgb.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=params.get("verbose", False),
    )

    model_path = out_dir / "models" / "lead_xgb.pkl"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model_xgb, model_path)

    val_pred = model_xgb.predict_proba(X_val)[:, 1]
    pd.Series(val_pred).to_csv(data_dir / "proba_xgboost.csv", index=False)
    metrics = {
        "logloss": log_loss(y_val, val_pred),
        "auc": roc_auc_score(y_val, val_pred),
    }
    return model_xgb, metrics


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

    params = lead_cfg.get("catboost_params", {}).copy()
    params.setdefault(
        "thread_count", lead_cfg.get("catboost_params", {}).get("thread_count", 1)
    )

    if lead_cfg.get("fine_tuning", False):
        grid = {
            "depth": [4, 6, 8],
            "l2_leaf_reg": [1, 5, 10],
            "learning_rate": [0.01, 0.1],
        }
        space = {
            "depth": Integer(4, 8),
            "l2_leaf_reg": Integer(1, 10),
            "learning_rate": Real(0.01, 0.3, prior="log-uniform"),
        }
        base_model = CatBoostClassifier(**params)
        best_params = _run_hyperparameter_search(base_model, grid, space, X_train, y_train)
        params.update(best_params)

    model_cat = CatBoostClassifier(**params)
    model_cat.fit(
        X_train, y_train, eval_set=(X_val, y_val), verbose=params.get("verbose", False)
    )

    model_path = out_dir / "models" / "lead_catboost.cbm"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_cat.save_model(str(model_path))

    val_pred = model_cat.predict_proba(X_val)[:, 1]
    pd.Series(val_pred).to_csv(data_dir / "proba_catboost.csv", index=False)
    metrics = {
        "logloss": log_loss(y_val, val_pred),
        "auc": roc_auc_score(y_val, val_pred),
    }
    return model_cat, metrics


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
    "train_logistic_lead",
    "train_xgboost_lead",
    "train_catboost_lead",
    "train_arima_conv_rate",
    "train_prophet_conv_rate",
]
