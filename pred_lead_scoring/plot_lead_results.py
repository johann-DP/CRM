#!/usr/bin/env python3
"""Generate summary plots for the lead scoring pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path
import os

import yaml
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.calibration import calibration_curve
from xgboost import XGBRegressor
from statsforecast.models import AutoARIMA
from prophet import Prophet
from catboost import CatBoostRegressor
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import numpy as np

from pred_aggregated_amount.train_xgboost import _to_supervised
from pred_aggregated_amount.lstm_forecast import create_lstm_sequences
from pred_aggregated_amount.catboost_forecast import (
    prepare_supervised,
    rolling_forecast_catboost,
)


# ---------------------------------------------------------------------------
# Plotting utilities
# ---------------------------------------------------------------------------


def _plot_roc(ax, y_true, probas, labels):
    """Plot ROC curves for several models on ``ax``."""
    for proba, label in zip(probas, labels):
        fpr, tpr, _ = roc_curve(y_true, proba)
        ax.plot(fpr, tpr, label=f"{label} (AUC={auc(fpr, tpr):.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()
    ax.set_title("ROC Curves")


def _plot_calibration(ax, y_true, probas, labels):
    """Plot reliability curves for each model."""
    for proba, label in zip(probas, labels):
        frac_pos, mean_pred = calibration_curve(y_true, proba, n_bins=10)
        ax.plot(mean_pred, frac_pos, marker="o", label=label)
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.legend()
    ax.set_title("Calibration plots")


def _plot_conv_rate(ax, ts_test, arima, prophet):
    """Plot conversion rate forecasts against actuals."""
    ax.plot(ts_test.index, ts_test.values, label="Actual", marker="o")
    ax.plot(arima.index, arima.values, label="ARIMA")
    ax.plot(prophet.index, prophet.values, label="Prophet")
    ax.set_xlabel("Date")
    ax.set_ylabel("Conversion rate")
    ax.legend()
    ax.set_title("Forecast comparison")


def _plot_histograms(axs, y_true, probas, labels):
    """Plot predicted probability histograms for won/lost classes."""
    for ax, proba, label in zip(axs, probas, labels):
        ax.hist(proba[y_true == 1], bins=20, alpha=0.5, label="Won")
        ax.hist(proba[y_true == 0], bins=20, alpha=0.5, label="Lost")
        ax.set_title(label)
        ax.set_xlabel("Predicted probability")
        ax.set_ylabel("Count")
        ax.legend()


def _plot_conf_matrices(axs, y_true, probas, labels):
    """Plot confusion matrices at threshold 0.5."""
    for ax, proba, label in zip(axs, probas, labels):
        cm = confusion_matrix(y_true, proba > 0.5)
        im = ax.imshow(cm, cmap="Blues", vmin=0)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(label)
    return im


def _plot_metrics_bar(ax, metrics):
    """Plot bar chart comparing global metrics."""
    metrics.plot.bar(x="model", y=["auc", "logloss"], ax=ax)
    ax.set_ylabel("Value")
    ax.set_title("Classification metrics")
    ax.legend()


def _plot_scatter(df: pd.DataFrame, col_date: str, col_val: str, out: Path, *, sort_dates: bool = False) -> None:
    """Scatter or line plot of ``col_val`` against ``col_date``."""
    if sort_dates:
        df = df.sort_values(col_date)
    plt.figure(figsize=(8, 4))
    plt.plot(df[col_date], df[col_val], marker=".")
    plt.xlabel(col_date)
    plt.ylabel(col_val)
    plt.tight_layout()
    plt.savefig(out)
    plt.close()


def _rolling_arima(series: pd.Series, horizon: int, *, seasonal: bool, m: int) -> pd.Series:
    """Return ARIMA one-step predictions for the last ``horizon`` periods."""
    history = series.iloc[:-horizon].copy()
    test = series.iloc[-horizon:]
    preds = []
    for t, val in enumerate(test):
        model = AutoARIMA(season_length=m if seasonal else 1)
        model.fit(history.values)
        res = model.predict(h=1)
        if isinstance(res, pd.DataFrame):
            pred = float(res["mean"].iloc[0])
        elif hasattr(res, "__getitem__"):
            try:
                pred = float(res[0])
            except Exception:
                pred = float(res["mean"].iloc[0])
        else:
            pred = float(res)
        preds.append(pred)
        history.loc[test.index[t]] = val
    return pd.Series(preds, index=test.index)


def _rolling_prophet(series: pd.Series, horizon: int, *, yearly: bool) -> pd.Series:
    """Return Prophet one-step predictions for ``horizon`` periods."""
    history = series.iloc[:-horizon].copy()
    test = series.iloc[-horizon:]
    preds = []
    freq = series.index.freqstr or "M"
    for t, val in enumerate(test):
        model = Prophet(yearly_seasonality=yearly, weekly_seasonality=False, daily_seasonality=False)
        df_hist = pd.DataFrame({"ds": history.index, "y": history.values})
        model.fit(df_hist)
        future = model.make_future_dataframe(periods=1, freq=freq)
        forecast = model.predict(future)
        preds.append(float(forecast.iloc[-1]["yhat"]))
        history.loc[test.index[t]] = val
    return pd.Series(preds, index=test.index)


def _rolling_xgb(series: pd.Series, horizon: int, *, n_lags: int, add_time_features: bool) -> pd.Series:
    """Return XGBoost one-step predictions for ``horizon`` periods."""
    history = series.iloc[:-horizon].copy()
    test = series.iloc[-horizon:]
    preds = []
    freq = series.index.freqstr or pd.infer_freq(series.index)
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

        next_idx = test.index[t]
        extended = pd.concat([history, pd.Series([pd.NA], index=[next_idx])])
        extended = extended.asfreq(freq)
        X_full, _ = _to_supervised(extended, n_lags, add_time_features=add_time_features)
        X_pred = X_full.iloc[-1:]
        pred = float(model.predict(X_pred)[0])
        preds.append(pred)
        history.loc[next_idx] = val
    return pd.Series(preds, index=test.index)


def _rolling_lstm(series: pd.Series, horizon: int, *, window: int, epochs: int = 20) -> pd.Series:
    """Return LSTM one-step predictions for ``horizon`` periods."""
    train = series.iloc[:-horizon]
    test = series.iloc[-horizon:]
    X, y = create_lstm_sequences(train, window)
    scaler = MinMaxScaler()
    scaler.fit(np.vstack([X.reshape(-1, 1), y.reshape(-1, 1)]))
    X_s = scaler.transform(X.reshape(-1, 1)).reshape(X.shape)
    y_s = scaler.transform(y.reshape(-1, 1)).reshape(-1)

    model = Sequential([LSTM(50, input_shape=(window, 1)), Dense(1)])
    model.compile(optimizer="adam", loss="mse")
    model.fit(X_s, y_s, epochs=epochs, batch_size=16, verbose=0)

    history = train.copy()
    preds = []
    for t, val in enumerate(test):
        seq = history.values[-window:].reshape(1, window, 1)
        seq_s = scaler.transform(seq.reshape(-1, 1)).reshape(1, window, 1)
        pred_s = model.predict(seq_s, verbose=0)[0, 0]
        pred = scaler.inverse_transform([[pred_s]])[0, 0]
        preds.append(float(pred))
        history.loc[test.index[t]] = val
    return pd.Series(preds, index=test.index)


def _rolling_catboost(series: pd.Series, freq: str, horizon: int) -> pd.Series:
    """Return CatBoost one-step predictions for ``horizon`` periods."""
    df_sup = prepare_supervised(series, freq)
    preds, _ = rolling_forecast_catboost(df_sup, freq, test_size=horizon)
    return pd.Series(preds, index=series.index[-horizon:])


def _plot_rolling_forecasts(series: pd.Series, freq: str, out: Path) -> None:
    """Plot observed series with rolling forecasts from several models."""
    horizon = 60 if freq == "M" else (20 if freq == "Q" else 5)
    obs = series.tail(horizon)

    arima = _rolling_arima(series, horizon, seasonal=freq != "A", m={"M": 12, "Q": 4, "A": 1}[freq])
    prophet = _rolling_prophet(series, horizon, yearly=freq != "A")
    xgb = _rolling_xgb(series, horizon, n_lags={"M": 12, "Q": 4, "A": 3}[freq], add_time_features=True)
    lstm = _rolling_lstm(series, horizon, window={"M": 12, "Q": 4, "A": 3}[freq])
    cat = _rolling_catboost(series, freq, horizon)

    plt.figure(figsize=(10, 5))
    plt.plot(obs.index, obs.values, label="observed", color="black")
    plt.plot(arima.index, arima.values, label="ARIMA", color="red")
    plt.plot(prophet.index, prophet.values, label="Prophet", color="purple")
    plt.plot(xgb.index, xgb.values, label="XGBoost", color="blue")
    plt.plot(cat.index, cat.values, label="CatBoost", color="orange")
    plt.plot(lstm.index, lstm.values, label="LSTM", color="green")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out)
    plt.close()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Generate lead scoring plots")
    p.add_argument("--config", default="config.yaml", help="Path to YAML config")
    args = p.parse_args(argv)

    with open(args.config, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    lead_cfg = cfg.get("lead_scoring", {})
    out_dir = Path(lead_cfg.get("output_dir", cfg.get("output_dir", ".")))
    data_dir = out_dir / "data_cache"
    fig_dir = out_dir / "figures"
    os.makedirs(fig_dir, exist_ok=True)

    X_test = pd.read_csv(data_dir / "X_test.csv")
    y_test = pd.read_csv(data_dir / "y_test.csv").squeeze()
    ts_conv_rate_test = pd.read_csv(
        data_dir / "ts_conv_rate_test.csv", index_col=0, parse_dates=True
    )["conv_rate"]
    pred_arima = pd.read_csv(
        data_dir / "pred_arima.csv", index_col=0, parse_dates=True
    ).squeeze()
    pred_prophet = pd.read_csv(
        data_dir / "pred_prophet.csv", index_col=0, parse_dates=True
    ).squeeze()
    metrics = pd.read_csv(data_dir / "lead_metrics_summary.csv")

    proba_xgb = pd.read_csv(data_dir / "proba_xgboost.csv").squeeze()
    proba_cat = pd.read_csv(data_dir / "proba_catboost.csv").squeeze()
    proba_lstm = pd.read_csv(data_dir / "proba_lstm.csv").squeeze()

    labels = ["XGBoost", "CatBoost", "LSTM"]
    probas = [proba_xgb, proba_cat, proba_lstm]

    # ROC curves
    fig, ax = plt.subplots(figsize=(6, 5))
    _plot_roc(ax, y_test, probas, labels)
    fig.tight_layout()
    fig.savefig(fig_dir / "roc_curves.png")
    plt.close(fig)

    # Calibration plots
    fig, ax = plt.subplots(figsize=(6, 5))
    _plot_calibration(ax, y_test, probas, labels)
    fig.tight_layout()
    fig.savefig(fig_dir / "calibration_plots.png")
    plt.close(fig)

    # Conversion rate forecast comparison
    fig, ax = plt.subplots(figsize=(8, 4))
    _plot_conv_rate(ax, ts_conv_rate_test, pred_arima, pred_prophet)
    fig.tight_layout()
    fig.savefig(fig_dir / "conv_rate_forecast.png")
    plt.close(fig)

    # Probability histograms
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    _plot_histograms(axs, y_test, probas, labels)
    fig.tight_layout()
    fig.savefig(fig_dir / "proba_histograms.png")
    plt.close(fig)

    # Confusion matrices
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    _plot_conf_matrices(axs, y_test, probas, labels)
    fig.tight_layout()
    fig.savefig(fig_dir / "confusion_matrices.png")
    plt.close(fig)

    # Metrics barplot
    metrics_clf = metrics[metrics["model_type"] == "classifier"]
    fig, ax = plt.subplots(figsize=(6, 5))
    _plot_metrics_bar(ax, metrics_clf)
    fig.tight_layout()
    fig.savefig(fig_dir / "classification_metrics_barplot.png")
    plt.close(fig)

    # --------------------------------------------------------------
    # Additional opportunity and forecast visualisations
    # --------------------------------------------------------------
    if "input_path" in lead_cfg:
        try:
            df_opportunities = pd.read_csv(lead_cfg["input_path"])
            df_opportunities["Date de fin actualisée"] = pd.to_datetime(
                df_opportunities["Date de fin actualisée"], dayfirst=False, errors="coerce"
            )
        except Exception:
            df_opportunities = None
    else:
        df_opportunities = None

    if df_opportunities is not None and {"Date de fin actualisée", "Total_estime"} <= set(df_opportunities.columns):
        _plot_scatter(
            df_opportunities,
            "Date de fin actualisée",
            "Total_estime",
            fig_dir / "total_vs_date.png",
            sort_dates=False,
        )
        _plot_scatter(
            df_opportunities,
            "Date de fin actualisée",
            "Total_estime",
            fig_dir / "total_vs_date_sorted.png",
            sort_dates=True,
        )

        ts = df_opportunities.set_index("Date de fin actualisée")["Total_estime"].dropna()
        ts_monthly = ts.resample("M").sum()
        ts_quarterly = ts.resample("Q").sum()
        ts_yearly = ts.resample("A").sum()

        _plot_rolling_forecasts(ts_monthly, "M", fig_dir / "monthly_forecasts.png")
        _plot_rolling_forecasts(ts_quarterly, "Q", fig_dir / "quarterly_forecasts.png")
        _plot_rolling_forecasts(ts_yearly, "A", fig_dir / "yearly_forecasts.png")

    # Metrics comparison barplot across granularities
    if not metrics.empty:
        try:
            df_metrics = metrics.copy()
            required_cols = [
                "model",
                "MAE_monthly",
                "RMSE_monthly",
                "MAPE_monthly",
                "MAE_quarterly",
                "RMSE_quarterly",
                "MAPE_quarterly",
                "MAE_yearly",
                "RMSE_yearly",
                "MAPE_yearly",
            ]
            if set(required_cols) <= set(df_metrics.columns):
                fig, ax = plt.subplots(figsize=(12, 6))
                df_metrics.set_index("model")[required_cols[1:]].plot.bar(ax=ax)
                ax.set_ylabel("Metric value")
                ax.set_title("Comparaison des métriques agrégées")
                ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
                fig.tight_layout()
                fig.savefig(fig_dir / "metrics_comparison_barplot.png")
                plt.close(fig)
        except Exception:
            pass


if __name__ == "__main__":  # pragma: no cover - simple CLI
    main()
