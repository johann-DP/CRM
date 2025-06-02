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


if __name__ == "__main__":  # pragma: no cover - simple CLI
    main()
