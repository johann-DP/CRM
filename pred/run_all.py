#!/usr/bin/env python3
"""Orchestration script for the forecasting modules in :mod:`pred`.

The helper loads the CRM data and **first cleans the closing dates**
using :func:`preprocess_dates`. This step removes aberrant values such as
dates in the year 2050 and returns the aggregated revenue series to the
pipeline. These series are subsequently preprocessed and used for every
model evaluation, guaranteeing that all transformations and forecasts
operate exclusively on corrected data.
Training and evaluation of each model run in parallel when several
``--jobs`` are specified.

Usage::

    python -m pred.run_all --config config.yaml [--jobs N]
"""
from __future__ import annotations

import argparse
import concurrent.futures
import os
from pathlib import Path
from typing import Dict

import yaml

from .preprocess_timeseries import preprocess_all
from .preprocess_dates import preprocess_dates
from .evaluate_models import (
    _evaluate_arima,
    _evaluate_prophet,
    _evaluate_xgb,
    _evaluate_lstm,
    _compute_metrics,
)
from .catboost_forecast import (
    prepare_supervised,
    rolling_forecast_catboost,
)
from .compare_granularities import build_performance_table
from .make_plots import main as make_plots_main


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------


def _eval_arima(m, q, y) -> Dict[str, Dict[str, float]]:
    return {
        "monthly": _evaluate_arima(m, 12, seasonal=True, m=12),
        "quarterly": _evaluate_arima(q, 4, seasonal=True, m=4),
        "yearly": _evaluate_arima(y, 3, seasonal=False, m=1),
    }


def _eval_prophet(m, q, y) -> Dict[str, Dict[str, float]]:
    return {
        "monthly": _evaluate_prophet(m, 12, yearly_seasonality=True),
        "quarterly": _evaluate_prophet(q, 4, yearly_seasonality=True),
        "yearly": _evaluate_prophet(y, 3, yearly_seasonality=False),
    }


def _eval_xgb(m, q, y) -> Dict[str, Dict[str, float]]:
    return {
        "monthly": _evaluate_xgb(m, 12, n_lags=12, add_time_features=True),
        "quarterly": _evaluate_xgb(q, 4, n_lags=4, add_time_features=True),
        "yearly": _evaluate_xgb(y, 3, n_lags=3, add_time_features=False),
    }


def _eval_lstm(m, q, y) -> Dict[str, Dict[str, float]]:
    return {
        "monthly": _evaluate_lstm(m, 12, window=12),
        "quarterly": _evaluate_lstm(q, 4, window=4),
        "yearly": _evaluate_lstm(y, 3, window=3),
    }


def _eval_catboost(m, q, y) -> Dict[str, Dict[str, float]]:
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


def evaluate_all(m, q, y, *, jobs: int) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Return metrics for each model in parallel when ``jobs`` > 1."""
    results: Dict[str, Dict[str, Dict[str, float]]] = {}
    if jobs > 1:
        with concurrent.futures.ProcessPoolExecutor(max_workers=jobs) as ex:
            futs = {ex.submit(func, m, q, y): name for name, func in EVAL_FUNCS.items()}
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
                results[name] = func(m, q, y)
            except Exception as exc:  # pragma: no cover - passthrough
                print(f"{name} failed: {exc}")
                results[name] = {}
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Run forecasting pipeline")
    p.add_argument(
        "--config", default="config.yaml", help="Fichier de configuration YAML"
    )
    p.add_argument(
        "--jobs",
        type=int,
        default=os.cpu_count(),
        help="Nombre de processus paralleles",
    )
    args = p.parse_args(argv)

    with open(args.config, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    csv_path = Path(cfg.get("input_file_cleaned_3_multi", "cleaned_3_multi.csv"))
    output_dir = Path(cfg.get("output_dir", "."))

    # ------------------------------------------------------------------
    # Stage 1 - cleaning closing dates before any other transformation
    # ------------------------------------------------------------------
    monthly, quarterly, yearly = preprocess_dates(csv_path, output_dir)

    # ------------------------------------------------------------------
    # Stage 2 - generic preprocessing of the aggregated time series
    # ------------------------------------------------------------------
    monthly, quarterly, yearly = preprocess_all(monthly, quarterly, yearly)

    results = evaluate_all(monthly, quarterly, yearly, jobs=args.jobs)
    table = build_performance_table(results)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / "model_performance.csv"
    print(table.to_string())
    table.to_csv(out_file)

    # Generate illustrative figures using the cleaned time series
    make_plots_main(
        str(output_dir),
        csv_path=None,
        metrics=table,
        ts_monthly=monthly,
        ts_quarterly=quarterly,
        ts_yearly=yearly,
    )


if __name__ == "__main__":  # pragma: no cover - CLI helper
    main()
