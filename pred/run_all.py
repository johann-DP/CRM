#!/usr/bin/env python3
"""Orchestration script for the forecasting modules in :mod:`pred`.

The helper loads the CRM data, preprocesses the revenue time series and
computes evaluation metrics for all available models.  Training and
evaluation of each model run in parallel when several ``--jobs`` are
specified.

The path to the cleaned dataset is read from ``config.yaml`` so that the
forecasting pipeline uses the same files as the rest of the project.

Usage::

    python -m pred.run_all --config config.yaml [--jobs N]
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
from pathlib import Path
from typing import Any, Dict

import yaml

from .aggregate_revenue import build_timeseries
from .preprocess_timeseries import preprocess_all
from .evaluate_models import (
    _evaluate_arima,
    _evaluate_prophet,
    _evaluate_xgb,
    _evaluate_lstm,
)
from .compare_granularities import build_performance_table


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


EVAL_FUNCS = {
    "ARIMA": _eval_arima,
    "Prophet": _eval_prophet,
    "XGBoost": _eval_xgb,
    "LSTM": _eval_lstm,
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

def _load_config(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        if path.suffix.lower() in {".yaml", ".yml"}:
            return yaml.safe_load(fh)
        return json.load(fh)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Run forecasting pipeline")
    p.add_argument("--config", default="config.yaml", help="Chemin du fichier de configuration")
    p.add_argument("--jobs", type=int, help="Nombre de processus paralleles")
    args = p.parse_args(argv)

    cfg = _load_config(Path(args.config))
    csv_path = cfg.get("input_file_cleaned_3_multi")
    if not csv_path:
        p.error("'input_file_cleaned_3_multi' manquant dans la configuration")

    jobs = args.jobs if args.jobs is not None else int(cfg.get("n_jobs", 1))

    monthly, quarterly, yearly = build_timeseries(Path(csv_path))
    monthly, quarterly, yearly = preprocess_all(monthly, quarterly, yearly)

    results = evaluate_all(monthly, quarterly, yearly, jobs=jobs)
    table = build_performance_table(results)

    out_dir = Path(cfg.get("output_dir", "."))
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "model_performance.csv"

    print(table.to_string())
    table.to_csv(out_file)


if __name__ == "__main__":  # pragma: no cover - CLI helper
    main()
