#!/usr/bin/env python
"""Master script orchestrating the lead scoring pipeline.

Pipeline overview
-----------------
The preprocessing step from :mod:`preprocess_lead_scoring` must run first as
it requires the raw CSV export.  Once the datasets are written to disk the
training routines can be parallelised:

- ``train_xgboost_lead``
- ``train_catboost_lead``
- ``train_mlp_lead``

The two forecasting models, ``train_arima_conv_rate`` and
``train_prophet_conv_rate``, may also run concurrently after preprocessing.
Finally :mod:`evaluate_lead_models` is executed sequentially once all models
have been trained and saved.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import logging
import multiprocessing as mp
import os
from pathlib import Path
import yaml
import pandas as pd

from .logging_utils import setup_logging

from .preprocess_lead_scoring import preprocess
from .train_lead_models import (
    train_xgboost_lead,
    train_catboost_lead,
    train_logistic_lead,
    train_mlp_lead,
    train_ensemble_lead,
    train_arima_conv_rate,
    train_prophet_conv_rate,
)
from .evaluate_lead_models import evaluate_lead_models
from .plot_lead_results import main as plot_results

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> None:
    setup_logging(log_file="training_output.txt")
    p = argparse.ArgumentParser(description="Run full lead scoring pipeline")
    p.add_argument("--config", default="config.yaml", help="Path to YAML config")
    args = p.parse_args(argv)

    with open(args.config, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    try:
        mp.set_start_method("forkserver", force=True)
    except ValueError:  # pragma: no cover - Windows fallback
        mp.set_start_method("spawn", force=True)

    (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        ts_conv_train,
        ts_conv_test,
        df_prophet_train,
    ) = preprocess(cfg)

    # ------------------------------------------------------------------
    # Classification models
    # ------------------------------------------------------------------
    executor_cls = (
        concurrent.futures.ProcessPoolExecutor
        if "PYTEST_CURRENT_TEST" not in os.environ
        else concurrent.futures.ThreadPoolExecutor
    )
    with executor_cls(max_workers=mp.cpu_count()) as ex:
        fut_xgb = ex.submit(train_xgboost_lead, cfg, X_train, y_train, X_val, y_val)
        fut_cat = ex.submit(train_catboost_lead, cfg, X_train, y_train, X_val, y_val)
        fut_log = ex.submit(train_logistic_lead, cfg, X_train, y_train, X_val, y_val)
        fut_mlp = ex.submit(train_mlp_lead, cfg, X_train, y_train, X_val, y_val)

        # Retrieve results to surface potential exceptions from worker threads
        fut_xgb.result()
        fut_cat.result()
        fut_log.result()
        fut_mlp.result()

    # Ensemble model using trained XGBoost and CatBoost
    train_ensemble_lead(cfg, X_val, y_val)

    # ------------------------------------------------------------------
    # Forecast models
    # ------------------------------------------------------------------
    with executor_cls(max_workers=mp.cpu_count()) as ex:
        fut_arima = ex.submit(
            train_arima_conv_rate,
            cfg,
            ts_conv_train["conv_rate"],
            ts_conv_test["conv_rate"],
        )
        fut_prophet = ex.submit(
            train_prophet_conv_rate, cfg, df_prophet_train, ts_conv_test["conv_rate"]
        )

        # Ensure forecasting models completed successfully
        fut_arima.result()
        fut_prophet.result()

    df_metrics = evaluate_lead_models(cfg, X_test, y_test, ts_conv_test["conv_rate"])

    def _format_report(df: pd.DataFrame) -> str:
        if "model_type" not in df.columns:
            return df.to_string(index=False)

        lines: list[str] = []
        for mtype, grp in df.groupby("model_type"):
            lines.append(f"{mtype.capitalize()} models")
            metrics_cols = [
                c
                for c in grp.columns
                if c not in {"model", "model_type", "tn", "fp", "fn", "tp"}
            ]
            best_idx = {}
            for col in metrics_cols:
                if col in {"logloss", "brier", "mae", "rmse", "mape"}:
                    best_idx[col] = grp[col].idxmin()
                else:
                    best_idx[col] = grp[col].idxmax()
            for idx, row in grp.iterrows():
                lines.append(f"  {row['model']}")
                for col in metrics_cols:
                    val = row[col]
                    star = "*" if idx == best_idx[col] else ""
                    if col in {"precision", "recall", "lost_recall", "balanced_accuracy", "f1"}:
                        lines.append(f"    {col}: {val*100:.2f}%{star}")
                    else:
                        lines.append(f"    {col}: {val:.3f}{star}")
                if {"tn", "fp", "fn", "tp"} <= set(grp.columns):
                    lines.append(
                        f"    TN={row['tn']} FP={row['fp']} FN={row['fn']} TP={row['tp']}"
                    )
                lines.append("")
        return "\n".join(lines)

    report_text = _format_report(df_metrics)
    logger.info("\n%s", report_text)

    lead_cfg = cfg.get("lead_scoring", {})
    out_dir = Path(lead_cfg.get("output_dir", cfg.get("output_dir", ".")))
    report_file = out_dir / "lead_scoring_report.txt"
    try:
        report_file.write_text(report_text)
    except Exception:
        pass

    # Generate all figures summarising the results
    try:
        plot_results(["--config", args.config])
    except Exception:
        pass


if __name__ == "__main__":  # pragma: no cover - simple script
    main()
