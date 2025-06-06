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
import multiprocessing as mp
import os
from pathlib import Path
import yaml
import pandas as pd

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



def main(argv: list[str] | None = None) -> None:
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
    # Using ``ProcessPoolExecutor`` caused excessive overhead and very slow
    # start-up times on some platforms (notably Windows).  The training
    # functions already leverage multi-threaded libraries, so a thread pool is
    # sufficient here and keeps the pipeline responsive.
    executor_cls = concurrent.futures.ThreadPoolExecutor

    with executor_cls(max_workers=4) as ex:
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
    with executor_cls(max_workers=2) as ex:
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

    # ------------------------------------------------------------------
    # Format textual report in the historic table layout
    # ------------------------------------------------------------------
    columns = [
        "model",
        "model_type",
        "logloss",
        "auc",
        "precision",
        "recall",
        "lost_recall",
        "f1",
        "mae",
        "rmse",
        "mape",
    ]
    for col in columns:
        if col not in df_metrics.columns:
            df_metrics[col] = pd.NA
    report_text = df_metrics[columns].to_string(index=False)
    print("\n" + report_text)

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
