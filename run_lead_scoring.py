#!/usr/bin/env python
"""Master script orchestrating the lead scoring pipeline.

Pipeline overview
-----------------
The preprocessing step from :mod:`preprocess_lead_scoring` must run first as
it requires the raw CSV export.  Once the datasets are written to disk the
training routines can be parallelised:

- ``train_xgboost_lead``
- ``train_catboost_lead``
- ``train_lstm_lead``

The two forecasting models, ``train_arima_conv_rate`` and
``train_prophet_conv_rate``, may also run concurrently after preprocessing.
Finally :mod:`evaluate_lead_models` is executed sequentially once all models
have been trained and saved.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import multiprocessing as mp
from pathlib import Path
import yaml

from preprocess_lead_scoring import preprocess
from train_lead_models import (
    train_xgboost_lead,
    train_catboost_lead,
    train_lstm_lead,
    train_arima_conv_rate,
    train_prophet_conv_rate,
)
from evaluate_lead_models import evaluate_lead_models
from plot_lead_results import main as plot_results


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Run full lead scoring pipeline")
    p.add_argument("--config", default="config.yaml", help="Path to YAML config")
    args = p.parse_args(argv)

    with open(args.config, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    mp.set_start_method("forkserver", force=True)

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
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:
        fut_xgb = ex.submit(train_xgboost_lead, cfg, X_train, y_train, X_val, y_val)
        fut_cat = ex.submit(train_catboost_lead, cfg, X_train, y_train, X_val, y_val)
        fut_lstm = ex.submit(train_lstm_lead, cfg, X_train, y_train, X_val, y_val)

        model_xgb, metrics_xgb = fut_xgb.result()
        model_cat, metrics_cat = fut_cat.result()
        model_lstm, metrics_lstm = fut_lstm.result()

    # ------------------------------------------------------------------
    # Forecast models
    # ------------------------------------------------------------------
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
        fut_arima = ex.submit(
            train_arima_conv_rate,
            cfg,
            ts_conv_train["conv_rate"],
            ts_conv_test["conv_rate"],
        )
        fut_prophet = ex.submit(
            train_prophet_conv_rate, cfg, df_prophet_train, ts_conv_test["conv_rate"]
        )

        model_arima, metrics_arima = fut_arima.result()
        model_prophet, metrics_prophet = fut_prophet.result()

    df_metrics = evaluate_lead_models(cfg, X_test, y_test, ts_conv_test["conv_rate"])
    print(df_metrics.to_string(index=False))

    lead_cfg = cfg.get("lead_scoring", {})
    out_dir = Path(lead_cfg.get("output_dir", cfg.get("output_dir", ".")))
    report_file = out_dir / "lead_scoring_report.txt"
    try:
        report_file.write_text(df_metrics.to_string(index=False))
    except Exception:
        pass

    # Generate all figures summarising the results
    try:
        plot_results(["--config", args.config])
    except Exception:
        pass


if __name__ == "__main__":  # pragma: no cover - simple script
    main()
