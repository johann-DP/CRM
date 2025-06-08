#!/usr/bin/env python3
"""Orchestration script for the forecasting modules in :mod:`pred`.

The helper loads the CRM data and **first cleans the closing dates**
using :func:`preprocess_dates`.  This step removes aberrant values such as
dates in the year 2050, imputes missing ones and returns the aggregated
revenue series to the pipeline.  It is mandatory that this cleaning stage
occurs before any other processing so that the following operations work
exclusively on corrected data.  The resulting monthly, quarterly and yearly
series are then preprocessed and used for every model evaluation,
guaranteeing consistent transformations and forecasts.
Training and evaluation of each model run in parallel when several
``--jobs`` are specified.

Usage::

    python -m pred.run_all --config config.yaml [--jobs N]
"""
from __future__ import annotations

import os
import warnings

# Silence verbose TensorFlow logs and deprecation notices from optional libs
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated as an API",
    category=UserWarning,
)

import argparse
from pathlib import Path

import yaml

from .preprocess_timeseries import preprocess_all
from .preprocess_dates import preprocess_dates
from .evaluate_models import evaluate_all_models
from .compare_granularities import build_performance_table
from .make_plots import main as make_plots_main


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
    p.add_argument(
        "--cross-val",
        action="store_true",
        default=True,
        help="Activer la validation croisee temporelle",
    )
    p.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="Nombre de segments pour la validation croisee",
    )
    args = p.parse_args(argv)

    with open(args.config, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    csv_path = Path(cfg.get("input_file_cleaned_3_all", "phase3_cleaned_all.csv"))
    output_dir = Path(cfg.get("output_dir", "."))

    # ------------------------------------------------------------------
    # Stage 1 - cleaning closing dates before any other transformation
    # ------------------------------------------------------------------
    monthly, quarterly, yearly = preprocess_dates(csv_path, output_dir)

    # Sanity check: no future dates should remain after cleaning
    for s in (monthly, quarterly, yearly):
        if (s.index.year >= 2040).any():
            raise ValueError(
                "preprocess_dates failed to remove future closing dates"
            )

    # ------------------------------------------------------------------
    # Stage 2 - generic preprocessing of the aggregated time series
    # ------------------------------------------------------------------

    monthly, quarterly, yearly = preprocess_all(monthly, quarterly, yearly)

    results = evaluate_all_models(
        monthly,
        quarterly,
        yearly,
        jobs=args.jobs,
        cross_val=args.cross_val,
        n_splits=args.n_splits,
    )
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
