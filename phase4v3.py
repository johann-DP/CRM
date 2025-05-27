#!/usr/bin/env python3
"""Modular and reproducible pipeline for Phase 4 analyses."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    yaml = None

from data_processing import load_datasets, select_variables
from phase4v2 import handle_missing_values
from dim_reduction import run_all_factor_methods, run_all_nonlin
from block6_visualization import generate_figures
from evaluation import (
    evaluate_methods,
    plot_methods_heatmap,
    unsupervised_cv_and_temporal_tests,
)
from reporting import export_report_to_pdf


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _setup_logging(log_file: Path) -> None:
    """Configure root logger writing to console and ``log_file``."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode="w", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the complete phase 4 pipeline based on a configuration file."""
    parser = argparse.ArgumentParser(description="Run phase 4 pipeline")
    parser.add_argument("--config", required=True, help="YAML or JSON configuration")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        raise FileNotFoundError(cfg_path)

    with open(cfg_path, "r", encoding="utf-8") as fh:
        if cfg_path.suffix.lower() == ".json":
            config: Dict[str, Any] = json.load(fh)
        else:
            if yaml is None:
                raise RuntimeError("PyYAML is required for YAML configuration")
            config = yaml.safe_load(fh)

    out_dir = Path(config.get("output_dir", "phase4_output"))
    out_dir.mkdir(parents=True, exist_ok=True)
    _setup_logging(out_dir / "phase4.log")
    logging.info("Configuration loaded from %s", cfg_path)

    seed = int(config.get("random_seed", 0))
    np.random.seed(seed)

    # ------------------------------------------------------------------
    # Data loading and preparation
    # ------------------------------------------------------------------
    datasets = load_datasets(config)
    df = (
        datasets.get("phase3")
        or datasets.get("phase2")
        or datasets.get("phase1")
        or datasets["raw"]
    )

    df_active, quant_vars, qual_vars = select_variables(df)
    df_active = handle_missing_values(df_active, quant_vars, qual_vars)

    # ------------------------------------------------------------------
    # Dimensionality reduction
    # ------------------------------------------------------------------
    factor_results = run_all_factor_methods(
        df_active,
        quant_vars,
        qual_vars,
        groups=config.get("mfa_groups"),
    )
    nonlin_results = run_all_nonlin(
        df_active,
        umap_params=config.get("umap", {}),
    )

    # ------------------------------------------------------------------
    # Optional cross-validation and temporal tests
    # ------------------------------------------------------------------
    cv_temporal = {}
    if config.get("run_temporal_tests", True):
        cv_temporal = unsupervised_cv_and_temporal_tests(
            df_active,
            quant_vars,
            qual_vars,
            n_splits=int(config.get("cv_splits", 5)),
        )

    # ------------------------------------------------------------------
    # Visualisations and metrics
    # ------------------------------------------------------------------
    figs = generate_figures(factor_results, nonlin_results, df_active, quant_vars, qual_vars)
    metrics = evaluate_methods({**factor_results, **nonlin_results}, df_active, quant_vars, qual_vars)

    metrics.to_csv(out_dir / "methods_comparison.csv")
    plot_methods_heatmap(metrics, out_dir)

    for name, fig in figs.items():
        fig.savefig(out_dir / f"{name}.png")

    if cv_temporal:
        with open(out_dir / "cv_temporal_results.json", "w", encoding="utf-8") as fh:
            json.dump(cv_temporal, fh, ensure_ascii=False, indent=2)

    tables = {
        "Comparaison des methodes": metrics,
        "Validation croisee / Temporal": pd.DataFrame.from_dict(cv_temporal, orient="index"),
    }

    output_pdf = Path(config.get("output_pdf", out_dir / "phase4_report.pdf"))
    export_report_to_pdf(figs, tables, output_pdf)
    logging.info("Report generated at %s", output_pdf)


if __name__ == "__main__":
    main()
