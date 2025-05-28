#!/usr/bin/env python3
"""Phase 4 pipeline orchestrating all modular functions.

This script ties together the helper modules present in the repository
(`data_preparation`, `variable_selection`, `factor_methods`,
`nonlinear_methods`, `evaluate_methods`, `visualization`,
`dataset_comparison`, `unsupervised_cv`, `pdf_report`) to reproduce the
complete dimensionality-reduction workflow.  It delegates the heavy
lifting to these modules and only handles configuration and ordering of
operations.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from sklearn.preprocessing import StandardScaler

import numpy as np
import pandas as pd
import yaml

# Import helper modules -------------------------------------------------------
from phase4_functions import (
    load_datasets,
    prepare_data,
    handle_missing_values,
    compare_datasets_versions,
    run_pca,
    run_mca,
    run_famd,
    run_mfa,
    run_umap,
    run_phate,
    run_pacmap,
    evaluate_methods,
    plot_methods_heatmap,
    generate_figures,
    select_variables,
    unsupervised_cv_and_temporal_tests,
    export_report_to_pdf,
    BEST_PARAMS,
)


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _setup_logging(output_dir: Path, level: str = "INFO") -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger()
    logger.setLevel(level)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    file_handler = logging.FileHandler(output_dir / "phase4.log", encoding="utf-8")
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(fmt)
    logger.addHandler(stream_handler)
    return logger


def _load_config(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        if path.suffix.lower() in {".yaml", ".yml"}:
            return yaml.safe_load(fh)
        return json.load(fh)


def _method_params(method: str, config: Mapping[str, Any]) -> Dict[str, Any]:
    params = BEST_PARAMS.get(method.upper(), {}).copy()
    if method.lower() in config and isinstance(config[method.lower()], Mapping):
        params.update(config[method.lower()])
    prefix = f"{method.lower()}_"
    for key, value in config.items():
        if key.startswith(prefix):
            params[key[len(prefix) :]] = value
    return params


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_pipeline(config: Dict[str, Any]) -> Dict[str, Any]:
    rs = config.get("random_state")
    random_state = int(rs) if rs is not None else None
    output_dir = Path(config.get("output_dir", "phase4_output"))
    _setup_logging(output_dir)

    datasets = load_datasets(config)
    data_key = config.get("dataset", config.get("main_dataset", "raw"))
    if data_key not in datasets:
        raise KeyError(f"dataset '{data_key}' not found")

    logging.info("Running pipeline on dataset '%s'", data_key)

    df_prep = prepare_data(datasets[data_key], exclude_lost=bool(config.get("exclude_lost", True)))
    df_active, quant_vars, qual_vars = select_variables(
        df_prep, min_modalite_freq=int(config.get("min_modalite_freq", 5))
    )
    df_active = handle_missing_values(df_active, quant_vars, qual_vars)

    methods = [m.lower() for m in config.get(
        "methods",
        ["pca", "mca", "famd", "mfa", "umap", "phate", "pacmap"],
    )]

    factor_results: Dict[str, Any] = {}
    if "pca" in methods and quant_vars:
        params = _method_params("pca", config)
        params.pop("n_components", None)
        factor_results["pca"] = run_pca(
            df_active,
            quant_vars,
            optimize=True,
            random_state=random_state,
            **params,
        )

    if "mca" in methods and qual_vars:
        params = _method_params("mca", config)
        params.pop("n_components", None)
        factor_results["mca"] = run_mca(
            df_active,
            qual_vars,
            optimize=True,
            random_state=random_state,
            **params,
        )

    if "famd" in methods and quant_vars and qual_vars:
        params = _method_params("famd", config)
        params.pop("n_components", None)
        try:
            factor_results["famd"] = run_famd(
                df_active,
                quant_vars,
                qual_vars,
                optimize=True,
                random_state=random_state,
                **params,
            )
        except ValueError as exc:
            logging.warning("FAMD skipped: %s", exc)

    groups = []
    if quant_vars:
        groups.append(quant_vars)
    if qual_vars:
        groups.append(qual_vars)
    if "mfa" in methods and len(groups) > 1:
        params = _method_params("mfa", config)
        params.pop("n_components", None)
        cfg_groups = params.pop("groups", None)
        if cfg_groups:
            groups = cfg_groups
        factor_results["mfa"] = run_mfa(
            df_active,
            groups,
            optimize=True,
            random_state=random_state,
            **params,
        )

    nonlin_results: Dict[str, Any] = {}
    if "umap" in methods:
        params = _method_params("umap", config)
        nonlin_results["umap"] = run_umap(df_active, random_state=random_state, **params)
    if "phate" in methods:
        params = _method_params("phate", config)
        nonlin_results["phate"] = run_phate(df_active, random_state=random_state, **params)
    if "pacmap" in methods:
        params = _method_params("pacmap", config)
        nonlin_results["pacmap"] = run_pacmap(df_active, random_state=random_state, **params)

    valid_nonlin = {
        k: v
        for k, v in nonlin_results.items()
        if isinstance(v.get("embeddings"), pd.DataFrame) and not v["embeddings"].empty
    }

    all_results = {**factor_results, **valid_nonlin}
    if not all_results:
        logging.warning("No results to evaluate")
        metrics = pd.DataFrame()
    else:
        metrics = evaluate_methods(
            all_results,
            df_active,
            quant_vars,
            qual_vars,
            n_clusters=3 if len(df_active) > 3 else 2,
        )
        metrics.to_csv(output_dir / "metrics.csv")
        plot_methods_heatmap(metrics, output_dir)

    figures = generate_figures(
        factor_results,
        nonlin_results,
        df_active,
        quant_vars,
        qual_vars,
        output_dir=output_dir,
    )

    comparison_metrics = None
    comparison_figures: Dict[str, Any] = {}
    if config.get("compare_versions"):
        versions = {k: v for k, v in datasets.items() if k != data_key}
        if versions:
            comp = compare_datasets_versions(
                versions,
                exclude_lost=bool(config.get("exclude_lost", True)),
                min_modalite_freq=int(config.get("min_modalite_freq", 5)),
                output_dir=output_dir / "comparisons",
            )
            comparison_metrics = comp["metrics"]
            comparison_figures = {
                f"{ver}_{name}": fig
                for ver, det in comp["details"].items()
                for name, fig in det["figures"].items()
            }
            comparison_metrics.to_csv(output_dir / "comparison_metrics.csv", index=False)

    robustness_df = None
    if config.get("run_temporal_tests"):
        robustness_df = unsupervised_cv_and_temporal_tests(
            df_active,
            quant_vars,
            qual_vars,
            n_splits=int(config.get("n_splits", 5)),
            random_state=random_state,
        )
        pd.DataFrame(robustness_df).to_csv(output_dir / "robustness.csv")

    if config.get("output_pdf"):
        all_figs = {**figures, **comparison_figures}
        tables: Dict[str, Any] = {"metrics": metrics}
        if comparison_metrics is not None:
            tables["comparison_metrics"] = comparison_metrics
        if robustness_df is not None:
            tables["robustness"] = pd.DataFrame(robustness_df)
        export_report_to_pdf(all_figs, tables, config["output_pdf"])

    logging.info("Analysis complete")
    return {
        "metrics": metrics,
        "figures": figures,
        "comparison_metrics": comparison_metrics,
        "robustness": robustness_df,
    }



# ---------------------------------------------------------------------------
# CLI Entrypoint
# ---------------------------------------------------------------------------

def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Phase 4 analysis (modular)")
    parser.add_argument("--config", required=True, help="Path to config YAML/JSON")
    args = parser.parse_args(argv)

    np.random.seed(0)
    random.seed(0)

    cfg = _load_config(Path(args.config))
    run_pipeline(cfg)


if __name__ == "__main__":  # pragma: no cover - manual execution
    main()
