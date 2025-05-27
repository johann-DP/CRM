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
from dataset_loader import load_datasets  # reuse the autonomous loader
from data_preparation import prepare_data
from dataset_comparison import handle_missing_values, compare_datasets_versions
from factor_methods import run_pca, run_mca, run_famd, run_mfa
from nonlinear_methods import run_umap, run_phate, run_pacmap
from evaluate_methods import evaluate_methods, plot_methods_heatmap
from visualization import generate_figures
from unsupervised_cv import unsupervised_cv_and_temporal_tests
from pdf_report import export_report_to_pdf
from best_params import BEST_PARAMS


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


def select_variables(df: pd.DataFrame, min_modalite_freq: int = 5) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Select and prepare active variables for dimensional analysis.

    Parameters
    ----------
    df:
        DataFrame returned by :func:`prepare_data`.
    min_modalite_freq:
        Minimum frequency below which categorical levels are grouped into
        ``"Autre"``.

    Returns
    -------
    tuple
        ``(df_active, quantitative_vars, qualitative_vars)`` where
        ``df_active`` contains scaled numeric columns and categorical columns
        cast to ``category``.
    """
    logger = logging.getLogger(__name__)

    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")

    df = df.copy()

    # Columns to exclude based on data dictionary / prior knowledge
    exclude = {
        "Code",
        "ID",
        "Id",
        "Identifiant",
        "Client",
        "Contact principal",
        "Titre",
        "texte",
        "commentaire",
        "Commentaires",
    }

    # Drop constant columns, excluded columns and datetimes
    n_unique = df.nunique(dropna=False)
    constant_cols = n_unique[n_unique <= 1].index.tolist()
    drop_cols = set(constant_cols) | {c for c in df.columns if c in exclude}
    drop_cols.update([c for c in df.select_dtypes(include="datetime").columns])
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    quantitative_vars: list[str] = []
    qualitative_vars: list[str] = []

    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            series = pd.to_numeric(df[col], errors="coerce")
            if series.var(skipna=True) == 0 or series.isna().all():
                logger.warning("Variable quantitative '%s' exclue", col)
                continue
            df[col] = series.astype(float)
            quantitative_vars.append(col)
        else:
            series = df[col].astype("category")
            unique_ratio = series.nunique(dropna=False) / len(series)
            if unique_ratio > 0.8:
                logger.warning("Variable textuelle '%s' exclue", col)
                continue
            counts = series.value_counts(dropna=False)
            threshold = 0 if len(series) < min_modalite_freq else min_modalite_freq
            rares = counts[counts < threshold].index
            if len(rares) > 0:
                logger.info(
                    "%d modalités rares dans '%s' regroupées en 'Autre'",
                    len(rares),
                    col,
                )
                if "Autre" not in series.cat.categories:
                    series = series.cat.add_categories(["Autre"])
                series = series.apply(lambda x: "Autre" if x in rares else x).astype(
                    "category"
                )
            if series.nunique(dropna=False) <= 1:
                logger.warning("Variable qualitative '%s' exclue", col)
                continue
            df[col] = series
            qualitative_vars.append(col)

    df_active = df[quantitative_vars + qualitative_vars].copy()

    if quantitative_vars:
        scaler = StandardScaler()
        df_active[quantitative_vars] = scaler.fit_transform(df_active[quantitative_vars])

    for col in qualitative_vars:
        df_active[col] = df_active[col].astype("category")

    logger.info("DataFrame actif avec %d variables", len(df_active.columns))
    return df_active, quantitative_vars, qualitative_vars


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
