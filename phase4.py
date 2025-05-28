#!/usr/bin/env python3
"""Phase 4 pipeline orchestrating all modular functions.

This script ties together the helper modules present in the repository
(`data_preparation`, `variable_selection`, `factor_methods`,
`nonlinear_methods`, `evaluate_methods`, `visualization`,
`dataset_comparison`, `unsupervised_cv`, `pdf_report`) to reproduce the
complete dimensionality-reduction workflow.  It delegates the heavy
lifting to these modules and only handles configuration and ordering of
operations.

Run the script with a YAML or JSON configuration file::

    python phase4.py --config config.yaml

The pinned dependencies listed in :code:`requirements.txt` must be
installed in order to reproduce the results reliably.  Use
``python -m pip install -r requirements.txt`` inside a fresh virtual
environment before executing the pipeline.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

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
        for key, value in config[method.lower()].items():
            if value is not None:
                params[key] = value
    prefix = f"{method.lower()}_"
    for key, value in config.items():
        if key.startswith(prefix) and value is not None:
            params[key[len(prefix) :]] = value
    return params


def build_pdf_report(
    output_dir: Path,
    pdf_path: Path,
    dataset_order: Sequence[str],
    tables: Optional[Mapping[str, pd.DataFrame]] = None,
) -> Path:
    """Assemble all PNG figures under ``output_dir`` into ``pdf_path``.

    A title page is added followed by sections for each dataset listed in
    ``dataset_order``. Any provided tables are rendered as figures and appended
    at the end of the document.
    """

    pdf_path.parent.mkdir(parents=True, exist_ok=True)

    def _format_caption(dataset: str, filename: str) -> str:
        name = filename.rsplit(".", 1)[0]
        parts = name.split("_")
        method = parts[0].upper() if parts else ""
        suffix = "_".join(parts[1:]) if len(parts) > 1 else ""
        if "scree" in suffix:
            desc = f"Éboulis {method}"
        elif "correlation" in suffix:
            desc = f"Cercle de corrélation {method}"
        elif "contributions" in suffix:
            desc = f"Contributions des variables – {method}"
        elif "scatter_2d" in suffix:
            desc = f"Nuage d'individus – {method} (2D)"
        elif "clusters" in suffix:
            desc = f"Segmentation K-means sur projection {method}"
        elif "scatter_3d" in suffix:
            desc = f"Nuage 3D – {method}"
        else:
            desc = name
        return f"{dataset} – {desc}"

    def _add_image(pdf: PdfPages, img_path: Path, dataset: str) -> None:
        img = plt.imread(img_path)
        fig, ax = plt.subplots(figsize=(8.27, 11.69), dpi=200)
        ax.imshow(img)
        ax.axis("off")
        ax.text(
            0.5,
            0.02,
            _format_caption(dataset, img_path.name),
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=8,
            color="gray",
        )
        pdf.savefig(fig)
        plt.close(fig)

    def _table_to_fig(df: pd.DataFrame, title: str) -> plt.Figure:
        height = 0.4 * len(df) + 1.5
        fig, ax = plt.subplots(figsize=(8.0, height), dpi=200)
        ax.axis("off")
        ax.set_title(title)
        table = ax.table(
            cellText=df.values,
            colLabels=list(df.columns),
            rowLabels=list(df.index),
            cellLoc="center",
            rowLoc="center",
            loc="center",
        )
        table.scale(1, 1.2)
        fig.tight_layout()
        return fig

    with PdfPages(pdf_path) as pdf:
        # Title page
        fig, ax = plt.subplots(figsize=(8.27, 11.69), dpi=200)
        ax.axis("off")
        ax.text(
            0.5,
            0.6,
            "Phase 4 Dimensional Analysis – Comparative Report",
            ha="center",
            va="center",
            fontsize=16,
            weight="bold",
        )
        ax.text(
            0.5,
            0.52,
            datetime.datetime.now().strftime("%Y-%m-%d"),
            ha="center",
            va="center",
            fontsize=12,
        )
        ax.text(
            0.5,
            0.44,
            "Automated compilation of Phase 4 results",
            ha="center",
            va="center",
            fontsize=10,
        )
        pdf.savefig(fig)
        plt.close(fig)

        for name in dataset_order:
            # Section page
            fig, ax = plt.subplots(figsize=(8.27, 11.69), dpi=200)
            ax.axis("off")
            ax.text(0.5, 0.9, name, ha="center", va="top", fontsize=14, weight="bold")
            pdf.savefig(fig)
            plt.close(fig)

            if name == dataset_order[0]:
                base_dir = output_dir
            else:
                base_dir = output_dir / "comparisons" / name
            if not base_dir.exists():
                continue
            for img in sorted(base_dir.rglob("*.png")):
                _add_image(pdf, img, name)

        if tables:
            for tname, df in tables.items():
                fig = _table_to_fig(df, tname)
                pdf.savefig(fig)
                plt.close(fig)

    return pdf_path


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def run_pipeline(config: Dict[str, Any]) -> Dict[str, Any]:
    output_dir = Path(config.get("output_dir", "phase4_output"))
    _setup_logging(output_dir)

    logging.info("Loading datasets...")
    datasets = load_datasets(config)
    data_key = config.get("dataset", config.get("main_dataset", "raw"))
    if data_key not in datasets:
        raise KeyError(f"dataset '{data_key}' not found")

    logging.info("Running pipeline on dataset '%s'", data_key)

    logging.info("Preparing data...")
    df_prep = prepare_data(
        datasets[data_key], exclude_lost=bool(config.get("exclude_lost", True))
    )
    logging.info("Selecting variables...")
    df_active, quant_vars, qual_vars = select_variables(
        df_prep, min_modalite_freq=int(config.get("min_modalite_freq", 5))
    )
    logging.info("Handling missing values...")
    df_active = handle_missing_values(df_active, quant_vars, qual_vars)

    methods = [
        m.lower()
        for m in config.get(
            "methods_to_run",
            config.get(
                "methods",
                ["pca", "mca", "famd", "mfa", "umap", "phate", "pacmap"],
            ),
        )
    ]

    factor_results: Dict[str, Any] = {}
    if "pca" in methods and quant_vars:
        logging.info("Running PCA...")
        params = _method_params("pca", config)
        params.pop("n_components", None)
        factor_results["pca"] = run_pca(
            df_active,
            quant_vars,
            optimize=True,
            **params,
        )

    if "mca" in methods and qual_vars:
        logging.info("Running MCA...")
        params = _method_params("mca", config)
        params.pop("n_components", None)
        factor_results["mca"] = run_mca(
            df_active,
            qual_vars,
            optimize=True,
            **params,
        )

    if "famd" in methods and quant_vars and qual_vars:
        logging.info("Running FAMD...")
        params = _method_params("famd", config)
        params.pop("n_components", None)
        try:
            factor_results["famd"] = run_famd(
                df_active,
                quant_vars,
                qual_vars,
                optimize=True,
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
        logging.info("Running MFA...")
        params = _method_params("mfa", config)
        params.pop("n_components", None)
        cfg_groups = params.pop("groups", None)
        # ``mfa: {groups: [[...], [...]]}`` in the config overrides the default
        # automatic grouping of quantitative and qualitative variables.
        if cfg_groups:
            groups = cfg_groups
        factor_results["mfa"] = run_mfa(
            df_active,
            groups,
            optimize=True,
            **params,
        )

    nonlin_results: Dict[str, Any] = {}
    if "umap" in methods:
        logging.info("Running UMAP...")
        params = _method_params("umap", config)
        nonlin_results["umap"] = run_umap(df_active, **params)
    if "phate" in methods:
        logging.info("Running PHATE...")
        params = _method_params("phate", config)
        nonlin_results["phate"] = run_phate(df_active, **params)
    if "pacmap" in methods:
        logging.info("Running PaCMAP...")
        params = _method_params("pacmap", config)
        nonlin_results["pacmap"] = run_pacmap(df_active, **params)

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
        logging.info("Computing metrics...")
        metrics = evaluate_methods(
            all_results,
            df_active,
            quant_vars,
            qual_vars,
            n_clusters=3 if len(df_active) > 3 else 2,
        )
        metrics.to_csv(output_dir / "metrics.csv")
        plot_methods_heatmap(metrics, output_dir)

    logging.info("Generating figures...")
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
    comparison_names: list[str] = []
    if config.get("compare_versions"):
        versions = {k: v for k, v in datasets.items() if k != data_key}
        if versions:
            logging.info("Comparing dataset versions...")
            comparison_names = list(versions.keys())
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
            comparison_metrics.to_csv(
                output_dir / "comparison_metrics.csv", index=False
            )

    robustness_df = None
    if config.get("run_temporal_tests"):
        logging.info("Running temporal stability tests...")
        robustness_df = unsupervised_cv_and_temporal_tests(
            df_active,
            quant_vars,
            qual_vars,
            n_splits=int(config.get("n_splits", 5)),
        )
        pd.DataFrame(robustness_df).to_csv(output_dir / "robustness.csv")

    if config.get("output_pdf"):
        logging.info("Building PDF report...")
        tables: Dict[str, pd.DataFrame] = {"metrics": metrics}
        if comparison_metrics is not None:
            tables["comparison_metrics"] = comparison_metrics
        if robustness_df is not None:
            tables["robustness"] = pd.DataFrame(robustness_df)
        dataset_order = [data_key] + comparison_names
        build_pdf_report(
            output_dir,
            Path(config["output_pdf"]),
            dataset_order,
            tables,
        )

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

    cfg = _load_config(Path(args.config))
    run_pipeline(cfg)


if __name__ == "__main__":  # pragma: no cover - manual execution
    main()
