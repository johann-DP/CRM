"""Aggregated helper functions for the phase 4 pipeline."""

from __future__ import annotations

from dataset_loader import load_datasets
from data_preparation import prepare_data
from dataset_comparison import handle_missing_values, compare_datasets_versions
from factor_methods import run_pca, run_mca, run_famd, run_mfa
from nonlinear_methods import run_umap, run_phate, run_pacmap
from evaluate_methods import evaluate_methods, plot_methods_heatmap
from visualization import generate_figures
from unsupervised_cv import unsupervised_cv_and_temporal_tests
from pdf_report import export_report_to_pdf
from best_params import BEST_PARAMS

__all__ = [
    "load_datasets",
    "prepare_data",
    "handle_missing_values",
    "compare_datasets_versions",
    "run_pca",
    "run_mca",
    "run_famd",
    "run_mfa",
    "run_umap",
    "run_phate",
    "run_pacmap",
    "evaluate_methods",
    "plot_methods_heatmap",
    "generate_figures",
    "unsupervised_cv_and_temporal_tests",
    "export_report_to_pdf",
    "BEST_PARAMS",
]
