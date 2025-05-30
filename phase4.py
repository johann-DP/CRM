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
import datetime
import json
import logging
import os

try:
    from threadpoolctl import threadpool_limits
except Exception:  # pragma: no cover - optional dependency
    threadpool_limits = None

# limite OpenBLAS à 24 threads (ou moins)
os.environ["OPENBLAS_NUM_THREADS"] = "24"
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Callable
import re
import tempfile
from contextlib import suppress

from joblib import Parallel, delayed

import matplotlib.pyplot as plt
import pandas as pd
import yaml
from matplotlib.backends.backend_pdf import PdfPages
from PyPDF2 import PdfMerger

import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings(
    "ignore",
    message="Tight layout not applied.*",
    module="matplotlib",
)
warnings.filterwarnings(
    "ignore",
    message="Workbook contains no default style",
)

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
    plot_general_heatmap,
    generate_figures,
    select_variables,
    unsupervised_cv_and_temporal_tests,
    format_metrics_table,
    BEST_PARAMS,
)


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def _setup_logging(output_dir: Path, level: str = "INFO") -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger()
    logger.setLevel(level)
    # Remove existing handlers to avoid duplicate log lines when running the
    # pipeline multiple times within the same process.
    for h in list(logger.handlers):
        logger.removeHandler(h)
        with suppress(Exception):
            h.close()

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


def set_blas_threads(n_jobs: int = -1) -> int:
    """Set thread count for common BLAS libraries."""
    if n_jobs is None or n_jobs < 1:
        n_jobs = os.cpu_count() or 1
    # OPENBLAS triggers a warning if the requested thread count exceeds the
    # value it was compiled with.  The bundled build in this repository uses a
    # limit of 24 threads.  Cap the environment variable accordingly while
    # leaving the others untouched.
    openblas_threads = min(n_jobs, 24)
    for var in [
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
    ]:
        os.environ[var] = str(n_jobs)
    os.environ["OPENBLAS_NUM_THREADS"] = str(openblas_threads)
    return n_jobs


def build_pdf_report(
    output_dir: Path,
    pdf_path: Path,
    dataset_order: Sequence[str],
    tables: Optional[Mapping[str, pd.DataFrame]] = None,
    df: Optional[pd.DataFrame] = None,
) -> Path:
    """Assemble all PNG figures under ``output_dir`` into ``pdf_path``.

    A title page is added followed by sections for each dataset listed in
    ``dataset_order``. Any provided tables are rendered as figures and appended
    at the end of the document. When ``df`` is provided, two additional pages
    summarising business segments are generated from this dataframe.
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
        elif "clusters_kmeans" in suffix:
            desc = f"Segmentation K-means sur projection {method}"
        elif "clusters_agglomerative" in suffix:
            desc = f"Segmentation agglomerative sur projection {method}"
        elif "clusters_spectral" in suffix:
            desc = f"Segmentation Spectral sur projection {method}"
        elif "clusters_gmm" in suffix:
            desc = f"Segmentation Gaussian mixture sur projection {method}"
        elif "cluster_grid" in suffix or "cluster_comparison" in suffix:
            desc = f"Nuages clusterisés comparatifs – {method}"
        elif "scatter_3d" in suffix:
            desc = f"Nuage 3D – {method}"
        elif "general_heatmap" in name:
            desc = "Heatmap générale des métriques"
        else:
            desc = name
        return f"{dataset} – {desc}"

    def _add_image(pdf: PdfPages, img_path: Path, dataset: str) -> None:
        img = plt.imread(img_path)
        fig, ax = plt.subplots(figsize=(11.69, 8.27), dpi=200)
        ax.imshow(img)
        ax.axis("off")
        ax.text(
            0.5,
            -0.04,
            _format_caption(dataset, img_path.name),
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=8,
            color="gray",
        )
        pdf.savefig(fig)
        plt.close(fig)

    def _table_to_fig(df: pd.DataFrame, title: str) -> plt.Figure:
        height = 0.4 * len(df) + 1.5
        fig_height = min(height, 8.27)
        fig, ax = plt.subplots(figsize=(11.69, fig_height), dpi=200)
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
        fig, ax = plt.subplots(figsize=(11.69, 8.27), dpi=200)
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

        segments_dir = output_dir / "old" / "segments"

        def _first_image(method_dir: Path, pattern: str) -> Path | None:
            files = sorted(method_dir.glob(pattern))
            return files[0] if files else None

        def _add_raw_scatter(method_dir: Path, dataset: str) -> None:
            img2d = _first_image(method_dir, "*scatter_2d*.png")
            if img2d is None:
                return
            img3d = _first_image(method_dir, "*scatter_3d*.png")
            if img3d is None:
                fig, ax = plt.subplots(figsize=(11, 8.5), dpi=200)
                ax.imshow(plt.imread(img2d))
                ax.axis("off")
                fig.suptitle(
                    f"{dataset} – {method_dir.name.upper()} – Nuages de points bruts",
                    fontsize=12,
                )
            else:
                fig, axes = plt.subplots(1, 2, figsize=(11, 8.5), dpi=200)
                for ax, img in zip(axes, [img2d, img3d]):
                    ax.imshow(plt.imread(img))
                    ax.axis("off")
                fig.suptitle(
                    f"{dataset} – {method_dir.name.upper()} – Nuages de points bruts",
                    fontsize=12,
                )
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

        def _add_cluster_page(method_dir: Path, dataset: str) -> None:
            grid = _first_image(method_dir, "*cluster_grid*.png")
            if grid is None:
                grid = _first_image(method_dir, "*cluster_comparison*.png")
            if grid is None:
                pats = [
                    "*clusters_kmeans*.png",
                    "*clusters_agglomerative*.png",
                    "*clusters_spectral*.png",
                    "*clusters_gmm*.png",
                ]
                imgs = [_first_image(method_dir, pat) for pat in pats]
                if not any(imgs):
                    return
                fig, axes = plt.subplots(2, 2, figsize=(11, 8.5), dpi=200)
                titles = ["K-means", "Agglomerative", "Spectral", "Gaussian Mixture"]
                for ax, img, title in zip(axes.ravel(), imgs, titles):
                    if img is not None and Path(img).exists():
                        ax.imshow(plt.imread(img))
                        ax.set_title(title, fontsize=9)
                    ax.axis("off")
                fig.suptitle(
                    f"{dataset} – {method_dir.name.upper()} – Nuages clusterisés",
                    fontsize=12,
                )
            else:
                fig, ax = plt.subplots(figsize=(11, 8.5), dpi=200)
                ax.imshow(plt.imread(grid))
                ax.axis("off")
                fig.suptitle(
                    f"{dataset} – {method_dir.name.upper()} – Nuages clusterisés",
                    fontsize=12,
                )
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

        def _add_analysis_page(method_dir: Path, dataset: str) -> None:
            img = _first_image(method_dir, "*analysis_summary*.png")
            if img is None:
                return
            data = plt.imread(img)
            fig, ax = plt.subplots(figsize=(11, 8.5), dpi=200)
            ax.imshow(data)
            ax.axis("off")
            fig.suptitle(
                f"{dataset} – {method_dir.name.upper()} – Analyse détaillée",
                fontsize=12,
            )
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

        for name in dataset_order:
            # Section page
            fig, ax = plt.subplots(figsize=(11.69, 8.27), dpi=200)
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

            method_dirs = sorted(p for p in base_dir.iterdir() if p.is_dir())
            for method_dir in method_dirs:
                _add_raw_scatter(method_dir, name)
                _add_cluster_page(method_dir, name)
                _add_analysis_page(method_dir, name)

        heatmap_path = output_dir / "methods_heatmap.png"
        if heatmap_path.exists():
            _add_image(pdf, heatmap_path, dataset_order[0])

        general_heatmap = output_dir / "general_heatmap.png"
        if general_heatmap.exists():
            _add_image(pdf, general_heatmap, "Synthèse")

        # Segment summary pages -------------------------------------------------
        def _add_segment_page(path: Path, title: str) -> None:
            if path.exists():
                img = plt.imread(path)
                fig, ax = plt.subplots(figsize=(11.69, 8.27), dpi=200)
                ax.imshow(img)
                ax.axis("off")
                ax.set_title(title, fontsize=12)
                fig.tight_layout()
            else:
                fig, ax = plt.subplots(figsize=(11.69, 8.27), dpi=200)
                ax.axis("off")
                ax.text(
                    0.5,
                    0.5,
                    f"{title} (donn\xe9es manquantes)",
                    ha="center",
                    va="center",
                    fontsize=12,
                )
            pdf.savefig(fig)
            plt.close(fig)

        seg1 = output_dir / "segment_summary_1.png"
        seg2 = output_dir / "segment_summary_2.png"
        _add_segment_page(seg1, "Synth\xe8se segmentaire 1")
        _add_segment_page(seg2, "Synth\xe8se segmentaire 2")

        def _segment_analysis_pages(
            data: Optional[pd.DataFrame],
        ) -> tuple[plt.Figure, plt.Figure]:
            seg_cols1 = [
                "Cat\xe9gorie",
                "Sous-cat\xe9gorie",
                "Entit\xe9 op\xe9rationnelle",
                "Statut commercial",
            ]
            seg_cols2 = ["Pilier", "Statut production", "Type opportunit\xe9"]
            all_cols = seg_cols1 + seg_cols2

            def _plot(ax: plt.Axes, series: pd.Series, name: str) -> None:
                counts = series.astype(str).value_counts(dropna=False)
                ax.bar(counts.index.astype(str), counts.values, edgecolor="black")
                ax.set_xlabel(name)
                ax.set_ylabel("Effectif")
                for label in ax.get_xticklabels():
                    label.set_rotation(45)
                    label.set_ha("right")

            fig1, axes1 = plt.subplots(2, 2, figsize=(11.69, 8.27), dpi=200)
            for ax, col in zip(axes1.ravel(), seg_cols1):
                if data is not None and col in data.columns:
                    _plot(ax, data[col], col)
                else:
                    ax.axis("off")
                    ax.text(
                        0.5,
                        0.5,
                        "Donn\xe9es manquantes",
                        ha="center",
                        va="center",
                        fontsize=8,
                    )
            fig1.tight_layout()

            fig2, axes2 = plt.subplots(2, 2, figsize=(11.69, 8.27), dpi=200)
            for ax, col in zip(axes2.ravel()[:3], seg_cols2):
                if data is not None and col in data.columns:
                    _plot(ax, data[col], col)
                else:
                    ax.axis("off")
                    ax.text(
                        0.5,
                        0.5,
                        "Donn\xe9es manquantes",
                        ha="center",
                        va="center",
                        fontsize=8,
                    )

            ax = axes2.ravel()[3]
            if data is not None:
                avail = [c for c in all_cols if c in data.columns]
                if avail:
                    pct = data[avail].isna().mean().mul(100)
                    ax.bar(pct.index.astype(str), pct.values, edgecolor="black")
                    ax.set_ylabel("% NA")
                    for label in ax.get_xticklabels():
                        label.set_rotation(45)
                        label.set_ha("right")
                    ax.set_xlabel("Segment")
                else:
                    ax.axis("off")
                    ax.text(
                        0.5,
                        0.5,
                        "Donn\xe9es manquantes",
                        ha="center",
                        va="center",
                        fontsize=8,
                    )
            else:
                ax.axis("off")
                ax.text(
                    0.5,
                    0.5,
                    "Donn\xe9es manquantes",
                    ha="center",
                    va="center",
                    fontsize=8,
                )

            fig2.tight_layout()
            return fig1, fig2

        fig_a, fig_b = _segment_analysis_pages(df)
        pdf.savefig(fig_a)
        plt.close(fig_a)
        pdf.savefig(fig_b)
        plt.close(fig_b)

        if tables:
            for tname, df in tables.items():
                fig = _table_to_fig(df, tname)
                pdf.savefig(fig)
                plt.close(fig)

        if segments_dir.exists():
            fig, ax = plt.subplots(figsize=(11.69, 8.27), dpi=200)
            ax.axis("off")
            ax.text(
                0.5,
                0.9,
                "Annexe – Comptage des segments",
                ha="center",
                va="top",
                fontsize=14,
                weight="bold",
            )
            pdf.savefig(fig)
            plt.close(fig)

            for img in sorted(segments_dir.glob("*.png")):
                _add_image(pdf, img, "Annexe")

    return pdf_path


def build_type_report(base_dir: Path, pdf_path: Path, datasets: Sequence[str]) -> Path:
    """Assemble figures by type in landscape orientation.

    Figures are grouped in the following order: scatter plots, correlation
    circles, clustering results and validation metrics.  Within each group the
    images are ordered by ``datasets``. Up to four figures are arranged on each
    page using a 2x2 grid.  An annex is appended with the segment figures,
    cluster–segment heatmaps and clustering validation curves (silhouette, Dunn
    or stability charts).
    """

    pdf_path.parent.mkdir(parents=True, exist_ok=True)

    page_size = (11.69, 8.27)  # A4 landscape

    def _grid_page(pdf: PdfPages, images: list[Path], title: str) -> None:
        for i in range(0, len(images), 4):
            fig, axes = plt.subplots(2, 2, figsize=page_size, dpi=200)
            for ax, img in zip(axes.ravel(), images[i : i + 4]):
                if img.exists():
                    ax.imshow(plt.imread(img))
                    ax.set_title(img.stem, fontsize=8)
                ax.axis("off")
            for ax in axes.ravel()[len(images[i : i + 4]) :]:
                ax.axis("off")
            fig.suptitle(title, fontsize=12)
            fig.tight_layout(rect=(0, 0.03, 1, 0.95))
            pdf.savefig(fig)
            plt.close(fig)

    categories: list[tuple[str, Callable[[str], bool]]] = [
        ("Nuages de points", lambda n: "scatter_2d" in n and "cluster" not in n),
        ("Cercles de corrélation", lambda n: "correlation" in n),
        (
            "Résultats clustering",
            lambda n: "cluster" in n
            and "segments" not in n
            and "silhouette" not in n
            and "dunn" not in n,
        ),
        (
            "Métriques de validation",
            lambda n: any(k in n for k in ["scree", "contrib", "robustness"]),
        ),
    ]

    annex_images: dict[str, list[Path]] = {
        "segments": [],
        "heatmaps": [],
        "cluster_validation": [],
    }

    figures: dict[str, dict[str, list[Path]]] = {
        key: {ds: [] for ds in datasets} for key, _ in categories
    }

    for ds in datasets:
        root = base_dir / ds
        if not root.exists():
            continue
        for img in sorted(root.rglob("*.png")):
            name = img.name.lower()
            if "cluster_segments" in name or name.endswith("_segments.png"):
                annex_images["heatmaps"].append(img)
                continue
            if any(k in name for k in ["silhouette", "dunn", "stability"]):
                annex_images["cluster_validation"].append(img)
                continue
            for title, cond in categories:
                if cond(name):
                    figures[title][ds].append(img)
                    break

    segments_dir = base_dir / "old" / "segments"
    if segments_dir.exists():
        annex_images["segments"] = sorted(segments_dir.glob("*.png"))

    with PdfPages(pdf_path) as pdf:
        fig, ax = plt.subplots(figsize=page_size, dpi=200)
        ax.axis("off")
        ax.text(
            0.5,
            0.5,
            "Phase 4 Report",
            ha="center",
            va="center",
            fontsize=16,
            weight="bold",
        )
        pdf.savefig(fig)
        plt.close(fig)

        for title, _ in categories:
            fig, ax = plt.subplots(figsize=page_size, dpi=200)
            ax.axis("off")
            ax.text(0.5, 0.9, title, ha="center", va="top", fontsize=14, weight="bold")
            pdf.savefig(fig)
            plt.close(fig)
            for ds in datasets:
                imgs = figures[title][ds]
                if not imgs:
                    continue
                _grid_page(pdf, imgs, f"{title} – {ds}")

        extra_imgs = []
        gen_heatmap = base_dir / "general_heatmap.png"
        if gen_heatmap.exists():
            extra_imgs.append((gen_heatmap, "Synthèse générale"))

        miss_fig = base_dir / datasets[0] / "segment_summary_2.png"
        if miss_fig.exists():
            extra_imgs.append((miss_fig, "% NA par segment"))

        if any(v for v in annex_images.values()) or extra_imgs:
            fig, ax = plt.subplots(figsize=page_size, dpi=200)
            ax.axis("off")
            ax.text(
                0.5, 0.9, "Annexe", ha="center", va="top", fontsize=14, weight="bold"
            )
            pdf.savefig(fig)
            plt.close(fig)

            for img, title in extra_imgs:
                _grid_page(pdf, [img], title)

            if annex_images["segments"]:
                _grid_page(pdf, annex_images["segments"], "Segments")
            if annex_images["heatmaps"]:
                _grid_page(pdf, annex_images["heatmaps"], "Clusters vs Segments")
            if annex_images["cluster_validation"]:
                _grid_page(
                    pdf, annex_images["cluster_validation"], "Validation clustering"
                )

    return pdf_path


# ---------------------------------------------------------------------------
# PDF concatenation helpers
# ---------------------------------------------------------------------------


def _derive_seg_titles(filename: str) -> tuple[str, str]:
    """Return a title and caption for a segment figure filename."""
    base = Path(filename).stem
    m = re.match(r"(.*)_cluster_(\d+)$", base)
    if m:
        prefix, cluster = m.groups()
        title = f"{prefix.replace('_', ' ').title()} – Cluster {cluster}"
        caption = f"Distribution of segment assignments for cluster {cluster}"
    else:
        title = base.replace("_", " ").title()
        caption = title
    return title, caption


def _images_to_pdf(
    images: Sequence[Path], pdf_path: Path, title: str | None = None
) -> Path:
    """Save ``images`` into ``pdf_path`` with optional title page."""
    pdf_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        from fpdf import FPDF  # type: ignore

        pdf = FPDF(orientation="L", format="A4", unit="mm")
        pdf.set_auto_page_break(auto=False)
        if title:
            pdf.add_page()
            pdf.set_font("Helvetica", "B", 16)
            pdf.cell(0, 10, txt=title, ln=1, align="C")
        for img in images:
            if not img.exists():
                continue
            page_title, caption = _derive_seg_titles(img.name)
            pdf.add_page()
            pdf.set_font("Helvetica", "B", 14)
            pdf.cell(0, 10, txt=page_title, ln=1, align="C")
            pdf.image(str(img), x=15, w=180)
            pdf.set_y(-20)
            pdf.set_font("Helvetica", size=10)
            pdf.cell(0, 10, txt=caption, ln=1, align="C")
        pdf.output(str(pdf_path))

    except Exception:
        with PdfPages(pdf_path) as pdf:
            if title:
                fig, ax = plt.subplots(figsize=(11.69, 8.27), dpi=200)
                ax.axis("off")
                ax.text(
                    0.5, 0.9, title, ha="center", va="top", fontsize=14, weight="bold"
                )
                pdf.savefig(fig)
                plt.close(fig)
            for img in images:
                if not img.exists():
                    continue
                page_title, caption = _derive_seg_titles(img.name)
                data = plt.imread(img)
                fig, ax = plt.subplots(figsize=(11.69, 8.27), dpi=200)
                ax.imshow(data)
                ax.axis("off")
                ax.set_title(page_title)
                ax.text(
                    0.5,
                    -0.05,
                    caption,
                    transform=ax.transAxes,
                    ha="center",
                    va="top",
                    fontsize=10,
                    color="gray",
                )
                pdf.savefig(fig)
                plt.close(fig)

    return pdf_path


def concat_pdf_reports(output_dir: Path, output_pdf: Path) -> Path:
    """Concat predefined reports and append a segments annex."""

    pdf_order = [
        output_dir / "phase4_report_raw.pdf",
        output_dir / "phase4_report_cleaned_1.pdf",
        output_dir / "phase4_report_cleaned_3_univ.pdf",
        output_dir / "phase4_report_cleaned_3_multi.pdf",
    ]

    merger = PdfMerger()
    for path in pdf_order:
        if path.exists():
            merger.append(str(path))

    general_img = output_dir / "general_heatmap.png"
    tmp_gen = None
    if general_img.exists():
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp_gen = Path(tmp.name)
        _images_to_pdf([general_img], tmp_gen, "Synthèse générale des métriques")
        merger.append(str(tmp_gen))

    segments_dir = output_dir / "old" / "segments"
    tmp_pdf = None
    if segments_dir.exists():
        images = sorted(segments_dir.glob("*.png"))
        if images:
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp_pdf = Path(tmp.name)
            _images_to_pdf(
                images, tmp_pdf, "Appendix – Business Segment Visualizations"
            )
            merger.append(str(tmp_pdf))

    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    with open(output_pdf, "wb") as fh:
        merger.write(fh)
    merger.close()

    if tmp_pdf is not None:
        with suppress(OSError):
            os.remove(tmp_pdf)
    if tmp_gen is not None:
        with suppress(OSError):
            os.remove(tmp_gen)

    return output_pdf


def save_segment_analysis_figures(df: pd.DataFrame, output_dir: Path) -> None:
    """Generate and save segment summary figures used in the final report."""

    seg_cols1 = [
        "Catégorie",
        "Sous-catégorie",
        "Entité opérationnelle",
        "Statut commercial",
    ]
    seg_cols2 = ["Pilier", "Statut production", "Type opportunité"]
    all_cols = seg_cols1 + seg_cols2

    def _plot(ax: plt.Axes, series: pd.Series, name: str) -> None:
        counts = series.astype(str).value_counts(dropna=False)
        ax.bar(counts.index.astype(str), counts.values, edgecolor="black")
        ax.set_xlabel(name)
        ax.set_ylabel("Effectif")
        for label in ax.get_xticklabels():
            label.set_rotation(45)
            label.set_ha("right")

    fig1, axes1 = plt.subplots(2, 2, figsize=(11.69, 8.27), dpi=200)
    for ax, col in zip(axes1.ravel(), seg_cols1):
        if col in df.columns:
            _plot(ax, df[col], col)
        else:
            ax.axis("off")
            ax.text(0.5, 0.5, "Données manquantes", ha="center", va="center", fontsize=8)
    fig1.tight_layout()
    path1 = output_dir / "segment_summary_1.png"
    fig1.savefig(path1)
    plt.close(fig1)

    fig2, axes2 = plt.subplots(2, 2, figsize=(11.69, 8.27), dpi=200)
    for ax, col in zip(axes2.ravel()[:3], seg_cols2):
        if col in df.columns:
            _plot(ax, df[col], col)
        else:
            ax.axis("off")
            ax.text(0.5, 0.5, "Données manquantes", ha="center", va="center", fontsize=8)

    ax = axes2.ravel()[3]
    avail = [c for c in all_cols if c in df.columns]
    if avail:
        pct = df[avail].isna().mean().mul(100)
        ax.bar(pct.index.astype(str), pct.values, edgecolor="black")
        ax.set_ylabel("% NA")
        for label in ax.get_xticklabels():
            label.set_rotation(45)
            label.set_ha("right")
        ax.set_xlabel("Segment")
    else:
        ax.axis("off")
        ax.text(0.5, 0.5, "Données manquantes", ha="center", va="center", fontsize=8)

    fig2.tight_layout()
    path2 = output_dir / "segment_summary_2.png"
    fig2.savefig(path2)
    plt.close(fig2)

    return None


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def run_pipeline(config: Dict[str, Any]) -> Dict[str, Any]:
    output_dir = Path(config.get("output_dir", "phase4_output"))
    _setup_logging(output_dir)
    n_jobs = int(config.get("n_jobs", -1))
    set_blas_threads(n_jobs)
    optimize = bool(config.get("optimize_params", False))

    logging.info("Loading datasets...")
    datasets = load_datasets(
        config, ignore_schema=bool(config.get("ignore_schema", False))
    )
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

    def _run_method(name: str, func, args: tuple, kwargs: dict) -> tuple[str, Any]:
        logging.info("Running %s...", name.upper())
        try:
            return name, func(*args, **kwargs)
        except Exception as exc:  # pragma: no cover - runtime failure
            logging.warning("%s failed: %s", name.upper(), exc)
            return name, {}

    tasks: list[tuple[str, Any, tuple, dict]] = []
    factor_names: list[str] = []
    nonlin_names: list[str] = []

    if "pca" in methods and quant_vars:
        params = _method_params("pca", config)
        if optimize:
            params.pop("n_components", None)
        tasks.append(
            ("pca", run_pca, (df_active, quant_vars), dict(optimize=optimize, **params))
        )
        factor_names.append("pca")

    if "mca" in methods and qual_vars:
        params = _method_params("mca", config)
        if optimize:
            params.pop("n_components", None)
        tasks.append(
            ("mca", run_mca, (df_active, qual_vars), dict(optimize=optimize, **params))
        )
        factor_names.append("mca")

    if "famd" in methods and quant_vars and qual_vars:
        params = _method_params("famd", config)
        if optimize:
            params.pop("n_components", None)
        tasks.append(
            (
                "famd",
                run_famd,
                (df_active, quant_vars, qual_vars),
                dict(optimize=optimize, **params),
            )
        )
        factor_names.append("famd")

    groups = []
    if quant_vars:
        groups.append(quant_vars)
    if qual_vars:
        groups.append(qual_vars)
    if "mfa" in methods and len(groups) > 1:
        params = _method_params("mfa", config)
        if optimize:
            params.pop("n_components", None)
        cfg_groups = params.pop("groups", None)
        if cfg_groups:
            groups = cfg_groups
        tasks.append(
            ("mfa", run_mfa, (df_active, groups), dict(optimize=optimize, **params))
        )
        factor_names.append("mfa")

    if "umap" in methods:
        params = _method_params("umap", config)
        tasks.append(("umap", run_umap, (df_active,), params))
        nonlin_names.append("umap")
    if "phate" in methods:
        params = _method_params("phate", config)
        tasks.append(("phate", run_phate, (df_active,), params))
        nonlin_names.append("phate")
    if "pacmap" in methods:
        params = _method_params("pacmap", config)
        tasks.append(("pacmap", run_pacmap, (df_active,), params))
        nonlin_names.append("pacmap")

    backend = config.get("joblib_backend", "loky")
    if len(tasks) <= 1 or (n_jobs is not None and n_jobs == 1):
        results = [
            _run_method(name, func, args, kwargs) for name, func, args, kwargs in tasks
        ]
    else:
        with Parallel(n_jobs=n_jobs or len(tasks), backend=backend) as parallel:
            results = parallel(
                delayed(_run_method)(name, func, args, kwargs)
                for name, func, args, kwargs in tasks
            )

    factor_results: Dict[str, Any] = {}
    nonlin_results: Dict[str, Any] = {}
    for name, res in results:
        if name in factor_names:
            factor_results[name] = res
        elif name in nonlin_names:
            nonlin_results[name] = res

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
        k_max = min(10, max(2, len(df_active) - 1))
        metrics = evaluate_methods(
            all_results,
            df_active,
            quant_vars,
            qual_vars,
            k_range=range(2, k_max + 1),
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
        segment_col=config.get("segment_col"),
        n_jobs=n_jobs,
        backend=backend,
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
                segment_col=config.get("segment_col"),
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

    # Save segment summary figures for later report assembly
    save_segment_analysis_figures(df_active, output_dir)

    logging.info("Analysis complete")
    logging.shutdown()
    return {
        "metrics": metrics,
        "figures": figures,
        "comparison_metrics": comparison_metrics,
        "robustness": robustness_df,
    }


def _run_pipeline_single(
    config: Dict[str, Any], name: str
) -> tuple[str, Dict[str, Any]]:
    """Helper for :func:`run_pipeline_parallel` executing a single dataset."""

    cfg = dict(config)
    cfg["dataset"] = name
    if "output_dir" in cfg:
        base = Path(cfg["output_dir"])
        cfg["output_dir"] = str(base / name)
    if "output_pdf" in cfg:
        pdf = Path(cfg["output_pdf"])
        cfg["output_pdf"] = str(pdf.with_name(f"{pdf.stem}_{name}{pdf.suffix}"))
    result = run_pipeline(cfg)
    # Avoid pickling large matplotlib objects in parallel mode
    if isinstance(result, dict) and "figures" in result:
        result.pop("figures", None)
    return name, result


def run_pipeline_parallel(
    config: Dict[str, Any],
    datasets: Sequence[str],
    *,
    n_jobs: Optional[int] = None,
    backend: str = "multiprocessing",
) -> Dict[str, Dict[str, Any]]:
    """Run :func:`run_pipeline` on several datasets in parallel."""
    from phase4_parallel import _run_pipeline_single

    n_jobs = n_jobs or len(datasets)
    with Parallel(n_jobs=n_jobs, backend=backend) as parallel:
        results = parallel(
            delayed(_run_pipeline_single)(config, ds) for ds in datasets
        )
    results = dict(results)

    metrics_frames = []
    for name, res in results.items():
        df_m = res.get("metrics")
        if isinstance(df_m, pd.DataFrame) and not df_m.empty:
            df_m = df_m.reset_index().rename(columns={"index": "method"})
            df_m["dataset"] = name
            metrics_frames.append(df_m)

    if metrics_frames:
        all_metrics = pd.concat(metrics_frames, ignore_index=True)
        plot_general_heatmap(
            all_metrics, Path(config.get("output_dir", "phase4_output"))
        )

    if "output_pdf" in config:
        base_dir = Path(config.get("output_dir", "phase4_output"))
        pdf = Path(config["output_pdf"])
        build_type_report(base_dir, pdf, datasets)

    logging.shutdown()
    return results


# ---------------------------------------------------------------------------
# CLI Entrypoint
# ---------------------------------------------------------------------------


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Phase 4 analysis (modular)")
    parser.add_argument("--config", required=True, help="Path to config YAML/JSON")
    parser.add_argument(
        "--datasets",
        nargs="+",
        help="Datasets to process in parallel (e.g. raw cleaned_1)",
    )
    parser.add_argument(
        "--dataset-jobs",
        type=int,
        default=None,
        help="Number of workers for dataset-level parallelism",
    )
    parser.add_argument(
        "--dataset-backend",
        default="multiprocessing",
        help="joblib backend for dataset parallelism",
    )
    args = parser.parse_args(argv)

    cfg = _load_config(Path(args.config))
    if args.datasets:
        run_pipeline_parallel(
            cfg,
            args.datasets,
            n_jobs=args.dataset_jobs,
            backend=args.dataset_backend,
        )
    else:
        run_pipeline(cfg)


if __name__ == "__main__":  # pragma: no cover - manual execution
    main()
