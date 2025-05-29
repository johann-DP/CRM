"""Utilities to build a consolidated PDF report of phase 4 results."""

from __future__ import annotations

import datetime
import logging
import os
import tempfile
from contextlib import suppress
from pathlib import Path
from typing import Mapping, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd

logger = logging.getLogger(__name__)


def _table_to_figure(df: pd.DataFrame, title: str) -> plt.Figure:
    """Return a Matplotlib figure displaying ``df`` as a table.

    Parameters
    ----------
    df : pandas.DataFrame
        Table to render.
    title : str
        Title of the figure.
    """
    # height grows with number of rows
    fig_height = 0.4 * len(df) + 1.5
    fig, ax = plt.subplots(figsize=(8.0, fig_height), dpi=200)
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


def export_report_to_pdf(
    figures: Mapping[str, Union[plt.Figure, str, Path]],
    tables: Mapping[str, Union[pd.DataFrame, str, Path]],
    output_path: str | Path,
) -> Path:
    """Create a structured PDF gathering all figures and tables from phase 4.

    The function tries to use :mod:`fpdf` for advanced layout. If ``fpdf`` is not
    available, it falls back to :class:`matplotlib.backends.backend_pdf.PdfPages`
    (used in earlier versions).

    Parameters
    ----------
    figures : mapping
        Mapping from figure name to either a Matplotlib :class:`~matplotlib.figure.Figure`
        or a path to an existing image file.
    tables : mapping
        Mapping from table name to a :class:`pandas.DataFrame` or a CSV file path.
    output_path : str or :class:`pathlib.Path`
        Destination path of the PDF file.
        Pages are added in portrait mode by default but switch to landscape when
        a table has many columns.

    Returns
    -------
    pathlib.Path
        Path to the generated PDF.
    """

    if not isinstance(output_path, (str, Path)):
        raise TypeError("output_path must be a path-like object")

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Exporting PDF report to %s", out)

    # Ensure previous figures do not accumulate and trigger warnings
    plt.close("all")

    try:
        from fpdf import FPDF  # type: ignore

        pdf = FPDF(orientation="L", format="A4", unit="mm")
        pdf.set_auto_page_break(auto=True, margin=10)

        def _add_title(text: str, size: int = 14) -> None:
            pdf.set_font("Helvetica", "B", size)
            pdf.cell(0, 10, txt=text, ln=1, align="C")

        # Title page
        pdf.add_page()
        _add_title("Rapport d'analyse Phase 4 – Résultats Dimensionnels", 16)
        pdf.set_font("Helvetica", size=12)
        today = datetime.datetime.now().strftime("%Y-%m-%d")
        pdf.cell(0, 10, f"Généré le {today}", ln=1, align="C")

        # Tables first (comparatif des méthodes, etc.)
        for name, table in tables.items():
            if isinstance(table, (str, Path)):
                try:
                    table = pd.read_csv(table)
                except Exception:
                    continue
            if not isinstance(table, pd.DataFrame):
                continue
            pdf.add_page()
            _add_title(name)
            pdf.set_font("Courier", size=8)
            table_str = table.to_string()
            for line in table_str.splitlines():
                pdf.cell(0, 4, line, ln=1)

        # Figures
        tmp_paths: list[str] = []
        for name, figure in figures.items():
            if figure is None:
                continue
            pdf.add_page()
            _add_title(name)
            if isinstance(figure, (str, Path)):
                img_path = str(figure)
            else:
                tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                figure.savefig(tmp.name, dpi=200, bbox_inches="tight")
                plt.close(figure)
                img_path = tmp.name
                tmp_paths.append(tmp.name)
            pdf.image(img_path, w=180)

        pdf.output(str(out))

        for p in tmp_paths:
            with suppress(OSError):
                os.remove(p)

        plt.close("all")

    except Exception:  # pragma: no cover - fallback when FPDF not installed
        logger.info("FPDF not available, falling back to PdfPages")

        with PdfPages(out) as pdf_backend:
            fig, ax = plt.subplots(figsize=(11.69, 8.27), dpi=200)
            today = datetime.datetime.now().strftime("%Y-%m-%d")
            ax.text(0.5, 0.6, "Rapport des analyses – Phase 4", fontsize=20, ha="center", va="center")
            ax.text(0.5, 0.4, f"Généré le {today}", fontsize=12, ha="center", va="center")
            ax.axis("off")
            pdf_backend.savefig(fig, dpi=300)
            plt.close(fig)

            for name, figure in figures.items():
                if figure is None:
                    continue
                if isinstance(figure, (str, Path)):
                    img = plt.imread(figure)
                    f, ax = plt.subplots()
                    ax.imshow(img)
                    ax.axis("off")
                    f.suptitle(name, fontsize=12)
                    pdf_backend.savefig(f, dpi=300)
                    plt.close(f)
                    continue
                if isinstance(fig, (str, Path)):
                    img = plt.imread(fig)
                    f, ax = plt.subplots()
                    ax.imshow(img)
                    ax.axis("off")
                    f.suptitle(name, fontsize=12)
                    pdf_backend.savefig(f, dpi=300)
                    plt.close(f)
                    continue
                try:
                    figure.suptitle(name, fontsize=12)
                    pdf_backend.savefig(figure, dpi=300)
                finally:
                    plt.close(figure)

            for name, table in tables.items():
                if isinstance(table, (str, Path)):
                    try:
                        table = pd.read_csv(table)
                    except Exception:
                        continue
                if not isinstance(table, pd.DataFrame):
                    continue
                fig = _table_to_figure(table, name)
                pdf_backend.savefig(fig, dpi=300)
                plt.close(fig)

        plt.close("all")

    return out
