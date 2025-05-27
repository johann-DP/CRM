"""Utilities to build a consolidated PDF report of phase 4 results."""

from __future__ import annotations

import datetime
import logging
import os
import tempfile
from pathlib import Path
from typing import Mapping

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
    figures: Mapping[str, plt.Figure],
    tables: Mapping[str, pd.DataFrame],
    output_path: str | Path,
) -> Path:
    """Create a structured PDF gathering all figures and tables from phase 4.

    The function tries to use :mod:`fpdf` for advanced layout. If ``fpdf`` is not
    available, it falls back to :class:`matplotlib.backends.backend_pdf.PdfPages`
    (used in earlier versions).

    Parameters
    ----------
    figures : mapping
        Mapping from figure name to Matplotlib :class:`~matplotlib.figure.Figure`.
    tables : mapping
        Mapping from table name to :class:`pandas.DataFrame`.
    output_path : str or :class:`pathlib.Path`
        Destination path of the PDF file.

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

    try:
        from fpdf import FPDF  # type: ignore

        pdf = FPDF(format="A4", unit="mm")
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
        for name, fig in figures.items():
            if fig is None:
                continue
            pdf.add_page()
            _add_title(name)
            tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            fig.savefig(tmp.name, dpi=200, bbox_inches="tight")
            plt.close(fig)
            pdf.image(tmp.name, w=180)
            tmp_paths.append(tmp.name)

        pdf.output(str(out))

        for p in tmp_paths:
            try:
                os.remove(p)
            except OSError:
                pass

    except Exception:  # pragma: no cover - fallback when FPDF not installed
        logger.info("FPDF not available, falling back to PdfPages")

        with PdfPages(out) as pdf_backend:
            fig, ax = plt.subplots(figsize=(8.27, 11.69), dpi=200)
            today = datetime.datetime.now().strftime("%Y-%m-%d")
            ax.text(0.5, 0.6, "Rapport des analyses – Phase 4", fontsize=20, ha="center", va="center")
            ax.text(0.5, 0.4, f"Généré le {today}", fontsize=12, ha="center", va="center")
            ax.axis("off")
            pdf_backend.savefig(fig, dpi=300)
            plt.close(fig)

            for name, fig in figures.items():
                if fig is None:
                    continue
                try:
                    fig.suptitle(name, fontsize=12)
                    pdf_backend.savefig(fig, dpi=300)
                finally:
                    plt.close(fig)

            for name, table in tables.items():
                if not isinstance(table, pd.DataFrame):
                    continue
                fig = _table_to_figure(table, name)
                pdf_backend.savefig(fig, dpi=300)
                plt.close(fig)

    return out
