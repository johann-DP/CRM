"""Utilities to build a consolidated PDF report of phase 4 results."""

from __future__ import annotations

import datetime
import logging
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
    """Create a PDF gathering all figures and tables from phase 4.

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

    with PdfPages(out) as pdf:
        # Title page
        fig, ax = plt.subplots(figsize=(8.27, 11.69), dpi=200)
        today = datetime.datetime.now().strftime("%Y-%m-%d")
        ax.text(
            0.5,
            0.6,
            "Rapport des analyses – Phase 4",
            fontsize=20,
            ha="center",
            va="center",
        )
        ax.text(0.5, 0.4, f"Généré le {today}", fontsize=12, ha="center", va="center")
        ax.axis("off")
        pdf.savefig(fig, dpi=300)
        plt.close(fig)

        # Append figures
        for name, fig in figures.items():
            if fig is None:
                continue
            try:
                fig.suptitle(name, fontsize=12)
                pdf.savefig(fig, dpi=300)
            finally:
                plt.close(fig)

        # Append tables as figures
        for name, table in tables.items():
            if not isinstance(table, pd.DataFrame):
                continue
            fig = _table_to_figure(table, name)
            pdf.savefig(fig, dpi=300)
            plt.close(fig)

    return out
