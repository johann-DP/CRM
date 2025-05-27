"""Reporting utilities to compile results into a PDF."""

from __future__ import annotations

from pathlib import Path
from datetime import datetime
from typing import Dict, Any

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def export_report_to_pdf(
    figures: Dict[str, plt.Figure],
    tables: Dict[str, Any],
    output_path: str | Path,
) -> None:
    """Save figures and tables into a single PDF file."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    def _df_to_fig(df: pd.DataFrame, title: str) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(8.27, 11.69))
        ax.axis("off")
        ax.set_title(title)
        tbl = ax.table(
            cellText=df.values,
            colLabels=df.columns,
            rowLabels=df.index,
            loc="center",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8)
        return fig

    with PdfPages(out) as pdf:
        fig, ax = plt.subplots(figsize=(8.27, 11.69))
        ax.axis("off")
        ax.text(0.5, 0.6, "Rapport Phase 4", ha="center", va="center", fontsize=16)
        ax.text(0.5, 0.4, datetime.now().strftime("%d/%m/%Y"), ha="center", va="center", fontsize=12)
        pdf.savefig(fig)
        plt.close(fig)

        for name, fig in figures.items():
            if fig is None:
                continue
            fig.suptitle(name.replace("_", " "))
            pdf.savefig(fig)
            plt.close(fig)

        for name, obj in tables.items():
            if isinstance(obj, plt.Figure):
                pdf.savefig(obj)
                plt.close(obj)
            elif isinstance(obj, pd.DataFrame):
                f = _df_to_fig(obj.round(2), name.replace("_", " "))
                pdf.savefig(f)
                plt.close(f)
            else:
                f, ax = plt.subplots(figsize=(8.27, 11.69))
                ax.axis("off")
                ax.set_title(name.replace("_", " "))
                ax.text(0.5, 0.5, str(obj), ha="center", va="center")
                pdf.savefig(f)
                plt.close(f)

        fig, ax = plt.subplots(figsize=(8.27, 11.69))
        ax.axis("off")
        ax.text(0.5, 0.5, "Fin du rapport", ha="center", va="center")
        pdf.savefig(fig)
        plt.close(fig)
