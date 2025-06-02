#!/usr/bin/env python3
"""Generate a base PDF document for the Phase 4 report."""

from __future__ import annotations

import datetime
from pathlib import Path

from fpdf import FPDF

# ASCII-only title to avoid encoding issues with PyFPDF
TITLE = (
    "Analyse Exploratoire CRM - "
    "Phase 4 : Visualisations et Resultats"
)


def create_cover_pdf(
    output_path: str | Path,
    *,
    author: str = "CRM",
    subtitle: str | None = None,
) -> Path:
    """Create a PDF with metadata and a title page."""

    pdf = FPDF()
    pdf.set_title(TITLE)
    pdf.set_author(author)
    pdf.set_subject(datetime.datetime.now().strftime("%Y-%m-%d"))

    pdf.add_page()
    pdf.set_font("Helvetica", "B", 20)
    pdf.cell(0, 60, "", ln=1)
    pdf.multi_cell(0, 10, TITLE, align="C")
    if subtitle:
        pdf.set_font("Helvetica", size=16)
        pdf.cell(0, 10, subtitle, ln=1, align="C")

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    pdf.output(str(out))
    return out


if __name__ == "__main__":  # pragma: no cover - manual use
    create_cover_pdf("phase4_report_base.pdf")
