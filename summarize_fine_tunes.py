#!/usr/bin/env python3
"""Assemble un rapport PDF des fine-tuning pour toutes les méthodes.

Le script parcourt les dossiers ``fine_tune_*`` à l'intérieur d'un répertoire
donné (par défaut ``phase4_output``) et agrège toutes les figures ``.png`` dans
un unique PDF. Chaque méthode dispose d'une page de titre suivie de ses images.
"""

from __future__ import annotations

import argparse
import datetime
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

try:  # optional, used to merge external PDFs
    from PyPDF2 import PdfReader, PdfWriter
except Exception:  # pragma: no cover - PyPDF2 may be missing
    PdfReader = None
    PdfWriter = None


def _add_images(pdf: PdfPages, images: list[Path], root: Path) -> None:
    """Append images to a PDF."""
    for img_path in images:
        img = plt.imread(img_path)
        fig, ax = plt.subplots(figsize=(12, 6), dpi=200)
        ax.imshow(img)
        ax.axis("off")
        fig.tight_layout()
        rel = img_path.relative_to(root)
        fig.text(0.99, 0.01, str(rel), ha="right", va="bottom", fontsize=6, color="gray")
        pdf.savefig(fig, dpi=300)
        plt.close(fig)


def _merge_with(pdf_path: Path, extra_pdf: Path) -> None:
    """Append all pages of *extra_pdf* to *pdf_path* using PyPDF2 if available."""
    if PdfReader is None or PdfWriter is None:
        logging.warning("PyPDF2 n'est pas disponible: impossible d'ajouter %s", extra_pdf)
        return
    writer = PdfWriter()
    for path in (pdf_path, extra_pdf):
        reader = PdfReader(str(path))
        for page in reader.pages:
            writer.add_page(page)
    with open(pdf_path, "wb") as f:
        writer.write(f)


def generate_fine_tune_pdf(output_dir: Path, pdf_name: str = "fine_tunes_summary.pdf") -> Path:
    """Génère un PDF rassemblant toutes les figures des fine-tunes."""
    logger = logging.getLogger(__name__)
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / pdf_name
    fine_dirs = sorted(d for d in output_dir.glob("fine_tune_*") if d.is_dir())

    if not fine_dirs:
        logger.warning("Aucun dossier 'fine_tune_*' trouvé dans %s", output_dir)
        return pdf_path

    famd_segments: list[Path] = []
    mca_pdf: Path | None = None

    with PdfPages(pdf_path) as pdf:
        # Page de garde
        fig, ax = plt.subplots(figsize=(8.27, 11.69), dpi=200)
        today = datetime.datetime.now().strftime("%Y-%m-%d")
        ax.text(0.5, 0.6, "Bilan des fine-tuning", fontsize=20, ha="center", va="center")
        ax.text(0.5, 0.4, f"Généré le {today}", fontsize=12, ha="center", va="center")
        ax.axis("off")
        pdf.savefig(fig, dpi=300)
        plt.close(fig)

        for d in fine_dirs:
            method = d.name.replace("fine_tune_", "").upper()

            # Gestion des dossiers spécifiques
            if method == "MCA":
                pdfs = list(d.glob("*.pdf"))
                if pdfs:
                    mca_pdf = pdfs[0]
                    logger.info("PDF MCA trouvé: %s", mca_pdf)
                    continue  # pdf sera fusionné plus tard
            if method == "FAMD":
                seg = d / "segment"
                if seg.exists():
                    famd_segments = sorted(seg.rglob("*.png"))
                    logger.info("Segments FAMD: %d images", len(famd_segments))
            skip_segment = method in {"MFA", "PCA", "PCAMIX"}

            fig, ax = plt.subplots(figsize=(8.27, 11.69), dpi=200)
            ax.text(0.5, 0.5, method, fontsize=24, ha="center", va="center")
            ax.axis("off")
            pdf.savefig(fig, dpi=300)
            plt.close(fig)

            images = []
            for img_path in sorted(d.rglob("*.png")):
                if skip_segment and "segment" in img_path.parts:
                    continue
                if method == "FAMD" and "segment" in img_path.parts:
                    continue
                images.append(img_path)
            _add_images(pdf, images, output_dir)

        if famd_segments:
            fig, ax = plt.subplots(figsize=(8.27, 11.69), dpi=200)
            ax.text(0.5, 0.5, "Annexe - FAMD segment", fontsize=24, ha="center", va="center")
            ax.axis("off")
            pdf.savefig(fig, dpi=300)
            plt.close(fig)

            _add_images(pdf, famd_segments, output_dir)

    if mca_pdf is not None:
        _merge_with(pdf_path, mca_pdf)

    logger.info("PDF généré : %s", pdf_path)
    return pdf_path


def main() -> None:
    p = argparse.ArgumentParser(description="Assemble les résultats de fine-tune en un PDF")
    p.add_argument(
        "--output",
        default="phase4_output",
        help="Répertoire contenant les dossiers fine_tune_*",
    )
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    out = Path(args.output)
    generate_fine_tune_pdf(out)


if __name__ == "__main__":
    main()
