#!/usr/bin/env python3
"""Assemble un rapport PDF des fine-tuning pour toutes les méthodes.

Ce script parcourt les dossiers ``fine_tune_*`` d'un répertoire (``phase4_output``
par défaut) et rassemble les figures ``.png`` dans un unique PDF. Les dossiers
``segment`` de ``fine_tune_famd`` sont placés en annexe tandis que ceux de MFA,
PCA et PCAmix sont ignorés. Le rapport MCA existant est inséré tel quel à la fin
si `PyPDF2` est disponible.
"""

from __future__ import annotations

import argparse
import datetime
import logging
from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

try:
    from PyPDF2 import PdfMerger
except Exception:  # pragma: no cover - optional dependency
    PdfMerger = None


def _add_images(pdf: PdfPages, title: str, images: Iterable[Path], base_dir: Path) -> None:
    """Add a title page then each image to ``pdf``."""
    fig, ax = plt.subplots(figsize=(8.27, 11.69), dpi=150)
    ax.text(0.5, 0.5, title, fontsize=24, ha="center", va="center")
    ax.axis("off")
    pdf.savefig(fig)
    plt.close(fig)

    for img_path in images:
        img = plt.imread(img_path)
        fig, ax = plt.subplots(figsize=(12, 6), dpi=150)
        ax.imshow(img)
        ax.axis("off")
        fig.tight_layout()
        rel = img_path.relative_to(base_dir)
        fig.text(0.99, 0.01, str(rel), ha="right", va="bottom", fontsize=6, color="gray")
        pdf.savefig(fig)
        plt.close(fig)


def generate_fine_tune_pdf(output_dir: Path, pdf_name: str = "fine_tunes_summary.pdf") -> Path:
    """Génère un PDF rassemblant toutes les figures des fine-tunes."""
    logger = logging.getLogger(__name__)
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / pdf_name
    fine_dirs = sorted(d for d in output_dir.glob("fine_tune_*") if d.is_dir())

    if not fine_dirs:
        logger.warning("Aucun dossier 'fine_tune_*' trouvé dans %s", output_dir)
        return pdf_path

    annex_imgs: list[Path] = []
    mca_pdf: Path | None = None

    with PdfPages(pdf_path) as pdf:
        # Page de garde
        fig, ax = plt.subplots(figsize=(8.27, 11.69), dpi=150)
        today = datetime.datetime.now().strftime("%Y-%m-%d")
        ax.text(0.5, 0.6, "Bilan des fine-tuning", fontsize=20, ha="center", va="center")
        ax.text(0.5, 0.4, f"Généré le {today}", fontsize=12, ha="center", va="center")
        ax.axis("off")
        pdf.savefig(fig)
        plt.close(fig)

        for d in fine_dirs:
            method = d.name.replace("fine_tune_", "").upper()
            logger.info("Traitement de %s", method)

            if method == "MCA":
                mca_pdf = d / "mca_fine_tuning_results.pdf"
                if not mca_pdf.exists():
                    imgs = [p for p in d.rglob("*.png") if "segment" not in p.parts]
                    _add_images(pdf, method, sorted(imgs), output_dir)
                continue

            imgs = [p for p in d.rglob("*.png") if "segment" not in p.parts]
            if method == "FAMD":
                seg_dir = d / "segment"
                if seg_dir.exists():
                    annex_imgs.extend(sorted(seg_dir.rglob("*.png")))

            if imgs:
                _add_images(pdf, method, sorted(imgs), output_dir)

        if annex_imgs:
            logger.info("Ajout de l'annexe segmentation (%d figures)", len(annex_imgs))
            _add_images(pdf, "ANNEXE - SEGMENTATION", annex_imgs, output_dir)

    if mca_pdf and mca_pdf.exists() and PdfMerger is not None:
        final_pdf = output_dir / pdf_name
        merger = PdfMerger()
        merger.append(str(pdf_path))
        merger.append(str(mca_pdf))
        merger.write(str(final_pdf))
        merger.close()
        pdf_path.unlink(missing_ok=True)
        pdf_path = final_pdf
        logger.info("PDF final avec rapport MCA inséré : %s", pdf_path)
    elif mca_pdf and mca_pdf.exists():
        logger.warning("PyPDF2 introuvable : le rapport MCA (%s) n'a pas été fusionné", mca_pdf)

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
