#!/usr/bin/env python3
"""Assemble un rapport PDF des fine-tuning pour toutes les méthodes.

Le script parcourt les dossiers ``fine_tune_*`` au sein du répertoire
``phase4_output`` (par défaut) et regroupe toutes les figures ``.png`` dans un
unique PDF. Les sous-dossiers ``segments`` sont ignorés puis ajoutés à la fin en
annexe. Pour ``fine_tune_mca`` on reproduit le contenu du rapport PDF généré par
``fine_tuning_mca.py`` en insérant directement ses figures.
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


def _pdf_add_images(pdf: PdfPages, images: list[Path], root: Path) -> None:
    """Append images to ``pdf`` as pages."""

    for img_path in images:
        img = plt.imread(img_path)
        fig, ax = plt.subplots(figsize=(12, 6), dpi=200)
        ax.imshow(img)
        ax.axis("off")
        fig.tight_layout()
        try:
            rel = img_path.relative_to(root)
        except ValueError:
            rel = img_path.name
        fig.text(0.99, 0.01, str(rel), ha="right", va="bottom", fontsize=6, color="gray")
        pdf.savefig(fig, dpi=300)
        plt.close(fig)


def _collect_images(root: Path, ignore_segments: bool = True) -> list[Path]:
    """Return all PNG files under ``root`` sorted lexicographically.

    If ``ignore_segments`` is True, any file located in a directory named
    ``segments`` (case insensitive) is skipped."""

    images: list[Path] = []
    for p in sorted(root.rglob("*.png")):
        if ignore_segments and any(part.lower() == "segments" for part in p.parts):
            continue
        images.append(p)
    return images

def generate_fine_tune_pdf(output_dir: Path, pdf_name: str = "fine_tunes_summary.pdf") -> Path:
    """Génère un PDF rassemblant toutes les figures des fine-tunes."""
    logger = logging.getLogger(__name__)
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / pdf_name
    fine_dirs = sorted(d for d in output_dir.glob("fine_tune_*") if d.is_dir())

    if not fine_dirs:
        logger.warning("Aucun dossier 'fine_tune_*' trouvé dans %s", output_dir)
        return pdf_path

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
            fig, ax = plt.subplots(figsize=(8.27, 11.69), dpi=200)
            ax.text(0.5, 0.5, method, fontsize=24, ha="center", va="center")
            ax.axis("off")
            pdf.savefig(fig, dpi=300)
            plt.close(fig)

            if method == "MCA":
                fig_dir = d / "figures"
                images = _collect_images(fig_dir)
            else:
                images = _collect_images(d)
            _pdf_add_images(pdf, images, output_dir)

        # Annexe segments
        seg_dir = output_dir / "segments"
        seg_images = _collect_images(seg_dir, ignore_segments=False) if seg_dir.exists() else []
        if seg_images:
            fig, ax = plt.subplots(figsize=(8.27, 11.69), dpi=200)
            ax.text(0.5, 0.5, "Annexe : segments", fontsize=24, ha="center", va="center")
            ax.axis("off")
            pdf.savefig(fig, dpi=300)
            plt.close(fig)
            _pdf_add_images(pdf, seg_images, seg_dir)

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
