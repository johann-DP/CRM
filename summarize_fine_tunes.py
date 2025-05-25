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


def generate_fine_tune_pdf(output_dir: Path, pdf_name: str = "fine_tunes_summary.pdf") -> Path:
    """Génère un PDF rassemblant toutes les figures des fine-tunes."""
    logger = logging.getLogger(__name__)
    pdf_path = output_dir / pdf_name
    fine_dirs = sorted(d for d in output_dir.glob("fine_tune_*") if d.is_dir())

    if not fine_dirs:
        logger.warning("Aucun dossier 'fine_tune_*' trouvé dans %s", output_dir)
        return pdf_path

    famd_segments: list[Path] = []

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
            method = d.name.replace("fine_tune_", "").lower()
            title = method.upper()

            fig, ax = plt.subplots(figsize=(8.27, 11.69), dpi=200)
            ax.text(0.5, 0.5, title, fontsize=24, ha="center", va="center")
            ax.axis("off")
            pdf.savefig(fig, dpi=300)
            plt.close(fig)

            img_paths: list[Path]
            if method == "famd":
                # keep segmentation images for the annex
                seg_dir = d / "segments"
                if seg_dir.exists():
                    famd_segments.extend(sorted(seg_dir.rglob("*.png")))
                img_paths = [
                    p for p in sorted(d.rglob("*.png"))
                    if seg_dir not in p.parents
                ]
            elif method in {"mfa", "pca", "pcamix"}:
                img_paths = [
                    p for p in sorted(d.rglob("*.png"))
                    if not any(part.lower() == "segments" for part in p.parts)
                ]
            else:
                img_paths = sorted(d.rglob("*.png"))

            for img_path in img_paths:
                img = plt.imread(img_path)
                fig, ax = plt.subplots(figsize=(12, 6), dpi=200)
                ax.imshow(img)
                ax.axis("off")
                fig.tight_layout()
                rel = img_path.relative_to(output_dir)
                fig.text(0.99, 0.01, str(rel), ha="right", va="bottom", fontsize=6, color="gray")
                pdf.savefig(fig, dpi=300)
                plt.close(fig)

        # Append FAMD segmentation images at the end
        if famd_segments:
            fig, ax = plt.subplots(figsize=(8.27, 11.69), dpi=200)
            ax.text(0.5, 0.5, "ANNEXE SEGMENTS FAMD", fontsize=18, ha="center", va="center")
            ax.axis("off")
            pdf.savefig(fig, dpi=300)
            plt.close(fig)

            for img_path in famd_segments:
                img = plt.imread(img_path)
                fig, ax = plt.subplots(figsize=(12, 6), dpi=200)
                ax.imshow(img)
                ax.axis("off")
                fig.tight_layout()
                rel = img_path.relative_to(output_dir)
                fig.text(0.99, 0.01, str(rel), ha="right", va="bottom", fontsize=6, color="gray")
                pdf.savefig(fig, dpi=300)
                plt.close(fig)

    logger.info("PDF généré : %s", pdf_path)
    return pdf_path


def main() -> None:
    p = argparse.ArgumentParser(description="Assemble les résultats de fine-tune en un PDF")
    p.add_argument(
        "--output",
        default=r"D:\DATAPREDICT\DATAPREDICT 2024\Missions\Digora\phase4_output",
        help="Répertoire contenant les dossiers fine_tune_*",
    )
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    out = Path(args.output)
    generate_fine_tune_pdf(out)


if __name__ == "__main__":
    main()