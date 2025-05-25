#!/usr/bin/env python3
"""Assemble un rapport PDF des fine-tuning pour toutes les méthodes.

Le script parcourt les dossiers ``fine_tune_*`` d'un répertoire donné (par
défaut ``phase4_output``) et rassemble toutes les figures ``.png`` dans un PDF
unique. Les images stockées dans les dossiers ``segment`` ou ``segments`` ne
sont intégrées que pour FAMD et placées en annexe. Le rapport
``mca_fine_tuning_results.pdf`` est ajouté tel quel à la fin du document.
"""

from __future__ import annotations

import argparse
import datetime
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # backend sans affichage
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

try:  # PyPDF2 est optionnel
    from PyPDF2 import PdfMerger
except Exception:  # pragma: no cover
    PdfMerger = None


def generate_fine_tune_pdf(output_dir: Path, pdf_name: str = "fine_tunes_summary.pdf") -> Path:
    """Génère un PDF consolidant les figures des fine-tunes."""
    logger = logging.getLogger(__name__)
    output_dir.mkdir(parents=True, exist_ok=True)
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
        pdf.savefig(fig, dpi=150)
        plt.close(fig)

        for d in fine_dirs:
            method = d.name.replace("fine_tune_", "").upper()

            if method == "MCA":
                # Le rapport MCA sera fusionné ultérieurement
                continue

            fig, ax = plt.subplots(figsize=(8.27, 11.69), dpi=200)
            ax.text(0.5, 0.5, method, fontsize=24, ha="center", va="center")
            ax.axis("off")
            pdf.savefig(fig, dpi=150)
            plt.close(fig)

            for img_path in sorted(d.rglob("*.png")):
                rel = img_path.relative_to(d)
                is_segment = rel.parts and rel.parts[0].lower().startswith("segment")

                if method == "FAMD" and is_segment:
                    famd_segments.append(img_path)
                    continue
                if method in {"MFA", "PCA", "PCAMIX"} and is_segment:
                    continue

                img = plt.imread(img_path)
                fig, ax = plt.subplots(figsize=(12, 6), dpi=200)
                ax.imshow(img)
                ax.axis("off")
                fig.tight_layout()
                fig.text(0.99, 0.01, str(img_path.relative_to(output_dir)), ha="right", va="bottom", fontsize=6, color="gray")
                pdf.savefig(fig, dpi=150)
                plt.close(fig)

        if famd_segments:
            fig, ax = plt.subplots(figsize=(8.27, 11.69), dpi=200)
            ax.text(0.5, 0.5, "Annexe - segments FAMD", fontsize=24, ha="center", va="center")
            ax.axis("off")
            pdf.savefig(fig, dpi=150)
            plt.close(fig)

            for img_path in famd_segments:
                img = plt.imread(img_path)
                fig, ax = plt.subplots(figsize=(12, 6), dpi=200)
                ax.imshow(img)
                ax.axis("off")
                fig.tight_layout()
                fig.text(0.99, 0.01, str(img_path.relative_to(output_dir)), ha="right", va="bottom", fontsize=6, color="gray")
                pdf.savefig(fig, dpi=150)
                plt.close(fig)

    logger.info("PDF intermédiaire généré : %s", pdf_path)

    mca_pdf = output_dir / "fine_tune_mca" / "mca_fine_tuning_results.pdf"
    if PdfMerger and mca_pdf.exists():
        merger = PdfMerger()
        merger.append(str(pdf_path))
        merger.append(str(mca_pdf))
        with open(pdf_path, "wb") as fh:
            merger.write(fh)
        merger.close()
        logger.info("Rapport MCA fusionné dans %s", pdf_path)
    elif not mca_pdf.exists():
        logger.warning("Rapport MCA introuvable: %s", mca_pdf)
    else:
        logger.warning("PyPDF2 non disponible, fusion impossible")

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
