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
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / pdf_name
    fine_dirs = sorted(d for d in output_dir.glob("fine_tune_*") if d.is_dir())

    if not fine_dirs:
        logger.warning("Aucun dossier 'fine_tune_*' trouvé dans %s", output_dir)
        return pdf_path

    with PdfPages(pdf_path) as pdf:
        # Page de garde
        fig, ax = plt.subplots(figsize=(8.27, 11.69), dpi=150)
        today = datetime.datetime.now().strftime("%Y-%m-%d")
        ax.text(0.5, 0.6, "Bilan des fine-tuning", fontsize=20, ha="center", va="center")
        ax.text(0.5, 0.4, f"Généré le {today}", fontsize=12, ha="center", va="center")
        ax.axis("off")
        pdf.savefig(fig, dpi=150)
        plt.close(fig)

        for d in fine_dirs:
            method = d.name.replace("fine_tune_", "").upper()
            logger.info("Traitement de %s", method)

            fig, ax = plt.subplots(figsize=(8.27, 11.69), dpi=150)
            ax.text(0.5, 0.5, method, fontsize=24, ha="center", va="center")
            ax.axis("off")
            pdf.savefig(fig, dpi=150)
            plt.close(fig)

            for img_path in sorted(d.rglob("*.png")):
                logger.debug("Ajout de %s", img_path)
                img = plt.imread(img_path)
                fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
                ax.imshow(img)
                ax.axis("off")
                fig.tight_layout()
                rel = img_path.relative_to(output_dir)
                fig.text(0.99, 0.01, str(rel), ha="right", va="bottom", fontsize=6, color="gray")
                pdf.savefig(fig, dpi=150)
                plt.close(fig)

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
