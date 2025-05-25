#!/usr/bin/env python3
"""Assemble un rapport PDF des fine-tuning pour toutes les méthodes.

Le script parcourt les dossiers ``fine_tune_*`` dans ``phase4_output`` (ou le
répertoire passé en argument) et rassemble toutes les figures ``.png`` dans un
fichier ``fine_tunes_summary.pdf``. Les images situées dans un dossier
``segments`` sont ignorées, sauf pour FAMD où elles sont regroupées dans une
annexe en fin de document. Lorsque ``fine_tune_mca`` contient un fichier
``mca_fine_tuning_results.pdf`` celui‑ci est inséré tel quel à la fin du
rapport.
"""

from __future__ import annotations

import argparse
import datetime
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # backend headless pour éviter l'ouverture de fenêtres
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def _add_image(pdf: PdfPages, img_path: Path, base_dir: Path) -> None:
    """Append an image to ``pdf`` with a small footer path."""
    img = plt.imread(img_path)
    fig, ax = plt.subplots(figsize=(12, 6), dpi=200)
    ax.imshow(img)
    ax.axis("off")
    fig.tight_layout()
    rel = img_path.relative_to(base_dir)
    fig.text(0.99, 0.01, str(rel), ha="right", va="bottom", fontsize=6, color="gray")
    pdf.savefig(fig, dpi=300)
    plt.close(fig)


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
            method = d.name.replace("fine_tune_", "").upper()
            if method == "MCA":
                # Intégré plus tard via fusion de PDF
                continue

            fig, ax = plt.subplots(figsize=(8.27, 11.69), dpi=200)
            ax.text(0.5, 0.5, method, fontsize=24, ha="center", va="center")
            ax.axis("off")
            pdf.savefig(fig, dpi=300)
            plt.close(fig)

            for img_path in sorted(d.rglob("*.png")):
                parts_lower = [p.lower() for p in img_path.parts]
                if "segments" in parts_lower:
                    if method == "FAMD":
                        famd_segments.append(img_path)
                    # Ignore segments for other methods
                    continue
                _add_image(pdf, img_path, output_dir)

        if famd_segments:
            fig, ax = plt.subplots(figsize=(8.27, 11.69), dpi=200)
            ax.text(0.5, 0.5, "Segments FAMD", fontsize=24, ha="center", va="center")
            ax.axis("off")
            pdf.savefig(fig, dpi=300)
            plt.close(fig)

            for img_path in famd_segments:
                _add_image(pdf, img_path, output_dir)

    # Une fois le PDF principal écrit, on peut y insérer le rapport MCA s'il existe
    mca_pdf = output_dir / "fine_tune_mca" / "mca_fine_tuning_results.pdf"
    if mca_pdf.exists():
        try:
            from PyPDF2 import PdfMerger

            merger = PdfMerger()
            merger.append(pdf_path)
            merger.append(mca_pdf)
            with open(pdf_path, "wb") as fh:
                merger.write(fh)
            merger.close()
            logger.info("PDF final avec rapport MCA inséré : %s", pdf_path)
        except Exception as exc:
            logger.error("Échec de la fusion avec le rapport MCA: %s", exc)

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
