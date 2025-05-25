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
matplotlib.use("Agg")  # headless backend for batch execution
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

try:  # optional dependency
    from PyPDF2 import PdfMerger
except Exception:  # pragma: no cover - PyPDF2 might be absent
    PdfMerger = None


def generate_fine_tune_pdf(output_dir: Path, pdf_name: str = "fine_tunes_summary.pdf") -> Path:
    """Génère un PDF rassemblant toutes les figures des fine-tunes.

    - Les segments du dossier ``fine_tune_famd`` sont placés en annexe.
    - Le rapport ``mca_fine_tuning_results.pdf`` est inséré tel quel lorsque
      :mod:`PyPDF2` est disponible.
    - Les sous-dossiers ``segments`` des autres méthodes (MFA, PCA, PCAMIX) sont
      ignorés pour éviter les doublons.
    """

    logger = logging.getLogger(__name__)
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / pdf_name
    fine_dirs = sorted(d for d in output_dir.glob("fine_tune_*") if d.is_dir())

    if not fine_dirs:
        logger.warning("Aucun dossier 'fine_tune_*' trouvé dans %s", output_dir)
        return pdf_path

    famd_segments: list[Path] = []
    mca_pdf = output_dir / "fine_tune_mca" / "mca_fine_tuning_results.pdf"

    with PdfPages(pdf_path) as pdf:
        # ─── Page de garde ─────────────────────────────────────────────
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
                # Ce rapport est déjà mis en forme dans un PDF séparé
                continue

            fig, ax = plt.subplots(figsize=(8.27, 11.69), dpi=200)
            ax.text(0.5, 0.5, method, fontsize=24, ha="center", va="center")
            ax.axis("off")
            pdf.savefig(fig, dpi=300)
            plt.close(fig)

            for img_path in sorted(d.rglob("*.png")):
                if "segments" in img_path.parts:
                    if method == "FAMD":
                        famd_segments.append(img_path)
                    else:
                        continue

                img = plt.imread(img_path)
                fig, ax = plt.subplots(figsize=(12, 6), dpi=200)
                ax.imshow(img)
                ax.axis("off")
                fig.tight_layout()
                rel = img_path.relative_to(output_dir)
                fig.text(0.99, 0.01, str(rel), ha="right", va="bottom", fontsize=6, color="gray")
                pdf.savefig(fig, dpi=300)
                plt.close(fig)

        # ─── Annexe FAMD segments ─────────────────────────────────────
        if famd_segments:
            fig, ax = plt.subplots(figsize=(8.27, 11.69), dpi=200)
            ax.text(0.5, 0.5, "Annexe – Segments FAMD", fontsize=20, ha="center", va="center")
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

    logger.info("PDF intermédiaire généré : %s", pdf_path)

    # ─── Intégration du rapport MCA ──────────────────────────────────
    if mca_pdf.exists() and PdfMerger is not None:
        try:
            merger = PdfMerger()
            merger.append(str(pdf_path))
            merger.append(str(mca_pdf))
            with open(pdf_path, "wb") as fh:
                merger.write(fh)
            merger.close()
            logger.info("PDF final avec rapport MCA inséré : %s", pdf_path)
        except Exception as exc:  # pragma: no cover - merge failures are rare
            logger.error("Échec de la fusion PDF : %s", exc)

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
