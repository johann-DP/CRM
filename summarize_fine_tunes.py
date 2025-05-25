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

try:
    from PyPDF2 import PdfMerger
except Exception:  # pragma: no cover - optional dependency
    PdfMerger = None


def _collect_images(method_dir: Path, *, keep_segments: bool) -> tuple[list[Path], list[Path]]:
    """Retourne les images ordinaires et celles des segments."""
    images: list[Path] = []
    segs: list[Path] = []
    for img_path in sorted(method_dir.rglob("*.png")):
        lowers = {p.lower() for p in img_path.parts}
        if "segments" in lowers or "segment" in lowers:
            if keep_segments:
                segs.append(img_path)
            continue
        images.append(img_path)
    return images, segs


def _pdf_add_images(pdf: PdfPages, images: list[Path], root: Path) -> None:
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

def generate_fine_tune_pdf(output_dir: Path, pdf_name: str = "fine_tunes_summary.pdf") -> Path:
    """Génère un PDF rassemblant toutes les figures des fine-tunes."""
    logger = logging.getLogger(__name__)
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
            logger.info("Traitement de %s", method)

            title, ax = plt.subplots(figsize=(8.27, 11.69), dpi=200)
            ax.text(0.5, 0.5, method, fontsize=24, ha="center", va="center")
            ax.axis("off")
            pdf.savefig(title, dpi=300)
            plt.close(title)

            if method == "MCA":
                candidate = d / "mca_fine_tuning_results.pdf"
                if candidate.exists():
                    mca_pdf = candidate
                else:
                    images, _ = _collect_images(d, keep_segments=False)
                    _pdf_add_images(pdf, images, output_dir)
                continue

            keep_seg = method == "FAMD"
            skip_seg = method in {"MFA", "PCA", "PCAMIX"}
            images, segs = _collect_images(d, keep_segments=keep_seg and not skip_seg)
            _pdf_add_images(pdf, images, output_dir)
            if method == "FAMD":
                famd_segments.extend(segs)

        if famd_segments:
            title, ax = plt.subplots(figsize=(8.27, 11.69), dpi=200)
            ax.text(0.5, 0.5, "Annexe – Segments FAMD", fontsize=20, ha="center", va="center")
            ax.axis("off")
            pdf.savefig(title, dpi=300)
            plt.close(title)
            _pdf_add_images(pdf, famd_segments, output_dir)

    if mca_pdf and PdfMerger:
        merger = PdfMerger()
        merger.append(str(pdf_path))
        merger.append(str(mca_pdf))
        with open(pdf_path, "wb") as fh:
            merger.write(fh)
        merger.close()

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

