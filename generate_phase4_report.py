#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_phase4_report_fixed.py
Script autonome pour générer le PDF final Phase 4
(selon votre plan : 3 pages / dataset-méthode + heatmap générale + 2 pages segments).
"""
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.image as mpimg

# Chemin absolu vers votre dossier de sortie phase4.py
BASE_DIR = Path(r"\\Lenovo\d\DATAPREDICT\DATAPREDICT 2024\Missions\Digora\phase4_output")
# Chemin du PDF de sortie DANS le dossier phase4_output
OUTPUT_PDF = BASE_DIR / "RapportAnalyse_fixed.pdf"

# Ordre exact des jeux de données et méthodes factorielles
DATASETS = ["raw", "cleaned_1", "cleaned_3_univ", "cleaned_3_multi"]
METHODS  = ["famd", "mca", "mfa", "pacmap", "pca", "phate", "umap"]

def add_image_page(pdf, img_path: Path, title: str):
    """Ajoute une page pleine image au PDF, avec titre en haut."""
    fig = plt.figure(figsize=(11.69, 8.27))  # A4 paysage
    fig.suptitle(title, fontsize=16, y=0.98)
    ax = fig.add_subplot(111)
    ax.imshow(mpimg.imread(str(img_path)))
    ax.axis("off")
    pdf.savefig(fig, dpi=300)
    plt.close(fig)

def main():
    BASE_DIR.mkdir(parents=True, exist_ok=True)
    with PdfPages(str(OUTPUT_PDF)) as pdf:
        # --- Page de garde ---
        fig = plt.figure(figsize=(11.69, 8.27))
        plt.text(0.5, 0.6, "Rapport d'analyse Phase 4", ha="center", va="center", fontsize=24)
        plt.text(0.5, 0.4, "Généré automatiquement", ha="center", va="center", fontsize=12)
        plt.axis("off")
        pdf.savefig(fig, dpi=300)
        plt.close(fig)

        # --- 3 pages par dataset × méthode ---
        for ds in DATASETS:
            for m in METHODS:
                folder = BASE_DIR / ds / m

                # Page 1 : nuages de points bruts (2D & 3D)
                p2 = folder / f"{m}_scatter_2d.png"
                p3 = folder / f"{m}_scatter_3d.png"
                if p2.exists() or p3.exists():
                    if p2.exists() and p3.exists():
                        fig, axs = plt.subplots(1, 2, figsize=(11.69, 8.27))
                        for ax, img_path, label in zip(axs, [p2, p3], ["2D", "3D"]):
                            ax.imshow(mpimg.imread(str(img_path)))
                            ax.set_title(label)
                            ax.axis("off")
                        fig.suptitle(f"{ds} – {m.upper()} – Nuages de points bruts", fontsize=16, y=0.95)
                        pdf.savefig(fig, dpi=300)
                        plt.close(fig)
                    else:
                        img = p2 if p2.exists() else p3
                        add_image_page(pdf, img, f"{ds} – {m.upper()} – Nuage { '2D' if p2.exists() else '3D'} brut")

                # Page 2 : nuages clusterisés (cluster_grid)
                grid = folder / f"{m}_cluster_grid.png"
                if grid.exists():
                    add_image_page(pdf, grid, f"{ds} – {m.upper()} – Nuages clusterisés")

                # Page 3 : analyse détaillée (analysis_summary)
                summary = folder / f"{m}_analysis_summary.png"
                if summary.exists():
                    add_image_page(pdf, summary, f"{ds} – {m.upper()} – Analyse détaillée")

        # --- Heatmap générale ---
        gh = BASE_DIR / "general_heatmap.png"
        if gh.exists():
            add_image_page(pdf, gh, "Heatmap générale")

        # --- 2 pages Segments (dossier old/segments) ---
        segdir = BASE_DIR / "old" / "segments"
        # 1re page : 4 variables
        vars1 = ["Catégorie", "Sous-catégorie", "Entité opérationnelle", "Statut commercial"]
        fig, axs = plt.subplots(2, 2, figsize=(11.69, 8.27))
        for ax, var in zip(axs.flatten(), vars1):
            img = segdir / f"segment_{var}.png"
            if img.exists():
                ax.imshow(mpimg.imread(str(img)))
            ax.set_title(var)
            ax.axis("off")
        plt.tight_layout()
        pdf.savefig(fig, dpi=300)
        plt.close(fig)

        # 2e page : 3 variables + placeholder % missing
        vars2 = ["Pilier", "Statut production", "Type opportunité"]
        fig, axs = plt.subplots(2, 2, figsize=(11.69, 8.27))
        for ax, var in zip(axs.flatten()[:3], vars2):
            img = segdir / f"segment_{var}.png"
            if img.exists():
                ax.imshow(mpimg.imread(str(img)))
            ax.set_title(var)
            ax.axis("off")
        axm = axs.flatten()[3]
        axm.text(0.5, 0.5, "% Missing par segment\n(à calculer)",
                 ha="center", va="center", fontsize=14)
        axm.axis("off")
        plt.tight_layout()
        pdf.savefig(fig, dpi=300)
        plt.close(fig)

    print(f"PDF généré dans : {OUTPUT_PDF}")

if __name__ == "__main__":
    main()
