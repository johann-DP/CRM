#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_phase4_report_fixed_lowdpi.py
Script autonome pour générer le PDF final Phase 4 à 72 dpi.
(On reprend la même structure que l’ancien code,
mais on sauve chaque page avec dpi=72 pour alléger le fichier.)
"""
from pathlib import Path
import re
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.image as mpimg

# -------------------------------------------------------------------
# 1) Chemins (à adapter si besoin) :
#    - BASE_DIR : dossier phase4_output où sont stockées les images
#    - OUTPUT_PDF : fichier PDF à générer (IL SERA CRÉÉ DANS BASE_DIR)
# -------------------------------------------------------------------
BASE_DIR = Path(r"\\Lenovo\d\DATAPREDICT\DATAPREDICT 2024\Missions\Digora\phase4_output")
OUTPUT_PDF = BASE_DIR / "RapportAnalyse_fixed_lowdpi.pdf"

# 2) Ordre exact des datasets et des méthodes factorielles
DATASETS = ["raw", "cleaned_1", "cleaned_3_univ", "cleaned_3_multi"]
METHODS  = ["famd", "mca", "mfa", "pacmap", "pca", "phate", "umap"]

# -------------------------------------------------------------------
# 3) Fonction utilitaire : ajoute une page pleine image au PDF à 72 dpi
# -------------------------------------------------------------------
def add_image_page(pdf: PdfPages, img_path: Path, title: str):
    """
    Ajoute une page pleine image à l’objet PdfPages `pdf`,
    raccorde l’image `img_path` et affiche `title` en haut.
    On sauve la page à 72 dpi (résolution basse) pour alléger le PDF.
    """
    fig = plt.figure(figsize=(11.69, 8.27))  # A4 paysage
    fig.suptitle(title, fontsize=16, y=0.98)
    ax = fig.add_subplot(111)
    ax.imshow(mpimg.imread(str(img_path)))
    ax.axis("off")
    pdf.savefig(fig, dpi=72)   # <<-<<< on sauve à 72 dpi
    plt.close(fig)

# -------------------------------------------------------------------
# 4) Fonction principale
# -------------------------------------------------------------------
def main():
    # S’assurer que le dossier existe
    BASE_DIR.mkdir(parents=True, exist_ok=True)

    with PdfPages(str(OUTPUT_PDF)) as pdf:
        # ---------------------------------------------------------------
        # Page de garde (à 72 dpi également)
        # ---------------------------------------------------------------
        fig = plt.figure(figsize=(11.69, 8.27))
        plt.text(0.5, 0.6, "Rapport d'analyse Phase 4", ha="center", va="center", fontsize=24)
        plt.text(0.5, 0.4, "Généré automatiquement (72 dpi)", ha="center", va="center", fontsize=12)
        plt.axis("off")
        pdf.savefig(fig, dpi=72)
        plt.close(fig)

        # -------------------------------------------------------------------
        # 5) Pour chaque (dataset, méthode factorielle) :
        #    a) Nuages bruts 2D & 3D
        #    b) Pour chaque (méthode_clustering, valeur_k) → page à 72 dpi
        #    c) Nuages clusterisés (cluster_grid)
        #    d) Analyse détaillée (analysis_summary)
        # -------------------------------------------------------------------
        for ds in DATASETS:
            for m in METHODS:
                folder = BASE_DIR / ds / m

                # --- a) Nuages bruts (scatter_2d & scatter_3d) ---
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
                        pdf.savefig(fig, dpi=72)
                        plt.close(fig)
                    else:
                        img = p2 if p2.exists() else p3
                        dimension = "2D" if p2.exists() else "3D"
                        add_image_page(pdf, img, f"{ds} – {m.upper()} – Nuage brut {dimension}")

                # --- b) Pages pour chaque (méthode_factorielle / méthode de clustering) × k ---
                # On recherche tous les fichiers au format : "<m>_clusters_<méthode_clustering>_k<valeur>_3d.png"
                pattern = re.compile(rf"^{re.escape(m)}_clusters_([a-zA-Z0-9]+)_k([0-9]+)_3d\.png$")
                all_cluster_files = []
                for fichier in folder.glob(f"{m}_clusters_*_k*_3d.png"):
                    nom = fichier.name
                    match = pattern.match(nom)
                    if match:
                        methode_cluster = match.group(1)  # ex: "kmeans", "gmm", "agglomerative", …
                        k_value = int(match.group(2))     # ex: 2, 3, 4, …
                        all_cluster_files.append((methode_cluster, k_value, fichier))

                # Tri par nom de clustering puis par k croissant
                all_cluster_files.sort(key=lambda x: (x[0].lower(), x[1]))

                # Pour chaque combinaison trouvée, on ajoute UNE page à 72 dpi
                for (methode_cluster, k_val, fichier) in all_cluster_files:
                    title = f"{ds} – {m.upper()} – {methode_cluster.upper()} – k={k_val}"
                    add_image_page(pdf, fichier, title)

                # --- c) Nuages clusterisés (cluster_grid) ---
                grid = folder / f"{m}_cluster_grid.png"
                if grid.exists():
                    add_image_page(pdf, grid, f"{ds} – {m.upper()} – Nuages clusterisés (cluster_grid)")

                # --- d) Analyse détaillée (analysis_summary) ---
                summary = folder / f"{m}_analysis_summary.png"
                if summary.exists():
                    add_image_page(pdf, summary, f"{ds} – {m.upper()} – Analyse détaillée")

        # -------------------------------------------------------------------
        # 6) Heatmap générale (une page pleine, à 72 dpi)
        # -------------------------------------------------------------------
        gh = BASE_DIR / "general_heatmap.png"
        if gh.exists():
            add_image_page(pdf, gh, "Heatmap générale")

        # -------------------------------------------------------------------
        # 7) 2 pages Segments (dossier old/segments), inchangées à 72 dpi
        # -------------------------------------------------------------------
        segdir = BASE_DIR / "old" / "segments"

        # 7a) Première page (2×2) : 4 variables
        vars1 = ["Catégorie", "Sous-catégorie", "Entité opérationnelle", "Statut commercial"]
        fig, axs = plt.subplots(2, 2, figsize=(11.69, 8.27))
        for ax, var in zip(axs.flatten(), vars1):
            img = segdir / f"segment_{var}.png"
            if img.exists():
                ax.imshow(mpimg.imread(str(img)))
            ax.set_title(var)
            ax.axis("off")
        plt.tight_layout()
        pdf.savefig(fig, dpi=72)
        plt.close(fig)

        # 7b) Deuxième page (3 images + placeholder % missing)
        vars2 = ["Pilier", "Statut production", "Type opportunité"]
        fig, axs = plt.subplots(2, 2, figsize=(11.69, 8.27))
        for ax, var in zip(axs.flatten()[:3], vars2):
            img = segdir / f"segment_{var}.png"
            if img.exists():
                ax.imshow(mpimg.imread(str(img)))
            ax.set_title(var)
            ax.axis("off")
        axm = axs.flatten()[3]
        axm.text(0.5, 0.5, "% Missing par segment\n(à calculer)", ha="center", va="center", fontsize=14)
        axm.axis("off")
        plt.tight_layout()
        pdf.savefig(fig, dpi=72)
        plt.close(fig)

    # Fin du contexte PdfPages
    print(f"PDF généré (72 dpi) dans : {OUTPUT_PDF}")


# -------------------------------------------------------------------
# 8) Point d’entrée
# -------------------------------------------------------------------
if __name__ == "__main__":
    main()
