#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
phase4_famd.py

Pipeline complet pour la Phase 4 : FAMD sur données CRM mixtes.
Modules :
1. load_data
2. prepare_data
3. select_variables + sanity_check
4. run_famd (fallback PCA manuel si prince.FAMD indisponible ou incomplet)
5. plot_famd_results
6. export_famd_results

Usage :
    python phase4_famd.py --input export_everwin.xlsx --output phase4_output
"""

import os
import argparse
import logging

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# Configuration métier (à adapter si besoin)
QUANT_VARS_CANDIDATES = [
    'Total recette actualisé',
    'Total recette réalisé',
    'Total recette produit',
    'Budget client estimé',
    'duree_projet_jours',
    'taux_realisation',
    'marge_estimee'
]
QUAL_VARS_CANDIDATES = [
    'Statut commercial',
    'Statut production',
    'Type opportunité',
    'Catégorie',
    'Sous-catégorie',
    'Pilier',
    'Entité opérationnelle'
]
NA_THRESHOLD = 0.30
CORR_THRESHOLD = 0.98


# ----------------------------------------------------------------------
def setup_logger():
    logger = logging.getLogger("Phase4")
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = "%(asctime)s - %(levelname)s - %(message)s"
        handler.setFormatter(logging.Formatter(fmt))
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


logger = setup_logger()


# ----------------------------------------------------------------------
def _dedup_columns(cols):
    """
    Si deux colonnes partagent le même nom, suffixe .1, .2, etc.
    """
    seen = {}
    new_cols = []
    for col in cols:
        if col not in seen:
            seen[col] = 0
            new_cols.append(col)
        else:
            seen[col] += 1
            new_cols.append(f"{col}.{seen[col]}")
    return new_cols


def load_data(file_path: str) -> pd.DataFrame:
    """
    Module 1 : Chargement des données brutes.
    """
    logger.info(f"Lecture du fichier Excel : {file_path}")
    df = pd.read_excel(file_path)  # plus de mangle_dupe_cols
    orig_cols = list(df.columns)
    df.columns = _dedup_columns(orig_cols)
    logger.info(f"Données chargées : {df.shape[0]} lignes × {df.shape[1]} colonnes")
    return df


# ----------------------------------------------------------------------
def prepare_data(df: pd.DataFrame, metrics_dir: str = None) -> pd.DataFrame:
    """
    Module 2 : Cleaning et types.
    metrics_dir : chemin éventuel vers CSV NA/stats pour dashboard.
    """
    df = df.copy()

    # 1) Dates : forcer datetime, warnings
    date_cols = [c for c in df.columns if 'Date' in c]
    for col in date_cols:
        before_na = df[col].isna().sum()
        df[col] = pd.to_datetime(df[col], errors='coerce')
        n_na = df[col].isna().sum() - before_na
        if n_na > 0:
            logger.warning(f"{n_na} dates invalides dans « {col} » remplacées par NaT")

    # 2) Monétaires négatifs → NaN
    money_cols = [c for c in df.columns if 'Total' in c or 'Budget' in c]
    for col in money_cols:
        mask = df[col] < 0
        n_neg = mask.sum()
        if n_neg:
            df.loc[mask, col] = np.nan
            logger.warning(f"{n_neg} valeurs négatives dans « {col} » remplacées par NaN")

    # 3) Suppression doublons ligne-entière
    before = df.shape[0]
    df.drop_duplicates(inplace=True)
    removed = before - df.shape[0]
    logger.info(f"Duplication supprimée : {removed} lignes retirées")

    # 4) (Optionnel) Dashboard NA
    if metrics_dir:
        # ex: charger completeness.csv, stats.csv
        pass  # vous pouvez générer un barplot de NA ici

    # 5) Variables dérivées (si non faites en Phase 1)
    if 'Date de début initiale' in df.columns and 'Date de fin initiale' in df.columns:
        df['duree_projet_jours'] = (
                df['Date de fin initiale'] - df['Date de début initiale']
        ).dt.days
    # ex : taux_realisation = CA réalisé / CA estimé (si existants)
    if ('Total recette réalisé' in df.columns and
            'Budget client estimé' in df.columns):
        df['taux_realisation'] = (
                df['Total recette réalisé'] / df['Budget client estimé']
        )

    # marge_estimee déjà présente sinon à calculer
    return df


# ----------------------------------------------------------------------
def select_variables(df: pd.DataFrame):
    """
    Module 3a : Présélection raisonnée.
    """
    quant = [v for v in QUANT_VARS_CANDIDATES if v in df.columns]
    qual = [v for v in QUAL_VARS_CANDIDATES if v in df.columns]
    logger.info(f"Quantitatives candidates : {quant}")
    logger.info(f"Qualitatives candidates : {qual}")
    return df[quant + qual].copy(), quant, qual


def sanity_check(df: pd.DataFrame, quant_vars: list, qual_vars: list):
    """
    Module 3b : Sanity checks -> drop variables non conformes.
    """
    # 1) Drop quant avec trop de NA
    kept_q = []
    for q in quant_vars:
        na_rate = df[q].isna().mean()
        if na_rate > NA_THRESHOLD:
            logger.warning(f"Drop {q} – NA {na_rate:.0%} > {NA_THRESHOLD:.0%}")
        else:
            kept_q.append(q)

    # 2) Drop quant trop corrélées
    corr = df[kept_q].corr().abs()
    to_drop = set()
    for i, vi in enumerate(kept_q):
        for vj in kept_q[i + 1:]:
            if corr.at[vi, vj] > CORR_THRESHOLD:
                logger.warning(
                    f"Drop {vj} – corr({vi},{vj})={corr.at[vi, vj]:.2f} > {CORR_THRESHOLD}"
                )
                to_drop.add(vj)
    kept_q = [q for q in kept_q if q not in to_drop]

    # 3) Regroupement modalités rares en quali
    kept_c = []
    for c in qual_vars:
        freq = df[c].value_counts(normalize=True, dropna=False)
        rares = freq[freq < 0.01].index
        if len(rares):
            df[c] = df[c].replace(list(rares), 'Autre')
            logger.info(f"{len(rares)} modalités rares dans '{c}' → regroupement en 'Autre'")
        kept_c.append(c)

    logger.info(f"Après sanity_check : {len(kept_q)} quanti, {len(kept_c)} quali")
    return kept_q, kept_c, df


# ----------------------------------------------------------------------
def run_famd(
    df: pd.DataFrame,
    quant_vars: list,
    qual_vars: list,
    n_components: int = None,
    optimize: bool = False,
):
    """
    Module 4 : FAMD (ou fallback PCA manuel).
    Retourne 4 objets : inertia (list), row_coords (DataFrame),
                       col_coords (DataFrame), contributions (DataFrame).
    """
    # Préparation numérique
    Xq = df[quant_vars].replace([np.inf, -np.inf], np.nan)
    Xq = Xq.fillna(Xq.median())
    scaler = StandardScaler()
    Xq_scaled = pd.DataFrame(
        scaler.fit_transform(Xq),
        index=df.index,
        columns=quant_vars
    )

    # Préparation catégorielle
    Xc = df[qual_vars].fillna('Missing')
    Xc_dummies = pd.get_dummies(Xc, prefix_sep='=', drop_first=False)

    # Combine
    X_mix = pd.concat([Xq_scaled, Xc_dummies], axis=1)
    n_comp = n_components

    if optimize and n_components is None:
        n_init = min(X_mix.shape)
        try:
            import prince
            tmp = prince.FAMD(n_components=n_init, random_state=42).fit(X_mix)
            inertia_tmp = tmp.eigenvalues_ / sum(tmp.eigenvalues_)
        except Exception:
            pca_tmp = PCA(n_components=n_init, random_state=42)
            pca_tmp.fit(X_mix)
            inertia_tmp = pca_tmp.explained_variance_ratio_
        cum = np.cumsum(inertia_tmp)
        n_comp = next((i + 1 for i, v in enumerate(cum) if v >= 0.9), n_init)
        logger.info(
            "FAMD auto: %d composantes retenues (%.1f%% variance cumulée)",
            n_comp,
            cum[n_comp - 1] * 100,
        )

    n_comp = n_comp or min(X_mix.shape)

    # Tentative prince.FAMD
    try:
        import prince
        famd = prince.FAMD(
            n_components=n_comp,
            copy=True,
            random_state=42
        )
        famd = famd.fit(X_mix)
        # prince master expose eigenvalues_
        inertia = famd.eigenvalues_ / sum(famd.eigenvalues_)  # part de variance
        row_coords = famd.row_coordinates(X_mix).iloc[:, :n_comp]
        col_coords = famd.column_principal_coordinates(X_mix).iloc[:, :n_comp]
        # contributions : carré des coords normalisées
        contrib = pd.DataFrame(
            np.square(col_coords.values),
            index=col_coords.index,
            columns=col_coords.columns
        )
        return inertia, row_coords, col_coords, contrib

    except Exception as e:
        logger.warning(f"prince.FAMD a échoué ({e}), fallback PCA manuel")
        # PCA manuel
        pca = PCA(n_components=n_comp, random_state=42)
        scores = pca.fit_transform(X_mix)
        inertia = pca.explained_variance_ratio_
        # DataFrame coords individus
        axes = [f"F{k + 1}" for k in range(scores.shape[1])]
        row_coords = pd.DataFrame(scores, index=df.index, columns=axes)
        # loadings (colonnes)
        loadings = pca.components_.T
        col_coords = pd.DataFrame(loadings, index=X_mix.columns, columns=axes)
        # contributions des variables
        contrib = pd.DataFrame(
            np.square(loadings) * 100,
            index=X_mix.columns,
            columns=axes
        )
        return inertia, row_coords, col_coords, contrib


# ----------------------------------------------------------------------
def plot_famd_results(inertia, row_coords, col_coords, contrib, output_dir):
    """
    Module 5 : Génère et sauve les graphiques.
    """
    os.makedirs(output_dir, exist_ok=True)
    axes = list(range(1, len(inertia) + 1))
    # 1) Scree plot
    plt.figure(figsize=(6, 4))
    plt.bar(axes, inertia * 100, edgecolor='black')
    plt.plot(axes, np.cumsum(inertia) * 100, '-o', linestyle='--')
    plt.xlabel("Axe factoriel")
    plt.ylabel("% inertie expliquée")
    plt.title("Éboulis des valeurs propres – AFDM")
    plt.xticks(axes)
    fn = os.path.join(output_dir, "phase4_scree_plot.png")
    plt.tight_layout()
    plt.savefig(fn)
    plt.close()
    logger.info(f"Scree plot enregistré : {fn}")

    # 2) Projection individus F1-F2
    if "F1" in row_coords.columns and "F2" in row_coords.columns:
        plt.figure(figsize=(6, 6))
        plt.scatter(row_coords["F1"], row_coords["F2"], s=10, alpha=0.5)
        plt.xlabel("F1")
        plt.ylabel("F2")
        plt.title("Projection individus (F1–F2)")
        fn = os.path.join(output_dir, "phase4_individus_1_2.png")
        plt.tight_layout()
        plt.savefig(fn)
        plt.close()
        logger.info(f"Projection individus enregistrée : {fn}")

    # 3) Projection modalités + vecteurs quanti
    if "F1" in col_coords.columns and "F2" in col_coords.columns:
        plt.figure(figsize=(6, 6))
        # modalités (dummies)
        mods = [c for c in col_coords.index if '=' in c]
        plt.scatter(col_coords.loc[mods, "F1"], col_coords.loc[mods, "F2"],
                    marker='D', alpha=0.7, label="Modalités")
        # flèches variables quantitatives
        for q in [c for c in col_coords.index if c in row_coords.columns]:
            x, y = col_coords.loc[q, ["F1", "F2"]]
            plt.arrow(0, 0, x, y, head_width=0.02, length_includes_head=True)
            plt.text(x, y, q, fontsize=8)
        plt.xlabel("F1")
        plt.ylabel("F2")
        plt.title("Modalités et vecteurs quanti")
        fn = os.path.join(output_dir, "phase4_modalites_1_2.png")
        plt.tight_layout()
        plt.savefig(fn)
        plt.close()
        logger.info(f"Projection modalités enregistrée : {fn}")

    # 4) Contributions F1 et F2
    for axis in ["F1", "F2"]:
        if axis in contrib.columns:
            plt.figure(figsize=(6, 4))
            contrib_axis = contrib[axis].sort_values(ascending=False).iloc[:20]
            contrib_axis.plot(kind='bar')
            plt.ylabel("% contribution")
            plt.title(f"Contrib {axis}")
            fn = os.path.join(output_dir, f"phase4_contributions_{axis}.png")
            plt.tight_layout()
            plt.savefig(fn)
            plt.close()
            logger.info(f"Contributions {axis} enregistrées : {fn}")


# ----------------------------------------------------------------------
def export_famd_results(inertia, row_coords, col_coords, contrib, output_dir):
    """
    Module 6 : Exporte CSV.
    """
    os.makedirs(output_dir, exist_ok=True)
    # 1) Individus
    row_coords.to_csv(os.path.join(output_dir, "phase4_individus_coordonnees.csv"))
    # 2) Variables/modalités
    col_coords.to_csv(os.path.join(output_dir, "phase4_modalites_coordonnees.csv"))
    # 3) Contributions
    contrib.to_csv(os.path.join(output_dir, "phase4_contributions_variables.csv"))
    # 4) Variance expliquée
    df_var = pd.DataFrame({
        "Axe": [f"F{k + 1}" for k in range(len(inertia))],
        "Variance %": inertia * 100,
        "Variance % cum.": np.cumsum(inertia) * 100
    })
    df_var.to_csv(os.path.join(output_dir, "phase4_variance_expliquee.csv"), index=False)
    logger.info("Exports CSV terminés")


# ----------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Chemin du fichier Excel source")
    p.add_argument("--output", required=True, help="Répertoire de sortie Phase 4")
    args = p.parse_args()

    # 1. Load
    df_raw = load_data(args.input)

    # 2. Prepare
    df_clean = prepare_data(df_raw)

    # 3. Select + sanity
    df_sel, quant0, qual0 = select_variables(df_clean)
    quant1, qual1, df_sel = sanity_check(df_sel, quant0, qual0)

    # 4. Run FAMD / PCA
    inertia, rows, cols, contrib = run_famd(df_sel, quant1, qual1)

    # 5. Plots
    plot_famd_results(inertia, rows, cols, contrib, args.output)

    # 6. Exports
    export_famd_results(inertia, rows, cols, contrib, args.output)

    logger.info("Phase 4 terminée avec succès.")


if __name__ == "__main__":
    main()


# python .\tempo.py --input "D:\DATAPREDICT\DATAPREDICT 2024\Missions\Digora\export_everwin (19).xlsx" --output "D:\DATAPREDICT\DATAPREDICT 2024\Missions\Digora\phase4_output"

# 2025-05-19 22:10:27,701 - INFO - Lecture du fichier Excel : D:\DATAPREDICT\DATAPREDICT 2024\Missions\Digora\export_everwin (19).xlsx
# C:\Users\johan\PycharmProjects\Fissure-master\.venv\lib\site-packages\openpyxl\styles\stylesheet.py:237: UserWarning: Workbook contains no default style, apply openpyxl's default
#   warn("Workbook contains no default style, apply openpyxl's default")
# 2025-05-19 22:10:46,196 - INFO - Données chargées : 16100 lignes × 75 colonnes
# 2025-05-19 22:10:46,217 - WARNING - 1 dates invalides dans « Date de fin initiale » remplacées par NaT
# 2025-05-19 22:10:46,256 - WARNING - 1 dates invalides dans « Date de fin actualisée » remplacées par NaT
# 2025-05-19 22:10:46,301 - WARNING - 6 valeurs négatives dans « Total recette actualisé » remplacées par NaN
# 2025-05-19 22:10:46,302 - WARNING - 4 valeurs négatives dans « Total recette réalisé » remplacées par NaN
# 2025-05-19 22:10:46,302 - WARNING - 3 valeurs négatives dans « Total recette produit » remplacées par NaN
# 2025-05-19 22:10:46,303 - WARNING - 3 valeurs négatives dans « Budget client estimé » remplacées par NaN
# 2025-05-19 22:10:46,418 - INFO - Duplication supprimée : 0 lignes retirées
# 2025-05-19 22:10:46,421 - INFO - Quantitatives candidates : ['Total recette actualisé', 'Total recette réalisé', 'Total recette produit', 'Budget client estimé', 'duree_projet_jours', 'taux_realisation']
# 2025-05-19 22:10:46,421 - INFO - Qualitatives candidates : ['Statut commercial', 'Statut production', 'Type opportunité', 'Catégorie', 'Sous-catégorie', 'Pilier', 'Entité opérationnelle']
# 2025-05-19 22:10:46,427 - WARNING - Drop duree_projet_jours – NA 79% > 30%
# 2025-05-19 22:10:46,431 - WARNING - Drop Total recette réalisé – corr(Total recette actualisé,Total recette réalisé)=1.00 > 0.98
# 2025-05-19 22:10:46,434 - INFO - 2 modalités rares dans 'Statut commercial' → regroupement en 'Autre'
# 2025-05-19 22:10:46,438 - INFO - 4 modalités rares dans 'Statut production' → regroupement en 'Autre'
# 2025-05-19 22:10:46,450 - INFO - 15 modalités rares dans 'Catégorie' → regroupement en 'Autre'
# 2025-05-19 22:10:46,479 - INFO - 40 modalités rares dans 'Sous-catégorie' → regroupement en 'Autre'
# 2025-05-19 22:10:46,482 - INFO - 2 modalités rares dans 'Pilier' → regroupement en 'Autre'
# 2025-05-19 22:10:46,486 - INFO - 3 modalités rares dans 'Entité opérationnelle' → regroupement en 'Autre'
# 2025-05-19 22:10:46,486 - INFO - Après sanity_check : 4 quanti, 7 quali
# 2025-05-19 22:10:46,517 - WARNING - prince.FAMD a échoué (module 'prince' has no attribute 'FAMD'), fallback PCA manuel
# .\tempo.py:279: UserWarning: linestyle is redundantly defined by the 'linestyle' keyword argument and the fmt string "-o" (-> linestyle='-'). The keyword argument will take precedence.
#   plt.plot(axes, np.cumsum(inertia) * 100, '-o', linestyle='--')
# 2025-05-19 22:10:47,059 - INFO - Scree plot enregistré : D:\DATAPREDICT\DATAPREDICT 2024\Missions\Digora\phase4_output\phase4_scree_plot.png
# 2025-05-19 22:10:47,260 - INFO - Projection individus enregistrée : D:\DATAPREDICT\DATAPREDICT 2024\Missions\Digora\phase4_output\phase4_individus_1_2.png
# 2025-05-19 22:10:47,413 - INFO - Projection modalités enregistrée : D:\DATAPREDICT\DATAPREDICT 2024\Missions\Digora\phase4_output\phase4_modalites_1_2.png
# 2025-05-19 22:10:47,720 - INFO - Contributions F1 enregistrées : D:\DATAPREDICT\DATAPREDICT 2024\Missions\Digora\phase4_output\phase4_contributions_F1.png
# 2025-05-19 22:10:48,003 - INFO - Contributions F2 enregistrées : D:\DATAPREDICT\DATAPREDICT 2024\Missions\Digora\phase4_output\phase4_contributions_F2.png
# 2025-05-19 22:10:49,639 - INFO - Exports CSV terminés
# 2025-05-19 22:10:49,639 - INFO - Phase 4 terminée avec succès.
