# !/usr/bin/env python3
# phase4v2.py

import sys
from pathlib import Path
import pandas as pd
from PIL import Image
import io
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple, Sequence, Dict, Any
import prince
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.manifold import TSNE
import logging
import os
import numpy as np
import umap
import time

# Ex. : lire un YAML/JSON de config, ici un simple dict
CONFIG = {
    'compare_baseline': True,
    'baseline_vars': {
        'quant': ['Total recette actualisé','Total recette réalisé','Budget client estimé','duree_projet_jours','taux_realisation','marge_estimee'],
        'qual': ['Statut commercial','Statut production','Type opportunité','Catégorie','Sous-catégorie']
    },
    'baseline_cfg': {'weighting':'balanced'}
}


def sanity_check(
    df: pd.DataFrame,
    quant_vars: List[str],
    qual_vars: List[str],
    na_threshold: float = 0.30,
    max_levels: int = 50,
    corr_threshold: float = 0.98
) -> Tuple[List[str], List[str]]:
    """
    Vérifie variance, NA, corrélations, cardinalité.
    Retire et logue en WARNING les variables non-conformes. Si deux
    quantitatives dépassent le seuil de corrélation, seule la seconde
    variable du couple est éliminée.
    """
    logger = logging.getLogger(__name__)
    # 1) Quantitatives
    drop_q = set()
    sub = df[quant_vars]
    # NA %
    na_pc = sub.isna().mean()
    for v in quant_vars:
        if na_pc[v] > na_threshold:
            drop_q.add(v); logger.warning(f"Drop {v} – NA {na_pc[v]:.0%} > {na_threshold:.0%}")
        elif sub[v].var() == 0:
            drop_q.add(v); logger.warning(f"Drop {v} – variance nulle")
    # Corrélations trop élevées : parcours du triangle supérieur uniquement
    # et suppression de la seconde variable de chaque paire corrélée
    corr = sub.corr().abs()
    for idx_i in range(len(quant_vars)):
        for idx_j in range(idx_i + 1, len(quant_vars)):
            i = quant_vars[idx_i]
            j = quant_vars[idx_j]
            if corr.loc[i, j] > corr_threshold and j not in drop_q:
                drop_q.add(j)
                logger.warning(
                    f"Drop {j} – corr({i},{j})={corr.loc[i, j]:.2f} > {corr_threshold}"
                )
    # 2) Qualitatives
    drop_c = set()
    for v in qual_vars:
        prop_na = df[v].isna().mean()
        nlev = df[v].nunique(dropna=False)
        if prop_na > na_threshold:
            drop_c.add(v); logger.warning(f"Drop {v} – NA {prop_na:.0%} > {na_threshold:.0%}")
        elif nlev > max_levels:
            drop_c.add(v); logger.warning(f"Drop {v} – {nlev} modalités > {max_levels}")
    # Résultats
    q_final = [v for v in quant_vars if v not in drop_q]
    c_final = [v for v in qual_vars if v not in drop_c]
    return q_final, c_final


def load_phase_metrics(metrics_dir: str) -> pd.DataFrame:
    """
    Charge les rapports de complétude des phases 1, 2 et 3 et renvoie
    un ``DataFrame`` avec le taux de valeurs manquantes par variable.
    """
    files = [
        Path(metrics_dir) / "phase1_missing_report.csv",
        Path(metrics_dir) / "phase2_data_dictionary.xlsx",
        Path(metrics_dir) / "phase3_categorical_overview.csv",
    ]
    dfs = []
    for f in files:
        if not f.exists():
            continue

        if f.suffix.lower() == ".xlsx":
            df = pd.read_excel(f)
        else:
            df = pd.read_csv(f)

        cols = {c.lower(): c for c in df.columns}
        var_col = next((cols[c] for c in ["variable", "colonne", "column"] if c in cols), None)
        pct_col = next((cols[c] for c in ["missing_pct", "pct_missing"] if c in cols), None)
        if var_col is None or pct_col is None:
            continue

        series_pct = df[pct_col]
        if series_pct.dtype == object:
            series_pct = series_pct.str.replace(",", ".", regex=False)

        df_sub = pd.DataFrame({
            "variable": df[var_col].astype(str),
            "missing_pct": pd.to_numeric(series_pct, errors="coerce"),
        })

        dfs.append(df_sub)

    if not dfs:
        return pd.DataFrame(columns=["variable", "missing_pct"])

    return pd.concat(dfs).groupby("variable", as_index=False).mean()


def plot_na_dashboard(na_df: pd.DataFrame, output_path: Path):
    """
    Trace et enregistre un barplot du % de valeurs manquantes par variable.
    """
    if na_df.empty:
        return
    plt.figure(figsize=(12,6), dpi=200)
    na_df = na_df.sort_values('missing_pct', ascending=False).head(30)
    plt.barh(na_df['variable'], na_df['missing_pct'], color='lightcoral', edgecolor='black')
    plt.xlabel('% Missing')
    plt.title("Top 30 variables par taux de NA (Phases 1–3)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def load_data(file_path: str) -> pd.DataFrame:
    """
    Charge les données CRM brutes depuis un fichier Excel
    et réalise quelques contrôles de base.

    Parameters
    ----------
    file_path : str
        Chemin vers le fichier Excel d’export Everwin (ex. "export_everwin (19).xlsx").

    Returns
    -------
    pd.DataFrame
        DataFrame pandas contenant les données CRM brutes, avec colonnes nettoyées.

    Raises
    ------
    FileNotFoundError
        Si le fichier n'existe pas.
    ValueError
        Si la lecture échoue pour une autre raison.
    """
    logger = logging.getLogger(__name__)
    # Lecture du fichier Excel
    try:
        df = pd.read_excel(file_path)
        logger.info(f"Données chargées : {df.shape[0]} lignes × {df.shape[1]} colonnes")
    except FileNotFoundError as fnf_err:
        logger.error(f"Fichier introuvable : {file_path}")
        raise
    except Exception as e:
        logger.error(f"Erreur lors du chargement de '{file_path}' : {e}")
        raise ValueError(f"Impossible de charger le fichier Excel : {e}")

    # Vérification rapide des colonnes
    logger.debug(f"Noms de colonnes avant nettoyage : {list(df.columns)}")

    # Uniformisation des noms de colonnes (strip des espaces, normalisation Unicode)
    def _dedup_columns(cols):
        seen = {}
        new_cols = []
        for col in cols:
            base = col
            i = seen.get(base, 0)
            name = base if i == 0 else f"{base}.{i}"
            while name in new_cols:
                i += 1
                name = f"{base}.{i}"
            seen[base] = i
            new_cols.append(name)
        return new_cols

    df.columns = _dedup_columns(df.columns.astype(str).map(str.strip))

    logger.debug(f"Noms de colonnes après nettoyage : {list(df.columns)}")

    # Aucune transformation métier ici : on renvoie le DataFrame brut prêt pour la préparation
    return df


def prepare_data(df: pd.DataFrame, metrics_dir: Optional[str] = None) -> pd.DataFrame:
    """
    Prépare et nettoie le DataFrame CRM pour la Phase 4.

    Steps:
    1. Correction des dates aberrantes
    2. Conversion et nettoyage des montants
    3. Suppression des doublons
    4. Gestion des valeurs manquantes
    5. Création de variables dérivées
    6. Filtrage optionnel des outliers (flags issus de la Phase 3)

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame brut issu de ``load_data()``.
        metrics_dir : Optional[str], optional
        Répertoire contenant les métriques des phases précédentes. Si fourni,
        un tableau récapitulatif des taux de valeurs manquantes est généré et
        sauvegardé dans ce dossier.

    Returns
    -------
    pd.DataFrame
        DataFrame prêt pour l’AFDM, typé, nettoyé et sans outliers si présents.
    """
    logger = logging.getLogger(__name__)
    df_clean = df.copy()

    # 1) Dates : conversion et correction des out-of-bounds
    date_cols = [c for c in df_clean.columns if 'date' in c.lower()]
    min_date = pd.Timestamp('1990-01-01')
    max_date = pd.Timestamp('2050-12-31')
    for col in date_cols:
        df_clean[col] = pd.to_datetime(df_clean[col], errors='coerce')
        mask = df_clean[col].lt(min_date) | df_clean[col].gt(max_date)
        if mask.any():
            logger.warning(f"{mask.sum()} dates invalides dans « {col} » remplacées par NaT")
            df_clean.loc[mask, col] = pd.NaT

    # 2) Montants : conversion numérique + traitement des négatifs
    montants = [
        'Total recette actualisé',
        'Total recette réalisé',
        'Total recette produit',
        'Budget client estimé'
    ]
    for col in montants:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
            neg = df_clean[col] < 0
            if neg.any():
                logger.warning(f"{neg.sum()} valeurs négatives dans « {col} » remplacées par NaN")
                df_clean.loc[neg, col] = pd.NA

    # 3) Suppression des doublons sur l’ID affaire
    if 'Code' in df_clean.columns:
        before = len(df_clean)
        df_clean = df_clean.drop_duplicates(subset=['Code'])
        logger.info(f"Duplication supprimée : {before - len(df_clean)} lignes retirées")

    # 4) Variables dérivées (si non présentes)
    # Durée du projet (jours)
    if all(x in df_clean.columns for x in ['Date de début actualisée', 'Date de fin réelle']):
        df_clean['duree_projet_jours'] = (
                df_clean['Date de fin réelle'] - df_clean['Date de début actualisée']
        ).dt.days

    # Taux de réalisation (CA réalisé / Budget estimé)
    if all(x in df_clean.columns for x in ['Total recette réalisé', 'Budget client estimé']):
        df_clean['taux_realisation'] = (
                df_clean['Total recette réalisé'] /
                df_clean['Budget client estimé'].replace(0, np.nan)
        )
        df_clean['taux_realisation'] = df_clean['taux_realisation'].replace([np.inf, -np.inf], np.nan)

    # Marge estimée (CA réalisé - Charge prévisionnelle projet)
    if 'Charge prévisionnelle projet' in df_clean.columns and 'Total recette réalisé' in df_clean.columns:
        df_clean['marge_estimee'] = (
                df_clean['Total recette réalisé'] - df_clean['Charge prévisionnelle projet']
        )

    # 5) Valeurs manquantes
    impute_cols = montants.copy()
    if 'taux_realisation' in df_clean.columns:
        impute_cols.append('taux_realisation')
    for col in impute_cols:
        if col in df_clean.columns:
            median = df_clean[col].median()
            df_clean[col] = df_clean[col].fillna(median)
    for col in df_clean.select_dtypes(include=['object']):
        df_clean[col] = df_clean[col].fillna('Non renseigné').astype('category')

    # 6) Filtrage des outliers (flag multivarié Phase 3)
    if 'flag_multivariate' in df_clean.columns:
        count_out = int(df_clean['flag_multivariate'].sum())
        df_clean = df_clean.loc[~df_clean['flag_multivariate']]
        logger.info(f"Filtrage outliers multivariés : {count_out} lignes exclues")

    # --- Lecture et affichage des métriques Phases 1/2/3 (optionnel) ---
    if 'metrics_dir' in locals() and metrics_dir:
        na_metrics = load_phase_metrics(metrics_dir)
        dash_path = Path(metrics_dir) / "phase4_na_dashboard.png"
        plot_na_dashboard(na_metrics, dash_path)
        logger.info(f"Dashboard NA généré : {dash_path.name}")

    return df_clean


def select_variables(
        df: pd.DataFrame,
        min_modalite_freq: int = 5
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    Sélectionne les variables actives pour l'AFDM à partir du DataFrame préparé.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame nettoyé issu de prepare_data().
    min_modalite_freq : int, default=5
        Seuil minimal de fréquence pour regrouper les modalités rares en 'Autre'.

    Returns
    -------
    df_active : pd.DataFrame
        Sous-ensemble de df ne contenant que les variables sélectionnées.
    quant_vars : List[str]
        Liste des noms de variables quantitatives retenues.
    qual_vars : List[str]
        Liste des noms de variables qualitatives retenues.

    Notes
    -----
    - Exclut les variables identifiants (ex. Code, Client).
    - Regroupe les modalités rares (< min_modalite_freq) en 'Autre' pour chaque qualitative.
    - Élimine les variables à variance nulle (quantitatives) ou à unique modalité (qualitatives).
    - Les listes de variables candidates doivent refléter les enseignements des Phases 1–3.
    """
    logger = logging.getLogger(__name__)

    # 1) Listes candidates basées sur Phases 1–3
    candidate_quant = [
        'Total recette actualisé',
        'Total recette réalisé',
        'Total recette produit',
        'Budget client estimé',
        'duree_projet_jours',
        'taux_realisation',
        'marge_estimee'
    ]
    candidate_qual = [
        'Statut commercial',
        'Statut production',
        'Type opportunité',
        'Catégorie',
        'Sous-catégorie',
        'Pilier',
        'Entité opérationnelle',
        'Présence partenaire'
    ]

    # 2) Intersection avec les colonnes disponibles
    quant_vars = [c for c in candidate_quant if c in df.columns]
    qual_vars = [c for c in candidate_qual if c in df.columns]
    logger.info(f"Variables quantitatives candidates retenues : {quant_vars}")
    logger.info(f"Variables qualitatives candidates retenues : {qual_vars}")

    # 3) Exclusion des identifiants et non-pertinentes
    exclude = {'Code', 'Client', 'Contact principal', 'Titre'}
    quant_vars = [c for c in quant_vars if c not in exclude]
    qual_vars = [c for c in qual_vars if c not in exclude]

    # 4) Filtrage des quantitatives : on retire celles à variance nulle
    quant_vars = [c for c in quant_vars if df[c].var() not in (0, float('nan'))]
    logger.info(f"Quantitatives après variance > 0 : {quant_vars}")

    # 5) Traitement des qualitatives :
    #    - Regrouper les modalités rares en 'Autre'
    #    - Supprimer variables à unique modalité
    final_qual = []
    for col in qual_vars:
        counts = df[col].value_counts(dropna=False)
        rares = counts[counts < min_modalite_freq].index
        if len(rares):
            logger.info(f"{len(rares)} modalités rares dans '{col}' → regroupement en 'Autre'")
            df[col] = df[col].cat.add_categories('Autre')
            df[col] = df[col].apply(lambda x: 'Autre' if x in rares else x).astype('category')
        # Vérifier la cardinalité après regroupement
        if df[col].nunique() > 1:
            final_qual.append(col)
        else:
            logger.warning(f"Variable qualitative '{col}' exclue (une seule modalité restante)")

    qual_vars = final_qual
    logger.info(f"Qualitatives finales : {qual_vars}")

    # 6) Constitution du DataFrame actif
    selected_cols = quant_vars + qual_vars
    df_active = df[selected_cols].copy()
    logger.info(f"DataFrame actif avec {len(selected_cols)} variables")

    return df_active, quant_vars, qual_vars


def handle_missing_values(df: pd.DataFrame, quant_vars: List[str], qual_vars: List[str]) -> pd.DataFrame:
    """Impute and optionally drop NA values for the provided DataFrame."""
    logger = logging.getLogger(__name__)
    na_count = int(df.isna().sum().sum())
    if na_count > 0:
        logger.info(f"Imputation des {na_count} valeurs manquantes restantes")
        if quant_vars:
            df[quant_vars] = df[quant_vars].fillna(df[quant_vars].median())
        for col in qual_vars:
            if df[col].dtype.name == "category" and 'Non renseigné' not in df[col].cat.categories:
                df[col] = df[col].cat.add_categories('Non renseigné')
            df[col] = df[col].fillna('Non renseigné').astype('category')
        remaining_na = int(df.isna().sum().sum())
        if remaining_na > 0:
            logger.warning(
                f"{remaining_na} NA subsistent après imputation → suppression des lignes concernées"
            )
            df.dropna(inplace=True)
    else:
        logger.info("Aucune valeur manquante détectée après sanity_check")

    if df.isna().any().any():
        logger.error("Des NA demeurent dans df après traitement")
    else:
        logger.info("DataFrame sans NA prêt pour FAMD")
    return df


def segment_data(
    df: pd.DataFrame,
    qual_vars: List[str],
    output_dir: Path
) -> None:
    """Generate simple segmentation reports for qualitative variables.

    For each variable in ``qual_vars`` present in ``df``, a CSV summary and a
    bar plot are produced under ``output_dir/segments``.
    """
    logger = logging.getLogger(__name__)
    seg_dir = Path(output_dir) / "segments"
    seg_dir.mkdir(parents=True, exist_ok=True)

    for col in qual_vars:
        if col not in df.columns:
            logger.warning(f"Variable qualitative '{col}' absente du DataFrame")
            continue

        counts = df[col].value_counts(dropna=False)
        seg_df = counts.reset_index()
        seg_df.columns = ["modalité", "count"]
        seg_df["pct"] = (seg_df["count"] / seg_df["count"].sum() * 100).round(1)

        csv_path = seg_dir / f"segment_{col}.csv"
        seg_df.to_csv(csv_path, index=False)

        plt.figure(figsize=(8, 4), dpi=200)
        plt.bar(seg_df["modalité"].astype(str), seg_df["count"], edgecolor="black")
        plt.title(f"Répartition par {col}")
        plt.xlabel(col)
        plt.ylabel("count")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        png_path = seg_dir / f"segment_{col}.png"
        plt.savefig(png_path, dpi=150)
        plt.close()

        logger.info(f"Rapport segmentation '{col}' → {csv_path.name}, {png_path.name}")


def run_mfa(
    df_active: pd.DataFrame,
    quant_vars: List[str],
    qual_vars: List[str],
    output_dir: Path,
    n_components: int = 5
) -> prince.MFA:
    """Exécute une MFA sur le jeu de données mixte.

    Args:
        df_active: DataFrame contenant uniquement les colonnes des variables actives.
        quant_vars: Liste des noms de colonnes quantitatives.
        qual_vars: Liste des noms de colonnes qualitatives.
        output_dir: Répertoire où sauver les graphiques.
        n_components: Nombre de composantes factorielles à extraire.

    Returns:
        L’objet prince.MFA entraîné.
    """
    import prince
    import matplotlib.pyplot as plt

    # One-hot encode qualitative variables
    df_dummies = pd.get_dummies(df_active[qual_vars].astype(str))

    # Build groups dictionary compatible with prince >= 0.16
    groups = {"Quantitatives": quant_vars}
    for var in qual_vars:
        cols = [c for c in df_dummies.columns if c.startswith(f"{var}_")]
        if cols:
            groups[var] = cols

    # Combine numeric columns with the dummy-encoded qualitative variables
    df_mfa = pd.concat([df_active[quant_vars], df_dummies], axis=1)

    mfa = prince.MFA(n_components=n_components)
    mfa = mfa.fit(df_mfa, groups=groups)

    # Ensure compatibility with earlier code expecting explained_inertia_
    mfa.explained_inertia_ = mfa.percentage_of_variance_ / 100

    axes = list(range(1, len(mfa.explained_inertia_) + 1))
    plt.figure()
    plt.bar(axes, [v * 100 for v in mfa.explained_inertia_], edgecolor="black")
    plt.xlabel("Composante")
    plt.ylabel("% Inertie expliquée")
    plt.title("Éboulis MFA")
    plt.xticks(axes)
    plt.tight_layout()
    plt.savefig(output_dir / "phase4_mfa_scree_plot.png")
    plt.close()

    return mfa


def run_pcamix(
    df_active: pd.DataFrame,
    quant_vars: List[str],
    qual_vars: List[str],
    output_dir: Path,
    n_components: int = 5,
) -> Tuple[prince.FAMD, pd.Series, pd.DataFrame, pd.DataFrame]:
    """Exécute une analyse de type PCAmix via :class:`prince.FAMD`."""

    logger = logging.getLogger(__name__)

    # 1) Standardisation des quantitatives
    scaler = StandardScaler()
    X_quanti = scaler.fit_transform(df_active[quant_vars])
    df_quanti_scaled = pd.DataFrame(
        X_quanti,
        index=df_active.index,
        columns=quant_vars,
    )

    # 2) Assemblage du DataFrame mixte
    df_mix = pd.concat(
        [df_quanti_scaled, df_active[qual_vars].astype("category")],
        axis=1,
    )

    # 3) FAMD (équivalent PCAmix)
    md_pca = prince.FAMD(
        n_components=n_components,
        n_iter=3,
        copy=True,
        check_input=True,
        engine="sklearn",
    )
    md_pca = md_pca.fit(df_mix)

    inertia = pd.Series(
        get_explained_inertia(md_pca),
        index=[f"F{i + 1}" for i in range(n_components)],
    )

    axes = list(range(1, len(inertia) + 1))
    plt.figure()
    plt.bar(axes, [i * 100 for i in inertia], edgecolor="black")
    plt.xlabel("Composante")
    plt.ylabel("% Inertie expliquée")
    plt.title("Éboulis PCAmix")
    plt.xticks(axes)
    plt.tight_layout()
    (output_dir / "phase4_pcamix_scree_plot.png").parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "phase4_pcamix_scree_plot.png")
    plt.close()

    row_coords = md_pca.row_coordinates(df_mix)

    if hasattr(md_pca, "column_coordinates"):
        col_coords = md_pca.column_coordinates(df_mix)
    elif hasattr(md_pca, "column_principal_coordinates"):
        col_coords = md_pca.column_principal_coordinates(df_mix)
    elif hasattr(md_pca, "column_coordinates_"):
        col_coords = md_pca.column_coordinates_
    else:
        logger.warning("Aucune méthode de coordonnées colonnes disponible")
        col_coords = pd.DataFrame()

    row_coords.to_csv(output_dir / "phase4_pcamix_individus_coord.csv", index=True)
    col_coords.to_csv(output_dir / "phase4_pcamix_modalites_coord.csv", index=True)

    return md_pca, inertia, row_coords, col_coords


def run_tsne(
    embeddings: pd.DataFrame,
    df_active: pd.DataFrame,
    output_dir: Path,
    perplexity: int = 30,
    learning_rate: float = 200.0,
    n_iter: int = 1_000,
    random_state: int = 42
) -> Tuple[TSNE, pd.DataFrame]:
    """
    Applique t-SNE sur des coordonnées factorielles existantes.

    Args:
        embeddings: DataFrame des coordonnées (index = ID affaire, colonnes = axes factoriels).
        df_active: DataFrame complet, utilisé pour le coloriage (colonne "Statut commercial").
        output_dir: Répertoire où sauvegarder le scatterplot et le CSV.
        perplexity: Paramètre de t-SNE (voisinage).
        learning_rate: Taux d’apprentissage pour t-SNE.
        n_iter: Nombre d’itérations.
        random_state: Graine pour la reproductibilité.

    Returns:
        - L’instance TSNE ajustée.
        - DataFrame des embeddings t-SNE (colonnes "TSNE1","TSNE2", index = ID affaire).
    """
    # 2.1 Instanciation
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate=learning_rate,
        n_iter=n_iter,
        random_state=random_state,
        init='pca'
    )

    # 2.2 Fit & transform
    tsne_results = tsne.fit_transform(embeddings.values)

    # 2.3 DataFrame t-SNE
    tsne_df = pd.DataFrame(
        tsne_results,
        columns=["TSNE1", "TSNE2"],
        index=embeddings.index
    )

    # 2.4 Scatter plot coloré par Statut commercial
    plt.figure()
    codes = df_active["Statut commercial"].astype("category").cat.codes
    scatter = plt.scatter(
        tsne_df["TSNE1"], tsne_df["TSNE2"],
        c=codes, s=15, alpha=0.7
    )
    plt.xlabel("TSNE1")
    plt.ylabel("TSNE2")
    plt.title("t-SNE sur axes factoriels")
    cbar = plt.colorbar(scatter,
                        ticks=range(len(df_active["Statut commercial"].unique())))
    cbar.set_label("Statut commercial")

    # 2.5 Sauvegardes
    (output_dir / "phase4_tsne_scatter.png").parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_dir / "phase4_tsne_scatter.png")
    plt.close()

    tsne_df.to_csv(output_dir / "phase4_tsne_embeddings.csv", index=True)

    return tsne, tsne_df

def run_umap(
    df_active: pd.DataFrame,
    quant_vars: List[str],
    qual_vars: List[str],
    output_dir: Path,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    n_components: int = 2,
    random_state: int = 42
) -> Tuple[umap.UMAP, pd.DataFrame]:
    """
    Exécute UMAP sur un jeu mixte de variables quantitatives et qualitatives.

    Args:
        df_active: DataFrame contenant uniquement les colonnes actives.
        quant_vars: Liste de noms de colonnes quantitatives.
        qual_vars: Liste de noms de colonnes qualitatives.
        output_dir: Répertoire où sauver les graphiques et CSV.
        n_neighbors: Paramètre UMAP « voisinage ».
        min_dist: Distance minimale UMAP.
        n_components: Dimension de sortie (2 ou 3).
        random_state: Graine pour reproductibilité.

    Returns:
        - L’objet UMAP entraîné,
        - DataFrame des embeddings, colonnes ['UMAP1', 'UMAP2' (, 'UMAP3')].
    """

    # 2.1 Prétraitement des quantitatives
    X_num = df_active[quant_vars].copy()
    X_num = StandardScaler().fit_transform(X_num)

    # 2.2 Encodage one‐hot des qualitatives
    X_cat = OneHotEncoder(sparse=False, handle_unknown='ignore') \
        .fit_transform(df_active[qual_vars])

    # 2.3 Fusion des données
    X_mix = pd.DataFrame(
        data=np.hstack([X_num, X_cat]),
        index=df_active.index
    )

    # 2.4 Exécution de UMAP
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        random_state=random_state
    )
    embedding = reducer.fit_transform(X_mix)

    # 2.5 Mise en DataFrame
    cols = [f"UMAP{i+1}" for i in range(n_components)]
    umap_df = pd.DataFrame(embedding, columns=cols, index=df_active.index)

    # 2.6 Scatterplot coloré par statut commercial
    plt.figure()
    scatter = plt.scatter(
        umap_df["UMAP1"], umap_df["UMAP2"],
        c=df_active["Statut commercial"].astype('category').cat.codes,
        s=10, alpha=0.7
    )
    plt.xlabel("UMAP1")
    plt.ylabel("UMAP2")
    plt.title("Projection UMAP (colorée par Statut commercial)")
    plt.colorbar(scatter, ticks=range(len(df_active["Statut commercial"].unique())),
                 label="Statut commercial")
    plt.tight_layout()
    (output_dir / "phase4_umap_scatter.png").parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "phase4_umap_scatter.png")
    plt.close()

    # 2.7 Export CSV
    umap_df.to_csv(output_dir / "phase4_umap_embeddings.csv", index=True)

    return reducer, umap_df




def get_explained_inertia(famd) -> List[float]:
    """Return the percentage of explained inertia for each FAMD component."""
    try:
        inertia = getattr(famd, "explained_inertia_", None)
        if inertia is not None:
            return list(inertia)
    except Exception:
        inertia = None
    try:
        eigenvalues = famd.eigenvalues_
    except Exception:
        eigenvalues = getattr(famd, "eigenvalues_", None)
    if eigenvalues is None:
        return []
    total = sum(eigenvalues)
    return [v / total for v in eigenvalues]


def run_famd(
    df_active: pd.DataFrame,
    quant_vars: List[str],
    qual_vars: List[str],
    n_components: Optional[int] = None,
    famd_cfg: dict = None
) -> Tuple[
    prince.FAMD,
    pd.Series,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame
]:
    """
    Exécute l’analyse factorielle des données mixtes (FAMD) sur le jeu de variables actives.

    Parameters
    ----------
    df_active : pd.DataFrame
        DataFrame contenant uniquement les variables quantitatives et qualitatives sélectionnées.
    quant_vars : List[str]
        Liste des noms de colonnes quantitatives à standardiser.
    qual_vars : List[str]
        Liste des noms de colonnes qualitatives (catégorielles).
    n_components : int, optional
        Nombre de composantes factorielles à extraire. Si None, utilise toutes les composantes possibles.
    famd_cfg:  dict

    Returns
    -------
    famd : prince.FAMD
        Objet FAMD entraîné.
    explained_inertia : pd.Series
        Pourcentage de variance expliquée par composante.
    row_coords : pd.DataFrame
        Coordonnées des individus (affaires) dans l’espace factoriel.
    col_coords : pd.DataFrame
        Coordonnées des variables (quantitatives et modalités qualitatives) dans l’espace factoriel.
    col_contrib : pd.DataFrame
        Contributions (%) des variables et modalités aux axes factoriels.
    """
    logger = logging.getLogger(__name__)

    # 1) Pré-traitement : centrage-réduction des variables quantitatives
    scaler = StandardScaler()
    X_quanti = scaler.fit_transform(df_active[quant_vars])
    df_quanti_scaled = pd.DataFrame(
        X_quanti,
        index=df_active.index,
        columns=quant_vars
    )

    # 2) Assemblage du DataFrame mixte (quantitatif réduit + qualitatif brut)
    df_for_famd = pd.concat(
        [df_quanti_scaled, df_active[qual_vars].astype('category')],
        axis=1
    )
    if df_for_famd.isnull().any().any():
        logger.error("Des valeurs manquantes subsistent dans df_for_famd. Corriger avant FAMD.")
        raise ValueError("NA détectés dans df_for_famd")

    # 3a) Paramétrage de la pondération
    cfg = famd_cfg or {}
    weighting = cfg.get('weighting', 'balanced')  # 'balanced' | 'auto' | 'manual'

    # 3b) Initialisation de l’AFDM
    n_comp = n_components or df_for_famd.shape[1]
    famd = prince.FAMD(
            n_components = n_comp,
            n_iter = 3,
            copy = True,
            check_input = True,
            # normalize = (weighting == 'balanced'),
            engine = 'sklearn'
    )
    logger.info(f"FAMD initialisé (weighting={weighting}) avec {n_comp} composantes")

    # 4) Exécution de l’analyse
    famd = famd.fit(df_for_famd)
    logger.info("FAMD entraîné avec succès")

    # 5) Extraction des résultats
    ei_values = get_explained_inertia(famd)
    if ei_values:
        logger.info("Inertie expliquée récupérée")
    else:
        logger.warning("Aucune information d'inertie disponible")
    explained_inertia = pd.Series(
        ei_values,
        index=[f"F{i + 1}" for i in range(len(ei_values))]
    )

    row_coords = famd.row_coordinates(df_for_famd)

    if hasattr(famd, "column_coordinates"):
        col_coords = famd.column_coordinates(df_for_famd)
        logger.info("Coordonnées colonnes via 'column_coordinates'")
    elif hasattr(famd, "column_principal_coordinates"):
        col_coords = famd.column_principal_coordinates(df_for_famd)
        logger.info("'column_coordinates' absent – utilisation de 'column_principal_coordinates'")
    elif hasattr(famd, "column_coordinates_"):
        col_coords = famd.column_coordinates_
        logger.info("Coordonnées colonnes lues depuis l'attribut 'column_coordinates_'")
    else:
        logger.warning("Aucune méthode de coordonnées colonnes disponible")
        col_coords = pd.DataFrame()

    if hasattr(famd, "column_contributions"):
        col_contrib = famd.column_contributions(df_for_famd)
        logger.info("Contributions via 'column_contributions'")
    elif hasattr(famd, "column_contributions_"):
        col_contrib = famd.column_contributions_
        logger.info("'column_contributions' absent – utilisation de l'attribut 'column_contributions_'")
    else:
        contrib = (col_coords ** 2)
        col_contrib = contrib.div(contrib.sum(axis=0), axis=1) * 100
        logger.info("'column_contributions' absent – calcul à partir des coordonnées")

    logger.info(
        "Résultats FAMD extraits : "
        f"{len(explained_inertia)} axes, "
        f"{row_coords.shape[0]} individus, "
        f"{col_coords.shape[0]} variables/modalités"
    )

    return famd, explained_inertia, row_coords, col_coords, col_contrib


def plot_multimethod_results(
    results_dict: Dict[str, Dict[str, Any]],
    df_active: pd.DataFrame,
    comp_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Visualisations comparatives pour plusieurs méthodes factorielles."""

    # ─── Scree-plots ─────────────────────────────────────────────
    methods_inertia = {
        m: info["inertia"] for m, info in results_dict.items() if info["inertia"]
    }
    n = len(methods_inertia)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    for ax, (m, inertia) in zip(np.atleast_1d(axes), methods_inertia.items()):
        axes_idx = list(range(1, len(inertia) + 1))
        ax.bar(axes_idx, [i * 100 for i in inertia], edgecolor="black")
        ax.set_title(f"Éboulis {m}")
        ax.set_xlabel("Composante")
        ax.set_ylabel("% inertie")
        ax.set_xticks(axes_idx)
    plt.tight_layout()
    plt.savefig(output_dir / "multi_scree.png")
    plt.close()

    # ─── Scatter F1–F2 comparés ───────────────────────────────────
    methods_emb = results_dict.keys()
    fig, axes = plt.subplots(1, len(methods_emb), figsize=(4 * len(methods_emb), 4))
    for ax, m in zip(np.atleast_1d(axes), methods_emb):
        emb = results_dict[m]["embeddings"]
        x, y = emb.iloc[:, 0], emb.iloc[:, 1]
        codes = df_active["Statut commercial"].astype("category").cat.codes
        ax.scatter(x, y, c=codes, s=15, alpha=0.6)
        ax.set_title(f"{m} (1-2)")
        ax.set_xlabel(emb.columns[0])
        ax.set_ylabel(emb.columns[1])
    plt.tight_layout()
    plt.savefig(output_dir / "multi_scatter.png")
    plt.close()

    # ─── Heatmap d'évaluation ────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(comp_df.values, aspect="auto", cmap="viridis")
    ax.set_xticks(np.arange(comp_df.shape[1]))
    ax.set_yticks(np.arange(comp_df.shape[0]))
    ax.set_xticklabels(comp_df.columns, rotation=45, ha="right")
    ax.set_yticklabels(comp_df.index)
    ax.set_title("Comparaison méthodes")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(output_dir / "multi_heatmap.png")
    plt.close()


def export_famd_results(
        famd,
        inertia,
        row_coords: pd.DataFrame,
        col_coords: pd.DataFrame,
        col_contrib: pd.DataFrame,
        quant_vars,
        qual_vars,
        output_dir: str
):
    """
    Exporte les résultats clés de l’AFDM sous forme de CSV pour réutilisation :
      1. Coordonnées des individus sur les axes factorels retenus
      2. Coordonnées des variables quantitatives
      3. Coordonnées des modalités qualitatives
      4. Contributions des variables / modalités
      5. Variance expliquée par axe

    Parameters
    ----------
    famd : prince.FAMD
        Modèle FAMD entraîné.
    inertia : Sequence[float] | pd.Series
        Pourcentage d'inertie expliqué par axe.
    row_coords : pd.DataFrame
        Coordonnées des individus.
    col_coords : pd.DataFrame
        Coordonnées des variables / modalités.
    col_contrib : pd.DataFrame
        Contributions des variables / modalités aux axes.
    quant_vars : list of str
        Noms des variables quantitatives.
    qual_vars : list of str
        Noms des variables qualitatives.
    output_dir : str
        Répertoire où écrire les fichiers CSV.
    """
    logger = logging.getLogger(__name__)
    os.makedirs(output_dir, exist_ok=True)

    # 1) Variance expliquée (éigenvalues et % inertie)
    if isinstance(inertia, pd.Series):
        inertia = inertia.values
    axes = [f"F{i + 1}" for i in range(len(inertia))]
    var_df = pd.DataFrame({
        'axe': axes,
        'variance_%': [100 * v for v in inertia]
    })
    var_df['variance_cum_%'] = var_df['variance_%'].cumsum()
    path_var = os.path.join(output_dir, "phase4_variance_expliquee.csv")
    var_df.to_csv(path_var, index=False)
    logger.info(f"Export variance expliquée → {path_var}")

    # 2) Coordonnées des individus
    path_rows = os.path.join(output_dir, "phase4_individus_coordonnees.csv")
    row_coords.to_csv(path_rows, index=True)
    logger.info(f"Export coordonnées individus → {path_rows}")

    # 3) Coordonnées des variables/modalités
    quant_coords = col_coords.loc[quant_vars]
    qual_coords = col_coords.loc[qual_vars]

    path_quant = os.path.join(output_dir, "phase4_variables_coordonnees.csv")
    quant_coords.to_csv(path_quant, index=True)
    logger.info(f"Export coords variables quantitatives → {path_quant}")

    path_qual = os.path.join(output_dir, "phase4_modalites_coordonnees.csv")
    qual_coords.to_csv(path_qual, index=True)
    logger.info(f"Export coords modalités qualitatives → {path_qual}")

    # 4) Contributions aux axes
    contrib_df = col_contrib * 100
    contrib_df.index.name = 'variable_or_modalite'
    path_contrib = os.path.join(output_dir, "phase4_contributions_variables.csv")
    contrib_df.to_csv(path_contrib, index=True)
    logger.info(f"Export contributions variables/modalités → {path_contrib}")

    # (Optionnel) contributions détaillées par modalité
    # -- non implémenté par défaut, activer si besoin --

    logger.info("Export des résultats AFDM terminé.")


def dunn_index(X: np.ndarray, labels: np.ndarray) -> float:
    """Calcule l'indice de Dunn pour un partitionnement."""
    from scipy.spatial.distance import pdist, squareform

    if len(np.unique(labels)) < 2:
        return float("nan")

    dist = squareform(pdist(X))
    unique = np.unique(labels)

    intra_diam = []
    min_inter = np.inf

    for i, ci in enumerate(unique):
        idx_i = np.where(labels == ci)[0]
        if len(idx_i) > 1:
            intra = dist[np.ix_(idx_i, idx_i)].max()
        else:
            intra = 0.0
        intra_diam.append(intra)

        for cj in unique[i + 1 :]:
            idx_j = np.where(labels == cj)[0]
            inter = dist[np.ix_(idx_i, idx_j)].min()
            if inter < min_inter:
                min_inter = inter

    max_intra = max(intra_diam)
    if max_intra == 0:
        return float("nan")
    return float(min_inter / max_intra)


def evaluate_methods(
    results_dict: Dict[str, Dict[str, Any]],
    output_dir: Path,
    n_clusters: int = 3,
) -> pd.DataFrame:
    """Compare diverses méthodes de réduction de dimension."""

    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    import seaborn as sns

    rows = []
    for method, info in results_dict.items():
        inertias = info.get("inertia") or []
        kaiser = sum(1 for eig in inertias if eig > 1)
        cum_inertia = sum(inertias) if inertias else None

        X = info["embeddings"].values
        labels = KMeans(n_clusters=n_clusters, random_state=0).fit_predict(X)
        sil = silhouette_score(X, labels)
        dunn = dunn_index(X, labels)
        runtime = info.get("runtime")

        rows.append(
            {
                "method": method,
                "kaiser": kaiser,
                "cum_inertia": cum_inertia,
                "silhouette": sil,
                "dunn": dunn,
                "runtime_s": runtime,
            }
        )

    df_comp = pd.DataFrame(rows).set_index("method")
    df_comp.to_csv(output_dir / "methods_comparison.csv")

    df_norm = df_comp.copy()
    for col in df_norm.columns:
        cmin, cmax = df_norm[col].min(), df_norm[col].max()
        if cmax > cmin:
            df_norm[col] = (df_norm[col] - cmin) / (cmax - cmin)

    plt.figure(figsize=(8, 4), dpi=200)
    sns.heatmap(df_norm, annot=True, cmap="viridis")
    plt.tight_layout()
    (output_dir / "methods_heatmap.png").parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "methods_heatmap.png")
    plt.close()

    return df_comp


### MAIN ###

def main() -> None:
    """Orchestre la phase 4 à l'aide d'un fichier de configuration."""

    import argparse
    import json
    try:
        import yaml  # type: ignore
    except Exception:  # pragma: no cover - yaml peut manquer
        yaml = None

    parser = argparse.ArgumentParser(description="Phase 4")
    parser.add_argument("--config", required=True, help="Chemin du fichier YAML/JSON")
    args = parser.parse_args()

    cfg = Path(args.config)
    if not cfg.exists():
        raise FileNotFoundError(cfg)
    with open(cfg, "r", encoding="utf-8") as fh:
        if cfg.suffix.lower() == ".json":
            config = json.load(fh)
        else:
            if yaml is None:
                raise RuntimeError("PyYAML est requis pour lire ce fichier")
            config = yaml.safe_load(fh)

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "phase4.log"

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(sh)
    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(fh)

    logger.info("Phase 4 démarrée")

    # 1. Chargement & préparation
    df_raw = load_data(config["input_file"])
    df_clean = prepare_data(df_raw, config.get("metrics_dir"))
    df_active, quant_vars, qual_vars = select_variables(df_clean)

    # Nettoyage supplémentaire des variables sélectionnées
    quant_vars, qual_vars = sanity_check(df_active, quant_vars, qual_vars)
    df_active = df_active[quant_vars + qual_vars]
    logger.info(
        f"Après sanity_check : {len(quant_vars)} quanti, {len(qual_vars)} quali"
    )

    df_active = handle_missing_values(df_active, quant_vars, qual_vars)
    if df_active.isna().any().any():
        logger.error("Des NA demeurent dans df_active après traitement")
    else:
        logger.info("DataFrame actif sans NA prêt pour FAMD")

    segment_data(df_active, qual_vars, output_dir)

    results: Dict[str, Dict[str, Any]] = {}

    # 2. FAMD
    if "famd" in config.get("methods", []):
        t0 = time.time()
        famd_model, famd_inertia, famd_rows, famd_cols, famd_contrib = run_famd(
            df_active, quant_vars, qual_vars, **config.get("famd", {})
        )
        rt = time.time() - t0
        results["FAMD"] = {
            "embeddings": famd_rows,
            "inertia": famd_inertia,
            "runtime": rt,
        }
        logger.info(
            "FAMD : %d composantes, %.1f%% variance cumulée, %.1fs",
            famd_model.n_components,
            sum(famd_inertia) * 100 if famd_inertia is not None else 0,
            rt,
        )

    # 3. MFA
    if "mfa" in config.get("methods", []):
        t0 = time.time()
        mfa_model = run_mfa(
            df_active, quant_vars, qual_vars, output_dir, **config.get("mfa", {})
        )
        rt = time.time() - t0
        results["MFA"] = {
            "embeddings": mfa_model.row_coordinates(df_active),
            "inertia": list(mfa_model.explained_inertia_),
            "runtime": rt,
        }
        logger.info(
            "MFA : %d composantes, %.1f%% variance cumulée, %.1fs",
            mfa_model.n_components,
            mfa_model.explained_inertia_.sum() * 100,
            rt,
        )

    # 4. PCAmix
    if "pcamix" in config.get("methods", []):
        t0 = time.time()
        mdpca_model, mdpca_inertia, mdpca_rows, _ = run_pcamix(
            df_active, quant_vars, qual_vars, output_dir, **config.get("pcamix", {})
        )
        rt = time.time() - t0
        results["PCAmix"] = {
            "embeddings": mdpca_rows,
            "inertia": mdpca_inertia,
            "runtime": rt,
        }
        logger.info(
            "PCAmix : %d composantes, %.1f%% variance cumulée, %.1fs",
            mdpca_model.n_components,
            sum(mdpca_inertia) * 100,
            rt,
        )

    # 5. UMAP
    if "umap" in config.get("methods", []):
        t0 = time.time()
        umap_model, umap_df = run_umap(
            df_active, quant_vars, qual_vars, output_dir, **config.get("umap", {})
        )
        rt = time.time() - t0
        results["UMAP"] = {
            "embeddings": umap_df,
            "inertia": None,
            "runtime": rt,
        }
        logger.info("UMAP : %d composantes, %.1fs", umap_model.n_components, rt)

    # 6. t-SNE
    if "tsne" in config.get("methods", []):
        if "FAMD" not in results:
            raise RuntimeError("t-SNE requires FAMD embeddings")
        t0 = time.time()
        tsne_model, tsne_df = run_tsne(
            results["FAMD"]["embeddings"], df_active, output_dir, **config.get("tsne", {})
        )
        rt = time.time() - t0
        results["TSNE"] = {
            "embeddings": tsne_df,
            "inertia": None,
            "runtime": rt,
        }
        logger.info("t-SNE : %.1fs", rt)

    # 7. Évaluation croisée
    comp_df = evaluate_methods(results, output_dir, n_clusters=3)

    # 8. PDF comparatif
    pdf_path = generate_report_pdf(output_dir)
    logger.info(f"Rapport PDF généré : {pdf_path}")


def generate_report_pdf(output_dir: Path) -> Path:
    """Assemble un PDF de synthèse à partir des figures générées.

    The PDF will include the following images if present in ``output_dir``:
    ``phase4_mfa_scree_plot.png``, ``phase4_pcamix_scree_plot.png``,
    ``phase4_umap_scatter.png``, ``phase4_tsne_scatter.png`` and
    ``methods_heatmap.png``.
    """
    logger = logging.getLogger(__name__)
    pdf_path = output_dir / "phase4_report.pdf"
    figures = [
        "phase4_mfa_scree_plot.png",
        "phase4_pcamix_scree_plot.png",
        "phase4_umap_scatter.png",
        "phase4_tsne_scatter.png",
        "methods_heatmap.png",
    ]
    with PdfPages(pdf_path) as pdf:
        for name in figures:
            img_path = output_dir / name
            if not img_path.exists():
                logger.warning(f"Figure manquante: {name}")
                continue
            img = plt.imread(img_path)
            fig, ax = plt.subplots()
            ax.imshow(img)
            ax.axis("off")
            pdf.savefig(fig)
            plt.close(fig)
    return pdf_path


def generate_pdf(output_dir: Path, pdf_name: str = "phase4_figures.pdf"):
    logger = logging.getLogger(__name__)
    """
    Concatène tous les PNG du répertoire en un seul PDF, précédé d'une page d'index.
    """
    png_files = sorted(output_dir.glob("*.png"))
    if not png_files:
        logger.warning("Aucune figure PNG trouvée pour générer le PDF.")
        return

    # Créer la page d'index
    index_lines = [f"Figure {i + 1:02d} – {p.name}" for i, p in enumerate(png_files)]
    buf = io.BytesIO()
    # On utilise PIL pour générer une image d'index
    from PIL import ImageDraw, ImageFont
    # Page A4 à 300 dpi
    W, H = 2480, 3508
    img_idx = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(img_idx)
    font = ImageFont.load_default()
    y = 50
    for line in index_lines:
        draw.text((50, y), line, fill="black", font=font)
        y += 30
    images = [img_idx]

    # Ajouter les figures
    for p in png_files:
        img = Image.open(p)
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")
        images.append(img)

    pdf_path = output_dir / pdf_name
    images[0].save(
        pdf_path,
        format="PDF",
        save_all=True,
        append_images=images[1:],
        resolution=300
    )
    logger.info(f"PDF des figures Phase 4 généré : {pdf_path.name}")


if __name__ == "__main__":
    main()
