# !/usr/bin/env python3
# phase4v2.py

import sys
from pathlib import Path
import pandas as pd
from PIL import Image
import io
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple, Sequence
import prince
from sklearn.preprocessing import StandardScaler
import logging
import os
import numpy as np

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
    Charge les CSV de complétude des phases 1/2/3 et renvoie un DataFrame
    avec taux de NA par variable.
    """
    files = [
        Path(metrics_dir) / "phase1_missing_report.csv",
        Path(metrics_dir) / "phase2_data_dictionary.csv",           # à ajuster si besoin
        Path(metrics_dir) / "phase3_categorical_overview.csv"
    ]
    dfs = []
    for f in files:
        if f.exists():
            df = pd.read_csv(f)
            if 'missing_pct' in df.columns:
                dfs.append(df[['variable', 'missing_pct']])
    if not dfs:
        return pd.DataFrame(columns=['variable','missing_pct'])
    return pd.concat(dfs).groupby('variable', as_index=False).mean()


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


def plot_famd_results(
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
    Génère et enregistre les principaux graphiques de l'AFDM :
    1. Scree plot des pourcentages d'inertie.
    2. Nuage de points des individus sur F1–F2.
    3. Projection des modalités qualitatives sur F1–F2.
    4. Flèches des variables quantitatives (cercle de corrélation) sur F1–F2.
    5. Histogrammes des contributions variables sur F1 et F2.

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
        Dossier où enregistrer les PNG.
    """
    logger = logging.getLogger(__name__)
    os.makedirs(output_dir, exist_ok=True)

    # Récupération des résultats
    if isinstance(inertia, pd.Series):
        inertia = inertia.values

    # 1. Scree plot
    plt.figure(figsize=(12,6), dpi=200)
    axes = [f"F{i + 1}" for i in range(len(inertia))]
    plt.bar(axes, [i * 100 for i in inertia], edgecolor='black')
    plt.plot(axes, [100 * sum(inertia[:i + 1]) for i in range(len(inertia))],
             marker='o', linestyle='--')
    plt.ylabel("% inertie")
    plt.title(
        "Éboulis des valeurs propres – AFDM")  # :contentReference[oaicite:0]{index=0}:contentReference[oaicite:1]{index=1}
    plt.xticks(rotation=45)
    plt.tight_layout()
    scree_path = os.path.join(output_dir, "phase4_scree_plot.png")
    plt.savefig(scree_path, dpi=300)
    plt.close()
    logger.info(f"Scree plot enregistré : {scree_path}")

    # 2. Nuage de points des individus F1–F2
    plt.figure(figsize=(12,6), dpi=200)
    plt.scatter(row_coords.iloc[:, 0], row_coords.iloc[:, 1],
                s=20, alpha=0.6)
    plt.xlabel("F1")
    plt.ylabel("F2")
    plt.title(
        "Projection des individus (F1 vs F2)")  # :contentReference[oaicite:2]{index=2}:contentReference[oaicite:3]{index=3}
    plt.tight_layout()
    ind_path = os.path.join(output_dir, "phase4_individus_F1_F2.png")
    plt.savefig(ind_path, dpi=300)
    plt.close()
    logger.info(f"Projection individus enregistrée : {ind_path}")

    # 3. Projection des modalités qualitatives F1–F2
    plt.figure(figsize=(12,6), dpi=200)
    subset = col_coords.loc[qual_vars, :]
    plt.scatter(subset.iloc[:, 0], subset.iloc[:, 1],
                s=50, marker='D')
    for var in qual_vars:
        x, y = col_coords.loc[var, [0, 1]]
        plt.text(x * 1.05, y * 1.05, var, fontsize=8)
    plt.xlabel("F1")
    plt.ylabel("F2")
    plt.title(
        "Modalités qualitatives (F1 vs F2)")  # :contentReference[oaicite:4]{index=4}:contentReference[oaicite:5]{index=5}
    plt.tight_layout()
    mod_path = os.path.join(output_dir, "phase4_modalites_F1_F2.png")
    plt.savefig(mod_path, dpi=300)
    plt.close()
    logger.info(f"Projection modalités enregistrée : {mod_path}")

    # 4. Cercle des corrélations (variables quantitatives)
    plt.figure(figsize=(12,6), dpi=200)
    origin = [0], [0]
    for var in quant_vars:
        x, y = col_coords.loc[var, [0, 1]]
        plt.arrow(0, 0, x, y, head_width=0.02, length_includes_head=True)
        plt.text(x * 1.1, y * 1.1, var, fontsize=8)
    circle = plt.Circle((0, 0), 1, facecolor='none', edgecolor='grey', linestyle='--')
    plt.gca().add_patch(circle)
    plt.xlabel("F1")
    plt.ylabel("F2")
    plt.title(
        "Cercle des corrélations – Variables quanti")  # :contentReference[oaicite:6]{index=6}:contentReference[oaicite:7]{index=7}
    plt.axis('equal')
    plt.tight_layout()
    corr_path = os.path.join(output_dir, "phase4_correlations_F1_F2.png")
    plt.savefig(corr_path, dpi=300)
    plt.close()
    logger.info(f"Cercle des corrélations enregistré : {corr_path}")

    # 5. Contributions sur F1 et F2
    for idx in [0, 1]:
        plt.figure(figsize=(12,6), dpi=200)
        contrib = col_contrib.iloc[:, idx] * 100
        contrib.sort_values(ascending=False).plot.bar(edgecolor='black')
        plt.ylabel("% contribution")
        plt.title(
            f"Contributions des variables – F{idx + 1}")  # :contentReference[oaicite:8]{index=8}:contentReference[oaicite:9]{index=9}
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        contrib_path = os.path.join(output_dir, f"phase4_contributions_F{idx + 1}.png")
        plt.savefig(contrib_path, dpi=300)
        plt.close()
        logger.info(f"Contributions F{idx + 1} enregistrées : {contrib_path}")


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


### MAIN ###

def main() -> None:
    """Pipeline principal de la phase 4."""
    # ─── 1) Définition des chemins ────────────────────────────
    RAW_PATH = r"D:\DATAPREDICT\DATAPREDICT 2024\Missions\Digora\export_everwin (19).xlsx"
    OUTPUT_DIR = Path(r"D:\DATAPREDICT\DATAPREDICT 2024\Missions\Digora\phase4_output")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_PATH = OUTPUT_DIR / "phase4.log"

    # ─── 2) Configuration du logger ────────────────────────────
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(ch)

    # File handler
    fh = logging.FileHandler(LOG_PATH, mode='w', encoding='utf-8')
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(fh)

    logger.info(f"Dossier d'export Phase 4 prêt : {OUTPUT_DIR}")

    # ─── 2b) Répertoire des metrics Phases 1/2/3 ───────────────
    METRICS_DIR = Path(r"D:\DATAPREDICT\DATAPREDICT 2024\Missions\Digora\phase3_output")
    logger.info(f"Répertoire metrics configuré : {METRICS_DIR}")

    # ─── 3) Pipeline d'exécution ───────────────────────────────
    try:
        # 3.1 Chargement des données
        df_raw = load_data(RAW_PATH)

        # 3.2 Préparation et nettoyage (avec dashboard NA optionnel)
        df_clean = prepare_data(df_raw, metrics_dir=str(METRICS_DIR))

        # 3.3 Sélection des variables actives
        df_active, quant_vars, qual_vars = select_variables(df_clean)

        # Sanity-check : retire variables non conformes
        quant_vars, qual_vars = sanity_check(df_active, quant_vars, qual_vars)
        df_active = df_active[quant_vars + qual_vars]
        logger.info(f"Après sanity_check : {len(quant_vars)} quanti, {len(qual_vars)} quali")

        # --- Imputation / suppression des valeurs manquantes restantes ---
        df_active = handle_missing_values(df_active, quant_vars, qual_vars)

        if df_active.isna().any().any():
            logger.error("Des NA demeurent dans df_active après traitement")
        else:
            logger.info("DataFrame actif sans NA prêt pour FAMD")

        # --- comparaison baseline vs plan ---
        if CONFIG.get('compare_baseline'):
            q0 = CONFIG['baseline_vars']['quant']
            c0 = CONFIG['baseline_vars']['qual']
            df0 = df_clean[q0 + c0].copy()
            df0 = handle_missing_values(df0, q0, c0)
            famd0, inertia0, *_ = run_famd(
                df0, q0, c0, famd_cfg=CONFIG['baseline_cfg']
            )
            # scree baseline
            axes0 = [f"F{i+1}" for i in range(len(inertia0))]
            plt.figure(figsize=(12,6), dpi=200)
            plt.bar(axes0, [i*100 for i in inertia0], edgecolor='black')
            plt.plot(
                axes0,
                [100*inertia0[:i+1].sum() for i in range(len(inertia0))],
                marker='o',
                linestyle='--'
            )
            plt.title("Baseline Scree Plot (6+5 vars)")
            plt.ylabel("% inertie")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(OUTPUT_DIR / "baseline_scree.png", dpi=300)
            plt.close()
            logger.info("Baseline scree-plot généré : baseline_scree.png")

        # 3.4 Exécution de l'AFDM
        famd, inertia, row_coords, col_coords, col_contrib = run_famd(
            df_active, quant_vars, qual_vars
        )

        # 3.5 Visualisation
        plot_famd_results(
            famd,
            inertia,
            row_coords,
            col_coords,
            col_contrib,
            quant_vars,
            qual_vars,
            str(OUTPUT_DIR)
        )

        # 3.6 Export des résultats
        export_famd_results(
            famd,
            inertia,
            row_coords,
            col_coords,
            col_contrib,
            quant_vars,
            qual_vars,
            str(OUTPUT_DIR)
        )

    except Exception as e:
        logger.error(f"Erreur fatale durant la phase 4 : {e}", exc_info=True)
        sys.exit(1)

    # ─── 4) Génération d’un PDF synthèse des figures ───────────
    generate_pdf(OUTPUT_DIR)

    # ─── 5) Listing des fichiers produits ───────────────────────
    listing_path = OUTPUT_DIR / "phase4_output_files.txt"
    with open(listing_path, "w", encoding="utf-8") as f:
        for p in sorted(OUTPUT_DIR.iterdir()):
            f.write(p.name + "\n")
    logger.info(f"Listing des fichiers produit : {listing_path.name}")


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
