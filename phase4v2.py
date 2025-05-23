# !/usr/bin/env python3
# phase4v2.py

import sys
from pathlib import Path
try:
    import pandas as pd
except ValueError as err:
    if "dtype size changed" in str(err):
        raise ImportError(
            "Detected an incompatible combination of pandas and NumPy. "
            "Reinstall the packages with the pinned versions from "
            "requirements.txt."
        ) from err
    raise
from PIL import Image
import io
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from typing import List, Optional, Tuple, Sequence, Dict, Any
import prince
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.manifold import TSNE
import logging
import os
import numpy as np
import umap
import time

try:  # PHATE is optional
    import phate
except Exception:  # pragma: no cover - PHATE may not be installed
    phate = None

try:  # PaCMAP is optional
    import pacmap
except Exception:  # pragma: no cover - PaCMAP may not be installed
    pacmap = None

# Ex. : lire un YAML/JSON de config, ici un simple dict
CONFIG = {
    'compare_baseline': True,
    'baseline_vars': {
        'quant': ['Total recette actualisé', 'Total recette réalisé', 'Budget client estimé', 'duree_projet_jours',
                  'taux_realisation', 'marge_estimee'],
        'qual': ['Statut commercial', 'Statut production', 'Type opportunité', 'Catégorie', 'Sous-catégorie']
    },
    'baseline_cfg': {'weighting': 'balanced'}
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
            drop_q.add(v);
            logger.warning(f"Drop {v} – NA {na_pc[v]:.0%} > {na_threshold:.0%}")
        elif sub[v].var() == 0:
            drop_q.add(v);
            logger.warning(f"Drop {v} – variance nulle")
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
            drop_c.add(v);
            logger.warning(f"Drop {v} – NA {prop_na:.0%} > {na_threshold:.0%}")
        elif nlev > max_levels:
            drop_c.add(v);
            logger.warning(f"Drop {v} – {nlev} modalités > {max_levels}")
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
    plt.figure(figsize=(12, 6), dpi=200)
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

        plt.figure(figsize=(12, 6), dpi=200)
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
        n_components: Optional[int] = None,
        optimize: bool = False,
) -> Tuple[prince.MFA, pd.DataFrame]:
    """Exécute une MFA sur le jeu de données mixte.

    Args:
        df_active: DataFrame contenant uniquement les colonnes des variables actives.
        quant_vars: Liste des noms de colonnes quantitatives.
        qual_vars: Liste des noms de colonnes qualitatives.
        output_dir: Répertoire où sauver les graphiques.
        n_components: Nombre de composantes factorielles à extraire.
        optimize: Si ``True`` et ``n_components`` est ``None``, choisit
            automatiquement le nombre d'axes (90 % de variance cumulée).

    Returns:
        - L’objet prince.MFA entraîné.
        - DataFrame des coordonnées des individus dans l'espace MFA.
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

    logger = logging.getLogger(__name__)

    n_comp = n_components
    if optimize and n_components is None:
        n_init = df_mfa.shape[1]
        tmp = prince.MFA(n_components=n_init).fit(df_mfa, groups=groups)
        inertia_tmp = tmp.percentage_of_variance_ / 100
        cum = np.cumsum(inertia_tmp)
        n_comp = next((i + 1 for i, v in enumerate(cum) if v >= 0.9), n_init)
        logger.info(
            "MFA auto: %d composantes retenues (%.1f%% variance cumulée)",
            n_comp,
            cum[n_comp - 1] * 100,
        )

    n_comp = n_comp or 5

    mfa = prince.MFA(n_components=n_comp)
    mfa = mfa.fit(df_mfa, groups=groups)
    row_coords = mfa.row_coordinates(df_mfa)

    # Ensure compatibility with earlier code expecting explained_inertia_
    mfa.explained_inertia_ = mfa.percentage_of_variance_ / 100

    axes = list(range(1, len(mfa.explained_inertia_) + 1))
    plt.figure(figsize=(12, 6), dpi=200)
    plt.bar(axes, [v * 100 for v in mfa.explained_inertia_], edgecolor="black")
    plt.xlabel("Composante")
    plt.ylabel("% Inertie expliquée")
    plt.title("Éboulis MFA")
    plt.xticks(axes)
    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "mfa_scree_plot.png")
    plt.close()

    return mfa, row_coords


def run_pcamix(
        df_active: pd.DataFrame,
        quant_vars: List[str],
        qual_vars: List[str],
        output_dir: Path,
        n_components: Optional[int] = None,
        optimize: bool = False,
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

    n_comp = n_components
    if optimize and n_components is None:
        n_init = df_mix.shape[1]
        tmp = prince.FAMD(
            n_components=n_init,
            n_iter=3,
            copy=True,
            check_input=True,
            engine="sklearn",
        ).fit(df_mix)
        inertia_tmp = get_explained_inertia(tmp)
        cum = np.cumsum(inertia_tmp)
        n_comp = next((i + 1 for i, v in enumerate(cum) if v >= 0.9), n_init)
        logger.info(
            "PCAmix auto: %d composantes retenues (%.1f%% variance cumulée)",
            n_comp,
            cum[n_comp - 1] * 100,
        )

    n_comp = n_comp or 5

    # 3) FAMD (équivalent PCAmix)
    md_pca = prince.FAMD(
        n_components=n_comp,
        n_iter=3,
        copy=True,
        check_input=True,
        engine="sklearn",
    )
    md_pca = md_pca.fit(df_mix)

    inertia_values = get_explained_inertia(md_pca)
    inertia = pd.Series(
        inertia_values,
        index=[f"F{i + 1}" for i in range(len(inertia_values))],
    )

    axes = list(range(1, len(inertia) + 1))
    plt.figure(figsize=(12, 6), dpi=200)
    plt.bar(axes, [i * 100 for i in inertia], edgecolor="black")
    plt.xlabel("Composante")
    plt.ylabel("% Inertie expliquée")
    plt.title("Éboulis PCAmix")
    plt.xticks(axes)
    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "pcamix_scree_plot.png")
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

    return md_pca, inertia, row_coords, col_coords


def run_pca(
        df_active: pd.DataFrame,
        quant_vars: List[str],
        qual_vars: List[str],
        output_dir: Path,
        n_components: Optional[int] = None,
        optimize: bool = False,
) -> Tuple[Any, pd.Series, pd.DataFrame, pd.DataFrame]:
    """Réalise une analyse en composantes principales (ACP) sur les seules
    variables quantitatives.

    Cette étape isole l'information portée par les montants, durées ou scores
    numériques afin d'identifier les grands facteurs de variation indépendamment
    des catégories CRM. Les coordonnées des individus et des variables sont
    renvoyées pour pouvoir comparer leur contribution aux axes principaux."""

    logger = logging.getLogger(__name__)

    # 1) Standardisation des variables quantitatives pour éviter qu'une échelle
    #    domine l'analyse
    scaler = StandardScaler()
    X = scaler.fit_transform(df_active[quant_vars])
    df_scaled = pd.DataFrame(X, index=df_active.index, columns=quant_vars)

    n_comp = n_components
    if optimize and n_components is None:
        n_init = df_scaled.shape[1]
        tmp = prince.PCA(n_components=n_init).fit(df_scaled)
        inertia_tmp = get_explained_inertia(tmp)
        cum = np.cumsum(inertia_tmp)
        n_comp = next((i + 1 for i, v in enumerate(cum) if v >= 0.9), n_init)
        logger.info(
            "PCA auto: %d composantes retenues (%.1f%% variance cumulée)",
            n_comp,
            cum[n_comp - 1] * 100,
        )

    n_comp = n_comp or min(df_scaled.shape)

    pca = prince.PCA(n_components=n_comp)
    pca = pca.fit(df_scaled)

    inertia = pd.Series(
        get_explained_inertia(pca),
        index=[f"F{i + 1}" for i in range(pca.n_components)],
    )

    row_coords = pca.row_coordinates(df_scaled)
    if hasattr(pca, "column_correlations"):
        attr = pca.column_correlations
        col_coords = attr(df_scaled) if callable(attr) else attr
    elif hasattr(pca, "column_correlations_"):
        col_coords = pca.column_correlations_
    else:
        col_coords = pd.DataFrame(pca.components_.T, index=quant_vars)

    contrib = (col_coords ** 2).div((col_coords ** 2).sum(axis=0), axis=1) * 100

    axes = list(range(1, len(inertia) + 1))
    plt.figure(figsize=(12, 6), dpi=200)
    plt.bar(axes, [i * 100 for i in inertia], edgecolor="black")
    plt.xlabel("Composante")
    plt.ylabel("% Inertie expliquée")
    plt.title("Éboulis PCA")
    plt.xticks(axes)
    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "pca_scree_plot.png")
    plt.close()

    return pca, inertia, row_coords, col_coords, contrib


def run_mca(
        df_active: pd.DataFrame,
        quant_vars: List[str],
        qual_vars: List[str],
        output_dir: Path,
        n_components: Optional[int] = None,
        optimize: bool = False,
) -> Tuple[Any, pd.Series, pd.DataFrame, pd.DataFrame]:
    """Effectue une analyse des correspondances multiples (ACM) sur les seules
    variables qualitatives.

    L'ACM révèle les proximités ou oppositions entre modalités (secteur,
    statut, typologie…) sans influence des mesures numériques. Elle complète
    ainsi l'ACP pour comprendre l'information catégorielle du CRM."""

    logger = logging.getLogger(__name__)

    df_cat = df_active[qual_vars].astype("category")

    n_comp = n_components
    if optimize and n_components is None:
        max_dim = sum(df_cat[c].nunique() for c in df_cat.columns) - len(df_cat.columns)
        tmp = prince.MCA(n_components=max_dim).fit(df_cat)
        inertia_tmp = getattr(tmp, "explained_inertia_", tmp.eigenvalues_ / tmp.eigenvalues_.sum())
        cum = np.cumsum(inertia_tmp)
        n_comp = next((i + 1 for i, v in enumerate(cum) if v >= 0.9), max_dim)
        logger.info(
            "MCA auto: %d dimensions retenues (%.1f%% inertie cumulée)",
            n_comp,
            cum[n_comp - 1] * 100,
        )

    n_comp = n_comp or 5

    mca = prince.MCA(n_components=n_comp)
    mca = mca.fit(df_cat)

    inertia = pd.Series(
        getattr(mca, "explained_inertia_", mca.eigenvalues_ / mca.eigenvalues_.sum()),
        index=[f"F{i + 1}" for i in range(mca.n_components)],
    )

    row_coords = mca.row_coordinates(df_cat)
    if hasattr(mca, "column_coordinates"):
        col_coords = mca.column_coordinates(df_cat)
    elif hasattr(mca, "column_coordinates_"):
        col_coords = mca.column_coordinates_
    else:
        col_coords = pd.DataFrame()

    contrib = (col_coords ** 2).div((col_coords ** 2).sum(axis=0), axis=1) * 100

    axes = list(range(1, len(inertia) + 1))
    plt.figure(figsize=(12, 6), dpi=200)
    plt.bar(axes, [i * 100 for i in inertia], edgecolor="black")
    plt.xlabel("Composante")
    plt.ylabel("% Inertie expliquée")
    plt.title("Éboulis MCA")
    plt.xticks(axes)
    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "mca_scree_plot.png")
    plt.close()

    return mca, inertia, row_coords, col_coords, contrib


def run_tsne(
        embeddings: pd.DataFrame,
        df_active: pd.DataFrame,
        output_dir: Path,
        perplexity: Optional[int] = None,
        learning_rate: float = 200.0,
        n_iter: int = 1_000,
        random_state: int = 42,
        n_components: int = 2,
        optimize: bool = False,
        perplexity_grid: Sequence[int] | None = None,
) -> Tuple[TSNE, pd.DataFrame]:
    """
    Applique t-SNE sur des coordonnées factorielles existantes et renvoie les
    embeddings calculés. Les figures et CSV sont enregistrés via
    :func:`export_tsne_results`.

    Parameters
    ----------
    embeddings : pd.DataFrame
        Coordonnées factorielles (lignes = individus).
    df_active : pd.DataFrame
        DataFrame complet pour récupérer la variable ``Statut commercial``.
    output_dir : Path
        Répertoire des résultats (créé dans ``main``).
    perplexity : int, optional
        Paramètre "voisinage" du t-SNE. S'il est omis et ``optimize`` vaut
        ``True``, une recherche simple est effectuée.
    learning_rate : float
        Taux d'apprentissage du t-SNE.
    n_iter : int
        Nombre d'itérations d'entraînement.
    random_state : int
        Graine pour la reproductibilité.
    n_components : int
        Dimension de la projection t-SNE (2 ou 3).
    optimize : bool
        Active l'optimisation automatique de ``perplexity``.
    perplexity_grid : Sequence[int], optional
        Valeurs testées si ``perplexity`` n'est pas fourni.

    Returns:
    -------
    tuple
        L'instance :class:`TSNE` ajustée et le ``DataFrame`` des embeddings
        (colonnes ``TSNE1``, ``TSNE2`` (, ``TSNE3``)).
    """
    logger = logging.getLogger(__name__)
    perpl = perplexity if perplexity is not None else 30

    def _fit_tsne(p):
        try:
            t = TSNE(
                n_components=n_components,
                perplexity=p,
                learning_rate=learning_rate,
                max_iter=n_iter,
                random_state=random_state,
                init="pca",
            )
        except TypeError:  # pragma: no cover - older scikit-learn
            t = TSNE(
                n_components=n_components,
                perplexity=p,
                learning_rate=learning_rate,
                n_iter=n_iter,
                random_state=random_state,
                init="pca",
            )
        emb = t.fit_transform(embeddings.values)
        return t, emb

    if optimize and perplexity is None:
        from sklearn.manifold import trustworthiness

        grid = perplexity_grid or [5, 30, 50]
        best = None
        for p in grid:
            t, emb = _fit_tsne(p)
            score = trustworthiness(embeddings.values, emb)
            if best is None or score > best[0]:
                best = (score, p, t, emb)
        best_score, perpl, tsne, tsne_results = best
        logger.info(
            "t-SNE optimal: perplexity=%d (trustworthiness=%.3f)",
            perpl,
            best_score,
        )
    else:
        tsne, tsne_results = _fit_tsne(perpl)

    # 2.3 DataFrame t-SNE
    cols = [f"TSNE{i + 1}" for i in range(n_components)]
    tsne_df = pd.DataFrame(tsne_results, columns=cols, index=embeddings.index)

    return tsne, tsne_df


def export_tsne_results(tsne_df: pd.DataFrame, df_active: pd.DataFrame, output_dir: Path) -> None:
    """Save scatter plot(s) and embeddings for t-SNE."""
    logger = logging.getLogger(__name__)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Scatter 2D
    if {"TSNE1", "TSNE2"}.issubset(tsne_df.columns):
        plt.figure(figsize=(12, 6), dpi=200)
        codes = df_active.loc[tsne_df.index, "Statut commercial"].astype("category").cat.codes
        sc = plt.scatter(
            tsne_df["TSNE1"], tsne_df["TSNE2"], c=codes, s=15, alpha=0.7
        )
        plt.xlabel("TSNE1")
        plt.ylabel("TSNE2")
        plt.title("t-SNE sur axes factoriels (FAMD)")
        plt.colorbar(sc, label="Statut commercial")
        plt.tight_layout()
        fig_path = output_dir / "tsne_scatter.png"
        plt.savefig(fig_path)
        plt.close()
        logger.info(f"Export t-SNE -> {fig_path}")

    # Scatter 3D
    if {"TSNE1", "TSNE2", "TSNE3"}.issubset(tsne_df.columns):
        fig = plt.figure(figsize=(12, 6), dpi=200)
        ax = fig.add_subplot(111, projection="3d")
        codes = df_active.loc[tsne_df.index, "Statut commercial"].astype("category").cat.codes
        sc = ax.scatter(
            tsne_df["TSNE1"], tsne_df["TSNE2"], tsne_df["TSNE3"],
            c=codes, s=15, alpha=0.7
        )
        ax.set_xlabel("TSNE1")
        ax.set_ylabel("TSNE2")
        ax.set_zlabel("TSNE3")
        ax.set_title("t-SNE 3D (axes factoriels)")
        fig.colorbar(sc, label="Statut commercial")
        plt.tight_layout()
        fig3d_path = output_dir / "tsne_scatter_3D.png"
        plt.savefig(fig3d_path)
        plt.close()
        logger.info(f"Export t-SNE -> {fig3d_path}")

    csv_path = output_dir / "tsne_embeddings.csv"
    tsne_df.to_csv(csv_path, index=True)
    logger.info(f"Export t-SNE -> {csv_path}")



def run_umap(
        df_active: pd.DataFrame,
        quant_vars: List[str],
        qual_vars: List[str],
        output_dir: Path,
        n_neighbors: Optional[int] = None,
        min_dist: Optional[float] = None,
        n_components: int = 2,
        random_state: int | None = 42,
        n_jobs: int | None = None,
        optimize: bool = False,
) -> Tuple[umap.UMAP, pd.DataFrame]:
    """
    Exécute UMAP sur un jeu mixte de variables quantitatives et qualitatives.

    Args:
        df_active: DataFrame contenant uniquement les colonnes actives.
        quant_vars: Liste de noms de colonnes quantitatives.
        qual_vars: Liste de noms de colonnes qualitatives.
        output_dir: Répertoire où seront exportés les résultats (via
            :func:`export_umap_results`).
        n_neighbors: Paramètre UMAP « voisinage ».
        min_dist: Distance minimale UMAP.
        n_components: Dimension de sortie (2 ou 3).
        random_state: Graine pour reproductibilité. ``None`` pour laisser
            UMAP utiliser le parallélisme.
        n_jobs: Nombre de threads UMAP. Si ``random_state`` est défini et
            ``n_jobs`` n'est pas ``1``, la valeur sera forcée à ``1`` pour
            éviter le warning de ``umap-learn``.
        optimize: si ``True`` et que ``n_neighbors``/``min_dist`` ne sont pas
            fournis, recherche la meilleure combinaison (trustworthiness).

    Returns:
        - L’objet UMAP entraîné,
        - DataFrame des embeddings, colonnes ['UMAP1', 'UMAP2' (, 'UMAP3')].
    """

    # 2.1 Prétraitement des quantitatives
    X_num = df_active[quant_vars].copy()
    X_num = StandardScaler().fit_transform(X_num)

    # 2.2 Encodage one‐hot des qualitatives
    try:
        # scikit-learn >= 1.2
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    except TypeError:  # pragma: no cover - older scikit-learn
        encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')

    X_cat = encoder.fit_transform(df_active[qual_vars])

    # 2.3 Fusion des données
    X_mix = pd.DataFrame(
        data=np.hstack([X_num, X_cat]),
        index=df_active.index
    )

    logger = logging.getLogger(__name__)

    n_neigh = n_neighbors if n_neighbors is not None else 15
    dist = min_dist if min_dist is not None else 0.1

    def _build_umap(nn, md):
        nj = n_jobs
        if random_state is not None:
            if nj not in (None, 1):
                logger.warning(
                    "random_state défini (%s) : n_jobs=%s forcé à 1 pour garantir la reproductibilité",
                    random_state,
                    nj,
                )
            nj = 1
        return umap.UMAP(
            n_neighbors=nn,
            min_dist=md,
            n_components=n_components,
            random_state=random_state,
            n_jobs=nj,
        )

    if optimize and (n_neighbors is None or min_dist is None):
        from joblib import Parallel, delayed
        from sklearn.manifold import trustworthiness

        neigh_grid = [5, 15, 30, 50] if n_neighbors is None else [n_neighbors]
        dist_grid = [0.1, 0.5] if min_dist is None else [min_dist]

        def eval_combo(nn, md):
            reducer = _build_umap(nn, md)
            emb = reducer.fit_transform(X_mix)
            score = trustworthiness(X_mix, emb)
            return score, nn, md, emb, reducer

        results = Parallel(n_jobs=-1)(
            delayed(eval_combo)(nn, md) for nn in neigh_grid for md in dist_grid
        )
        best = max(results, key=lambda x: x[0])
        best_score, n_neigh, dist, embedding, reducer = best
        logger.info(
            "UMAP optimal: n_neighbors=%d, min_dist=%.2f (trustworthiness=%.3f)",
            n_neigh,
            dist,
            best_score,
        )
    else:
        reducer = _build_umap(n_neigh, dist)
        embedding = reducer.fit_transform(X_mix)

    # 2.5 Mise en DataFrame
    cols = [f"UMAP{i + 1}" for i in range(n_components)]
    umap_df = pd.DataFrame(embedding, columns=cols, index=df_active.index)

    # Les figures et CSV seront générés par ``export_umap_results`` dans ``main``.

    return reducer, umap_df


def export_umap_results(
        umap_df: pd.DataFrame,
        df_active: pd.DataFrame,
        output_dir: Path,
) -> None:
    """Enregistre les figures et CSV pour les embeddings UMAP."""
    logger = logging.getLogger(__name__)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Scatter 2D
    if {"UMAP1", "UMAP2"}.issubset(umap_df.columns):
        plt.figure(figsize=(12, 6), dpi=200)
        codes = df_active.loc[umap_df.index, "Statut commercial"].astype("category").cat.codes
        sc = plt.scatter(umap_df["UMAP1"], umap_df["UMAP2"], c=codes, s=10, alpha=0.7)
        plt.xlabel("UMAP1")
        plt.ylabel("UMAP2")
        plt.title("Projection UMAP")
        plt.colorbar(sc, label="Statut commercial")
        plt.tight_layout()
        fig_path = output_dir / "umap_scatter.png"
        plt.savefig(fig_path)
        plt.close()
        logger.info("Projection UMAP 2D enregistrée: %s", fig_path)

    # Scatter 3D
    if {"UMAP1", "UMAP2", "UMAP3"}.issubset(umap_df.columns):
        fig = plt.figure(figsize=(12, 6), dpi=200)
        ax = fig.add_subplot(111, projection="3d")
        codes = df_active.loc[umap_df.index, "Statut commercial"].astype("category").cat.codes
        sc3d = ax.scatter(
            umap_df["UMAP1"], umap_df["UMAP2"], umap_df["UMAP3"],
            c=codes, s=10, alpha=0.7,
        )
        ax.set_xlabel("UMAP1")
        ax.set_ylabel("UMAP2")
        ax.set_zlabel("UMAP3")
        ax.set_title("Projection UMAP 3D")
        fig.colorbar(sc3d, ax=ax, label="Statut commercial")
        plt.tight_layout()
        fig3d_path = output_dir / "umap_scatter_3D.png"
        plt.savefig(fig3d_path)
        plt.close()
        logger.info("Projection UMAP 3D enregistrée: %s", fig3d_path)

    csv_path = output_dir / "umap_embeddings.csv"
    umap_df.to_csv(csv_path, index=True)
    logger.info("CSV embeddings UMAP enregistré: %s", csv_path)


def run_pacmap(
        df_active: pd.DataFrame,
        quant_vars: List[str],
        qual_vars: List[str],
        output_dir: Path,
        n_components: int = 2,
        n_neighbors: int = 10,
        MN_ratio: float = 0.5,
        FP_ratio: float = 2.0,
) -> Tuple[Any, pd.DataFrame]:
    """Exécute PaCMAP sur les données CRM pour visualiser clients et tendances.

    PaCMAP vise à préserver simultanément les structures locales et globales ;
    il peut ainsi révéler des regroupements de clients équilibrés, utiles pour
    la segmentation CRM. L'initialisation ``pca`` améliore la stabilité des
    résultats. Les paramètres par défaut conviennent généralement (``n_neighbors``
    agit sur la granularité).
    """

    logger = logging.getLogger(__name__)
    if pacmap is None:
        logger.error(
            "Le module pacmap n\u2019est pas install\u00e9. Veuillez l\u2019installer pour utiliser PaCMAP."
        )
        return None, pd.DataFrame()

    X_num = StandardScaler().fit_transform(df_active[quant_vars])
    try:
        enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    except TypeError:  # pragma: no cover - older scikit-learn
        enc = OneHotEncoder(sparse=False, handle_unknown="ignore")
    X_cat = enc.fit_transform(df_active[qual_vars])

    X = np.hstack([X_num, X_cat])

    logger.info(
        "Paramètres PaCMAP: n_components=%d, n_neighbors=%d, MN_ratio=%.2f, FP_ratio=%.2f",
        n_components,
        n_neighbors,
        MN_ratio,
        FP_ratio,
    )

    pacmap_model = pacmap.PaCMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        MN_ratio=MN_ratio,
        FP_ratio=FP_ratio,
        init="pca",
    )

    embedding = pacmap_model.fit_transform(X)

    cols = [f"PACMAP{i + 1}" for i in range(n_components)]
    pacmap_df = pd.DataFrame(embedding, index=df_active.index, columns=cols)

    return pacmap_model, pacmap_df


def export_pacmap_results(
        pacmap_df: pd.DataFrame,
        df_active: pd.DataFrame,
        output_dir: Path,
) -> None:
    """Génère les visualisations et CSV pour PaCMAP."""

    logger = logging.getLogger(__name__)
    output_dir.mkdir(parents=True, exist_ok=True)

    if {"PACMAP1", "PACMAP2"}.issubset(pacmap_df.columns):
        plt.figure(figsize=(12, 6), dpi=200)
        codes = df_active.loc[pacmap_df.index, "Statut commercial"].astype(
            "category"
        ).cat.codes
        sc = plt.scatter(
            pacmap_df["PACMAP1"],
            pacmap_df["PACMAP2"],
            c=codes,
            s=10,
            alpha=0.7,
        )
        plt.xlabel("PACMAP1")
        plt.ylabel("PACMAP2")
        plt.title("PaCMAP \u2013 individus (dim 1 vs dim 2)")
        plt.colorbar(sc, label="Statut commercial")
        plt.tight_layout()
        plt.savefig(output_dir / "pacmap_scatter.png")
        plt.close()
        logger.info("Projection PaCMAP 2D enregistr\u00e9e")

    if {"PACMAP1", "PACMAP2", "PACMAP3"}.issubset(pacmap_df.columns):
        fig = plt.figure(figsize=(12, 6), dpi=200)
        ax = fig.add_subplot(111, projection="3d")
        codes = df_active.loc[pacmap_df.index, "Statut commercial"].astype(
            "category"
        ).cat.codes
        sc3 = ax.scatter(
            pacmap_df["PACMAP1"],
            pacmap_df["PACMAP2"],
            pacmap_df["PACMAP3"],
            c=codes,
            s=10,
            alpha=0.7,
        )
        ax.set_xlabel("PACMAP1")
        ax.set_ylabel("PACMAP2")
        ax.set_zlabel("PACMAP3")
        ax.set_title("PaCMAP \u2013 individus (3D)")
        fig.colorbar(sc3, ax=ax, label="Statut commercial")
        plt.tight_layout()
        plt.savefig(output_dir / "pacmap_scatter_3D.png")
        plt.close()
        logger.info("Projection PaCMAP 3D enregistr\u00e9e")

    pacmap_df.to_csv(output_dir / "pacmap_coordinates.csv", index=True)
    logger.info("CSV PaCMAP enregistr\u00e9")


def run_phate(
        df_active: pd.DataFrame,
        quant_vars: List[str],
        qual_vars: List[str],
        output_dir: Path,
        n_components: int = 2,
) -> Tuple[Any | None, pd.DataFrame]:
    """Exécute PHATE sur les données CRM pour détecter des trajectoires potentielles.

    PHATE est particulièrement adapté pour révéler des évolutions progressives,
    par exemple le passage de prospect à client fidèle. Il s'appuie sur un
    graphe de voisins et une diffusion pour préserver la structure globale. Les
    valeurs par défaut (``knn=5``, ``t='auto'``) conviennent généralement aux
    volumes CRM ; ``n_jobs=-1`` exploite tous les cœurs et ``random_state=42``
    assure la reproductibilité.
    """

    logger = logging.getLogger(__name__)
    if phate is None:
        logger.error(
            "Le module PHATE n'est pas installé. Installez-le pour utiliser l'analyse PHATE."
        )
        return None, pd.DataFrame()

    X_num = StandardScaler().fit_transform(df_active[quant_vars])
    try:
        enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    except TypeError:  # pragma: no cover - older scikit-learn
        enc = OneHotEncoder(sparse=False, handle_unknown="ignore")
    X_cat = enc.fit_transform(df_active[qual_vars])

    X_mix = np.hstack([X_num, X_cat])

    logger.info("Paramètres PHATE: n_components=%d", n_components)

    phate_operator = phate.PHATE(
        n_components=n_components,
        n_jobs=-1,
        random_state=42,
    )

    embedding = phate_operator.fit_transform(X_mix)

    cols = [f"PHATE{i + 1}" for i in range(n_components)]
    phate_df = pd.DataFrame(embedding, index=df_active.index, columns=cols)

    return phate_operator, phate_df


def export_phate_results(
        phate_df: pd.DataFrame,
        df_active: pd.DataFrame,
        output_dir: Path,
) -> None:
    """Génère les visualisations et CSV pour PHATE."""

    logger = logging.getLogger(__name__)
    output_dir.mkdir(parents=True, exist_ok=True)

    if {"PHATE1", "PHATE2"}.issubset(phate_df.columns):
        plt.figure(figsize=(12, 6), dpi=200)
        codes = df_active.loc[phate_df.index, "Statut commercial"].astype("category").cat.codes
        sc = plt.scatter(
            phate_df["PHATE1"],
            phate_df["PHATE2"],
            c=codes,
            s=10,
            alpha=0.7,
        )
        plt.xlabel("PHATE1")
        plt.ylabel("PHATE2")
        plt.title("PHATE – individus (dim 1–2)")
        plt.colorbar(sc, label="Statut commercial")
        plt.tight_layout()
        plt.savefig(output_dir / "phate_scatter.png")
        plt.close()
        logger.info("Projection PHATE 2D enregistrée")

    if {"PHATE1", "PHATE2", "PHATE3"}.issubset(phate_df.columns):
        fig = plt.figure(figsize=(12, 6), dpi=200)
        ax = fig.add_subplot(111, projection="3d")
        codes = df_active.loc[phate_df.index, "Statut commercial"].astype("category").cat.codes
        sc3 = ax.scatter(
            phate_df["PHATE1"],
            phate_df["PHATE2"],
            phate_df["PHATE3"],
            c=codes,
            s=10,
            alpha=0.7,
        )
        ax.set_xlabel("PHATE1")
        ax.set_ylabel("PHATE2")
        ax.set_zlabel("PHATE3")
        ax.set_title("PHATE – individus (3D)")
        fig.colorbar(sc3, ax=ax, label="Statut commercial")
        plt.tight_layout()
        plt.savefig(output_dir / "phate_scatter_3D.png")
        plt.close()
        logger.info("Projection PHATE 3D enregistrée")

    phate_df.to_csv(output_dir / "phate_coordinates.csv", index=True)
    logger.info("CSV PHATE enregistré")


def get_explained_inertia(model) -> List[float]:
    """Return the percentage of explained inertia for a decomposition model."""
    try:
        inertia = getattr(model, "explained_inertia_", None)
        if inertia is not None:
            return list(inertia)
    except Exception:
        inertia = None
    try:
        eigenvalues = model.eigenvalues_
    except Exception:
        eigenvalues = getattr(model, "eigenvalues_", None)
    if eigenvalues is None:
        return []
    total = sum(eigenvalues)
    return [v / total for v in eigenvalues]


def run_famd(
        df_active: pd.DataFrame,
        quant_vars: List[str],
        qual_vars: List[str],
        n_components: Optional[int] = None,
        famd_cfg: dict = None,
        optimize: bool = False,
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
    optimize : bool
        Si ``True`` et ``n_components`` est ``None``, choisit automatiquement le
        nombre d'axes en fonction de la variance expliquée (seuil 90 %).

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

    n_comp = n_components

    if optimize and n_components is None:
        # Exploration avec le nombre maximal de composantes pour estimer la variance
        n_init = df_for_famd.shape[1]
        tmp = prince.FAMD(
            n_components=n_init,
            n_iter=3,
            copy=True,
            check_input=True,
            engine='sklearn'
        ).fit(df_for_famd)
        inertia_tmp = get_explained_inertia(tmp)
        cum = np.cumsum(inertia_tmp)
        thresh = 0.9
        n_comp = next((i + 1 for i, v in enumerate(cum) if v >= thresh), n_init)
        logger.info(
            "FAMD auto: %d composantes retenues (%.1f%% variance cumulée)",
            n_comp,
            cum[n_comp - 1] * 100
        )

    n_comp = n_comp or df_for_famd.shape[1]

    # 3b) Initialisation de l’AFDM
    famd = prince.FAMD(
        n_components=n_comp,
        n_iter=3,
        copy=True,
        check_input=True,
        # normalize = (weighting == 'balanced'),
        engine='sklearn'
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
        output_dir: Path,
        df_active: Optional[pd.DataFrame] = None,
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
    output_dir.mkdir(parents=True, exist_ok=True)

    # Harmonise les noms des axes
    axes_names = [f"F{i + 1}" for i in range(row_coords.shape[1])]
    row_coords = row_coords.copy()
    row_coords.columns = axes_names
    col_coords = col_coords.copy()
    col_coords.columns = axes_names
    col_contrib = col_contrib.copy()
    col_contrib.columns = axes_names

    # ─── Éboulis ──────────────────────────────────────────────────────
    ax_idx = list(range(1, len(inertia) + 1))
    plt.figure(figsize=(12, 6), dpi=200)
    plt.bar(ax_idx, [v * 100 for v in inertia], edgecolor="black")
    plt.xlabel("Composante")
    plt.ylabel("% Inertie expliquée")
    plt.title("Éboulis FAMD")
    plt.xticks(ax_idx)
    plt.tight_layout()
    plt.savefig(output_dir / "famd_scree_plot.png")
    plt.close()
    logger.info("Scree plot enregistré")

    # ─── Projection individus 2D ──────────────────────────────────────
    if {"F1", "F2"}.issubset(row_coords.columns):
        plt.figure(figsize=(12, 6), dpi=200)
        if df_active is not None and "Statut commercial" in df_active.columns:
            codes = df_active.loc[row_coords.index, "Statut commercial"].astype(
                "category"
            ).cat.codes
        else:
            codes = None
        scatter = plt.scatter(
            row_coords["F1"],
            row_coords["F2"],
            c=codes,
            s=10,
            alpha=0.7,
        )
        plt.xlabel("F1")
        plt.ylabel("F2")
        plt.title("FAMD – individus (F1 vs F2)")
        if codes is not None:
            plt.colorbar(scatter, label="Statut commercial")
        plt.tight_layout()
        plt.savefig(output_dir / "famd_indiv_plot.png")
        plt.close()
        logger.info("Projection F1-F2 enregistrée")

    # ─── Projection individus 3D ──────────────────────────────────────
    if {"F1", "F2", "F3"}.issubset(row_coords.columns):
        fig = plt.figure(figsize=(12, 6), dpi=200)
        ax = fig.add_subplot(111, projection="3d")
        if df_active is not None and "Statut commercial" in df_active.columns:
            codes = df_active.loc[row_coords.index, "Statut commercial"].astype(
                "category"
            ).cat.codes
        else:
            codes = None
        sc = ax.scatter(
            row_coords["F1"],
            row_coords["F2"],
            row_coords["F3"],
            c=codes,
            s=10,
            alpha=0.7,
        )
        ax.set_xlabel("F1")
        ax.set_ylabel("F2")
        ax.set_zlabel("F3")
        ax.set_title("FAMD – individus (3D)")
        if codes is not None:
            fig.colorbar(sc, label="Statut commercial")
        plt.tight_layout()
        plt.savefig(output_dir / "famd_indiv_plot_3D.png")
        plt.close()
        logger.info("Projection 3D enregistrée")

    # ─── Cercle des corrélations ─────────────────────────────────────
    if {"F1", "F2"}.issubset(col_coords.columns):
        qcoords = col_coords.loc[quant_vars].dropna(how="any")
        plt.figure(figsize=(12, 6), dpi=200)
        circle = plt.Circle((0, 0), 1, color="grey", fill=False)
        ax = plt.gca()
        ax.add_patch(circle)
        ax.axhline(0, color="grey", lw=0.5)
        ax.axvline(0, color="grey", lw=0.5)
        for var in qcoords.index:
            x, y = qcoords.loc[var, ["F1", "F2"]]
            ax.arrow(0, 0, x, y, head_width=0.02, length_includes_head=True)
            ax.text(x, y, var, fontsize=8)
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)
        ax.set_xlabel("F1")
        ax.set_ylabel("F2")
        ax.set_title("FAMD – cercle des corrélations (F1–F2)")
        ax.set_aspect("equal")
        plt.tight_layout()
        plt.savefig(output_dir / "famd_correlation_circle.png")
        plt.close()
        logger.info("Cercle des corrélations enregistré")

    # ─── Modalités qualitatives ──────────────────────────────────────
    if {"F1", "F2"}.issubset(col_coords.columns):
        modalities = col_coords.drop(index=quant_vars, errors="ignore")
        plt.figure(figsize=(12, 6), dpi=200)
        plt.scatter(modalities["F1"], modalities["F2"], marker="o", alpha=0.7)
        for mod in modalities.index:
            plt.text(
                modalities.loc[mod, "F1"],
                modalities.loc[mod, "F2"],
                str(mod),
                fontsize=8,
            )
        plt.xlabel("F1")
        plt.ylabel("F2")
        plt.title("FAMD – modalités (F1–F2)")
        plt.tight_layout()
        plt.savefig(output_dir / "famd_modalities_plot.png")
        plt.close()
        logger.info("Modalités enregistrées")

    # ─── Contributions ───────────────────────────────────────────────
    contrib = col_contrib * 100
    fig, axes_plot = plt.subplots(1, 2, figsize=(12, 6), dpi=200)
    for i, axis in enumerate(["F1", "F2"]):
        if axis in contrib.columns:
            top = contrib[axis].sort_values(ascending=False).head(10)
            axes_plot[i].bar(top.index.astype(str), top.values)
            axes_plot[i].set_title(f"Contributions FAMD – Axe {axis[-1]}")
            axes_plot[i].set_ylabel("% contribution")
            axes_plot[i].tick_params(axis="x", rotation=45)
        else:
            axes_plot[i].axis("off")
    plt.tight_layout()
    plt.savefig(output_dir / "famd_contributions.png")
    plt.close()
    logger.info("Contributions enregistrées")

    # ─── Exports CSV ─────────────────────────────────────────────────
    var_df = pd.DataFrame({
        "axe": axes_names,
        "variance_%": [v * 100 for v in inertia],
    })
    var_df["variance_cum_%"] = var_df["variance_%"].cumsum()
    var_df.to_csv(output_dir / "famd_explained_variance.csv", index=False)
    logger.info("CSV variance expliqué enregistré")

    row_coords.to_csv(output_dir / "famd_indiv_coords.csv", index=True)
    logger.info("CSV coordonnées individus enregistré")

    vars_coords = col_coords.loc[quant_vars]
    vars_coords.to_csv(output_dir / "famd_variables_coords.csv", index=True)
    logger.info("CSV coordonnées variables enregistré")

    mods_coords = col_coords.drop(index=quant_vars, errors="ignore")
    mods_coords.to_csv(output_dir / "famd_modalities_coords.csv", index=True)
    logger.info("CSV coordonnées modalités enregistré")

    contrib.to_csv(output_dir / "famd_contributions.csv", index=True)
    logger.info("CSV contributions enregistré")

    logger.info("Export des résultats FAMD terminé")


def export_mfa_results(
        mfa_model,
        row_coords: pd.DataFrame,
        output_dir: Path,
        quant_vars: List[str],
        qual_vars: List[str],
        df_active: Optional[pd.DataFrame] = None,
):
    """Exports figures and CSV results for MFA."""
    logger = logging.getLogger(__name__)
    output_dir.mkdir(parents=True, exist_ok=True)

    axes_names = [f"F{i + 1}" for i in range(row_coords.shape[1])]
    row_coords = row_coords.copy()
    row_coords.columns = axes_names

    col_coords = getattr(mfa_model, "column_coordinates_", pd.DataFrame()).copy()
    if not col_coords.empty:
        col_coords.columns = axes_names[:col_coords.shape[1]]

    col_contrib = getattr(mfa_model, "column_contributions_", pd.DataFrame()).copy()
    if col_contrib.empty and not col_coords.empty:
        tmp = col_coords ** 2
        col_contrib = tmp.div(tmp.sum(axis=0), axis=1)
    if not col_contrib.empty:
        col_contrib.columns = axes_names[:col_contrib.shape[1]]

    # ─── Projection individus 2D ──────────────────────────────────────
    if {"F1", "F2"}.issubset(row_coords.columns):
        plt.figure(figsize=(12, 6), dpi=200)
        if df_active is not None and "Statut commercial" in df_active.columns:
            codes = df_active.loc[row_coords.index, "Statut commercial"].astype(
                "category"
            ).cat.codes
        else:
            codes = None
        sc = plt.scatter(
            row_coords["F1"], row_coords["F2"], c=codes, s=10, alpha=0.7
        )
        plt.xlabel("F1")
        plt.ylabel("F2")
        plt.title("MFA – individus (F1–F2)")
        if codes is not None:
            plt.colorbar(sc, label="Statut commercial")
        plt.tight_layout()
        plt.savefig(output_dir / "mfa_indiv_plot.png")
        plt.close()
        logger.info("Projection MFA F1-F2 enregistrée")

    # ─── Projection individus 3D ──────────────────────────────────────
    if {"F1", "F2", "F3"}.issubset(row_coords.columns):
        fig = plt.figure(figsize=(12, 6), dpi=200)
        ax = fig.add_subplot(111, projection="3d")
        if df_active is not None and "Statut commercial" in df_active.columns:
            codes = df_active.loc[row_coords.index, "Statut commercial"].astype(
                "category"
            ).cat.codes
        else:
            codes = None
        sc = ax.scatter(
            row_coords["F1"],
            row_coords["F2"],
            row_coords["F3"],
            c=codes,
            s=10,
            alpha=0.7,
        )
        ax.set_xlabel("F1")
        ax.set_ylabel("F2")
        ax.set_zlabel("F3")
        ax.set_title("MFA – individus (3D)")
        if codes is not None:
            fig.colorbar(sc, label="Statut commercial")
        plt.tight_layout()
        plt.savefig(output_dir / "mfa_indiv_plot_3D.png")
        plt.close()
        logger.info("Projection MFA 3D enregistrée")

    # ─── Cercle des corrélations ─────────────────────────────────────
    if {"F1", "F2"}.issubset(col_coords.columns):
        qcoords = col_coords.loc[[v for v in quant_vars if v in col_coords.index]]
        if not qcoords.empty:
            plt.figure(figsize=(12, 6), dpi=200)
            circle = plt.Circle((0, 0), 1, color="grey", fill=False)
            ax = plt.gca()
            ax.add_patch(circle)
            ax.axhline(0, color="grey", lw=0.5)
            ax.axvline(0, color="grey", lw=0.5)
            for var in qcoords.index:
                x, y = qcoords.loc[var, ["F1", "F2"]]
                ax.arrow(0, 0, x, y, head_width=0.02, length_includes_head=True)
                ax.text(x, y, var, fontsize=8)
            ax.set_xlim(-1.1, 1.1)
            ax.set_ylim(-1.1, 1.1)
            ax.set_xlabel("F1")
            ax.set_ylabel("F2")
            ax.set_title("MFA – cercle des corrélations")
            ax.set_aspect("equal")
            plt.tight_layout()
            plt.savefig(output_dir / "mfa_correlation_circle.png")
            plt.close()
            logger.info("Cercle des corrélations MFA enregistré")

    # ─── Modalités qualitatives ──────────────────────────────────────
    if {"F1", "F2"}.issubset(col_coords.columns):
        modalities = col_coords.drop(index=[v for v in quant_vars if v in col_coords.index], errors="ignore")
        if not modalities.empty:
            plt.figure(figsize=(12, 6), dpi=200)
            plt.scatter(modalities["F1"], modalities["F2"], marker="o", alpha=0.7)
            for mod in modalities.index:
                label = mod
                if "_" in mod:
                    var, val = mod.split("_", 1)
                    label = f"{var}={val}"
                plt.text(modalities.loc[mod, "F1"], modalities.loc[mod, "F2"], label, fontsize=8)
            plt.xlabel("F1")
            plt.ylabel("F2")
            plt.title("MFA – modalités (F1–F2)")
            plt.tight_layout()
            plt.savefig(output_dir / "mfa_modalities_plot.png")
            plt.close()
            logger.info("Modalités MFA enregistrées")

    # ─── Contributions ───────────────────────────────────────────────
    contrib = col_contrib * 100
    if not contrib.empty:
        fig, axes_plot = plt.subplots(1, 2, figsize=(12, 6), dpi=200)
        for i, axis in enumerate(["F1", "F2"]):
            if axis in contrib.columns:
                top = contrib[axis].sort_values(ascending=False).head(10)
                axes_plot[i].bar(top.index.astype(str), top.values)
                axes_plot[i].set_title(f"Contributions MFA – Axe {axis[-1]}")
                axes_plot[i].set_ylabel("% contribution")
                axes_plot[i].tick_params(axis="x", rotation=45)
            else:
                axes_plot[i].axis("off")
        plt.tight_layout()
        plt.savefig(output_dir / "mfa_contributions.png")
        plt.close()
        logger.info("Contributions MFA enregistrées")

    # ─── Exports CSV ─────────────────────────────────────────────────
    var_df = pd.DataFrame({
        "axe": axes_names,
        "variance_%": [v * 100 for v in mfa_model.explained_inertia_],
    })
    var_df["variance_cum_%"] = var_df["variance_%"].cumsum()
    var_df.to_csv(output_dir / "mfa_explained_variance.csv", index=False)
    logger.info("CSV variance expliqué MFA enregistré")

    row_coords.to_csv(output_dir / "mfa_indiv_coords.csv", index=True)
    logger.info("CSV coordonnées individus MFA enregistré")

    if not col_coords.empty:
        vars_coords = col_coords.loc[[v for v in quant_vars if v in col_coords.index]]
        vars_coords.to_csv(output_dir / "mfa_variables_coords.csv", index=True)
        logger.info("CSV coordonnées variables MFA enregistré")

        mods_coords = col_coords.drop(index=[v for v in quant_vars if v in col_coords.index], errors="ignore")
        mods_coords.to_csv(output_dir / "mfa_modalities_coords.csv", index=True)
        logger.info("CSV coordonnées modalités MFA enregistré")

    if not contrib.empty:
        contrib.to_csv(output_dir / "mfa_contributions.csv", index=True)
        logger.info("CSV contributions MFA enregistré")

    logger.info("Export des résultats MFA terminé")


def export_pca_results(
        pca_model,
        inertia: pd.Series,
        row_coords: pd.DataFrame,
        col_coords: pd.DataFrame,
        output_dir: Path,
        quant_vars: List[str],
        df_active: Optional[pd.DataFrame] = None,
) -> None:
    """Exporte figures et CSV pour l'ACP."""
    logger = logging.getLogger(__name__)
    output_dir.mkdir(parents=True, exist_ok=True)

    axes = [f"F{i + 1}" for i in range(row_coords.shape[1])]
    row_coords = row_coords.copy(); row_coords.columns = axes
    col_coords = col_coords.copy(); col_coords.columns = axes[:col_coords.shape[1]]

    contrib = (col_coords ** 2).div((col_coords ** 2).sum(axis=0), axis=1) * 100

    # Scree plot
    ax_idx = list(range(1, len(inertia) + 1))
    plt.figure(figsize=(12, 6), dpi=200)
    plt.bar(ax_idx, [v * 100 for v in inertia], edgecolor="black")
    plt.plot(ax_idx, np.cumsum(inertia) * 100, "-o", color="orange")
    plt.xlabel("Composante")
    plt.ylabel("% Inertie expliquée")
    plt.title("Éboulis PCA")
    plt.xticks(ax_idx)
    plt.tight_layout()
    plt.savefig(output_dir / "pca_scree_plot.png")
    plt.close()

    # Individuals 2D
    if {"F1", "F2"}.issubset(row_coords.columns):
        plt.figure(figsize=(12, 6), dpi=200)
        if df_active is not None and "Statut commercial" in df_active.columns:
            codes = df_active.loc[row_coords.index, "Statut commercial"].astype("category").cat.codes
        else:
            codes = None
        sc = plt.scatter(row_coords["F1"], row_coords["F2"], c=codes, s=10, alpha=0.7)
        plt.xlabel("F1"); plt.ylabel("F2")
        plt.title("PCA – individus (F1–F2)")
        if codes is not None:
            plt.colorbar(sc, label="Statut commercial")
        plt.tight_layout()
        plt.savefig(output_dir / "pca_indiv_plot.png")
        plt.close()

    # Individuals 3D
    if {"F1", "F2", "F3"}.issubset(row_coords.columns):
        fig = plt.figure(figsize=(12, 6), dpi=200)
        ax = fig.add_subplot(111, projection="3d")
        if df_active is not None and "Statut commercial" in df_active.columns:
            codes = df_active.loc[row_coords.index, "Statut commercial"].astype("category").cat.codes
        else:
            codes = None
        sc3 = ax.scatter(row_coords["F1"], row_coords["F2"], row_coords["F3"], c=codes, s=10, alpha=0.7)
        ax.set_xlabel("F1"); ax.set_ylabel("F2"); ax.set_zlabel("F3")
        ax.set_title("PCA – individus (3D)")
        if codes is not None:
            fig.colorbar(sc3, label="Statut commercial")
        plt.tight_layout()
        plt.savefig(output_dir / "pca_indiv_plot_3D.png")
        plt.close()

    # Correlation circle
    if {"F1", "F2"}.issubset(col_coords.columns):
        plt.figure(figsize=(12, 6), dpi=200)
        circle = plt.Circle((0, 0), 1, color="grey", fill=False)
        ax = plt.gca(); ax.add_patch(circle)
        ax.axhline(0, color="grey", lw=0.5); ax.axvline(0, color="grey", lw=0.5)
        for var in col_coords.index:
            x, y = col_coords.loc[var, ["F1", "F2"]]
            ax.arrow(0, 0, x, y, head_width=0.02, length_includes_head=True)
            ax.text(x, y, var, fontsize=8)
        ax.set_xlim(-1.1, 1.1); ax.set_ylim(-1.1, 1.1)
        ax.set_xlabel("F1"); ax.set_ylabel("F2")
        ax.set_title("PCA – cercle des corrélations (F1–F2)")
        ax.set_aspect("equal")
        plt.tight_layout()
        plt.savefig(output_dir / "pca_correlation_circle.png")
        plt.close()

    # Contributions
    fig, axes_plot = plt.subplots(1, 2, figsize=(12, 6), dpi=200)
    for i, axis in enumerate(["F1", "F2"]):
        if axis in contrib.columns:
            top = contrib[axis].sort_values(ascending=False).head(10)
            axes_plot[i].bar(top.index.astype(str), top.values)
            axes_plot[i].set_title(f"Contributions PCA – Axe {axis[-1]}")
            axes_plot[i].set_ylabel("% contribution")
            axes_plot[i].tick_params(axis="x", rotation=45)
        else:
            axes_plot[i].axis("off")
    plt.tight_layout()
    plt.savefig(output_dir / "pca_contributions.png")
    plt.close()

    # CSV exports
    var_df = pd.DataFrame({"axe": axes, "variance_%": [v * 100 for v in inertia]})
    var_df["variance_cum_%"] = var_df["variance_%"].cumsum()
    var_df.to_csv(output_dir / "pca_explained_variance.csv", index=False)
    row_coords.to_csv(output_dir / "pca_indiv_coords.csv", index=True)
    col_coords.to_csv(output_dir / "pca_variables_coords.csv", index=True)
    contrib.to_csv(output_dir / "pca_contributions.csv", index=True)
    logger.info("Export PCA terminé")


def export_mca_results(
        mca_model,
        inertia: pd.Series,
        row_coords: pd.DataFrame,
        col_coords: pd.DataFrame,
        output_dir: Path,
        qual_vars: List[str],
        df_active: Optional[pd.DataFrame] = None,
) -> None:
    """Exporte figures et CSV pour l'ACM."""
    logger = logging.getLogger(__name__)
    output_dir.mkdir(parents=True, exist_ok=True)

    axes = [f"F{i + 1}" for i in range(row_coords.shape[1])]
    row_coords = row_coords.copy(); row_coords.columns = axes
    col_coords = col_coords.copy(); col_coords.columns = axes[:col_coords.shape[1]]

    contrib = (col_coords ** 2).div((col_coords ** 2).sum(axis=0), axis=1) * 100

    # Scree
    ax_idx = list(range(1, len(inertia) + 1))
    plt.figure(figsize=(12, 6), dpi=200)
    plt.bar(ax_idx, [v * 100 for v in inertia], edgecolor="black")
    plt.plot(ax_idx, np.cumsum(inertia) * 100, "-o", color="orange")
    plt.xlabel("Composante")
    plt.ylabel("% Inertie expliquée")
    plt.title("Éboulis MCA")
    plt.xticks(ax_idx)
    plt.tight_layout()
    plt.savefig(output_dir / "mca_scree_plot.png")
    plt.close()

    # Individuals 2D
    if {"F1", "F2"}.issubset(row_coords.columns):
        plt.figure(figsize=(12, 6), dpi=200)
        if df_active is not None and "Statut commercial" in df_active.columns:
            codes = df_active.loc[row_coords.index, "Statut commercial"].astype("category").cat.codes
        else:
            codes = None
        sc = plt.scatter(row_coords["F1"], row_coords["F2"], c=codes, s=10, alpha=0.7)
        plt.xlabel("F1"); plt.ylabel("F2")
        plt.title("MCA – individus (F1–F2)")
        if codes is not None:
            plt.colorbar(sc, label="Statut commercial")
        plt.tight_layout()
        plt.savefig(output_dir / "mca_indiv_plot.png")
        plt.close()

    # Individuals 3D
    if {"F1", "F2", "F3"}.issubset(row_coords.columns):
        fig = plt.figure(figsize=(12, 6), dpi=200)
        ax = fig.add_subplot(111, projection="3d")
        if df_active is not None and "Statut commercial" in df_active.columns:
            codes = df_active.loc[row_coords.index, "Statut commercial"].astype("category").cat.codes
        else:
            codes = None
        sc3 = ax.scatter(row_coords["F1"], row_coords["F2"], row_coords["F3"], c=codes, s=10, alpha=0.7)
        ax.set_xlabel("F1"); ax.set_ylabel("F2"); ax.set_zlabel("F3")
        ax.set_title("MCA – individus (3D)")
        if codes is not None:
            fig.colorbar(sc3, label="Statut commercial")
        plt.tight_layout()
        plt.savefig(output_dir / "mca_indiv_plot_3D.png")
        plt.close()

    # Modalities
    if {"F1", "F2"}.issubset(col_coords.columns):
        plt.figure(figsize=(12, 6), dpi=200)
        plt.scatter(col_coords["F1"], col_coords["F2"], alpha=0.7)
        for mod in col_coords.index:
            label = mod
            if "_" in mod:
                var, val = mod.split("_", 1)
                label = f"{var}={val}"
            plt.text(col_coords.loc[mod, "F1"], col_coords.loc[mod, "F2"], label, fontsize=8)
        plt.xlabel("F1"); plt.ylabel("F2")
        plt.title("MCA – modalités (F1–F2)")
        plt.tight_layout()
        plt.savefig(output_dir / "mca_modalities_plot.png")
        plt.close()

    # Contributions
    fig, axes_plot = plt.subplots(1, 2, figsize=(12, 6), dpi=200)
    for i, axis in enumerate(["F1", "F2"]):
        if axis in contrib.columns:
            top = contrib[axis].sort_values(ascending=False).head(10)
            axes_plot[i].bar(top.index.astype(str), top.values)
            axes_plot[i].set_title(f"Contributions MCA – Axe {axis[-1]}")
            axes_plot[i].set_ylabel("% contribution")
            axes_plot[i].tick_params(axis="x", rotation=45)
        else:
            axes_plot[i].axis("off")
    plt.tight_layout()
    plt.savefig(output_dir / "mca_contributions.png")
    plt.close()

    # CSV
    var_df = pd.DataFrame({"axe": axes, "variance_%": [v * 100 for v in inertia]})
    var_df["variance_cum_%"] = var_df["variance_%"].cumsum()
    var_df.to_csv(output_dir / "mca_explained_variance.csv", index=False)
    row_coords.to_csv(output_dir / "mca_indiv_coords.csv", index=True)
    col_coords.to_csv(output_dir / "mca_modalities_coords.csv", index=True)
    contrib.to_csv(output_dir / "mca_contributions.csv", index=True)
    logger.info("Export MCA terminé")



def export_pcamix_results(
        mdpca_model,
        mdpca_inertia: pd.Series,
        row_coords: pd.DataFrame,
        col_coords: pd.DataFrame,
        output_dir: Path,
        quant_vars: List[str],
        qual_vars: List[str],
        df_active: Optional[pd.DataFrame] = None,
) -> None:
    """Exports figures and CSV results for PCAmix."""
    logger = logging.getLogger(__name__)
    output_dir.mkdir(parents=True, exist_ok=True)

    axes = [f"F{i + 1}" for i in range(row_coords.shape[1])]
    row_coords = row_coords.copy()
    row_coords.columns = axes
    col_coords = col_coords.copy()
    col_coords.columns = axes[:col_coords.shape[1]]

    contrib = (col_coords ** 2)
    contrib = contrib.div(contrib.sum(axis=0), axis=1) * 100

    # ─── Projection individus 2D ──────────────────────────────────────
    if {"F1", "F2"}.issubset(row_coords.columns):
        plt.figure(figsize=(12, 6), dpi=200)
        if df_active is not None and "Statut commercial" in df_active.columns:
            codes = df_active.loc[row_coords.index, "Statut commercial"].astype(
                "category"
            ).cat.codes
        else:
            codes = None
        sc = plt.scatter(
            row_coords["F1"], row_coords["F2"], c=codes, s=10, alpha=0.7
        )
        plt.xlabel("F1")
        plt.ylabel("F2")
        plt.title("PCAmix – individus (F1–F2)")
        if codes is not None:
            plt.colorbar(sc, label="Statut commercial")
        plt.tight_layout()
        plt.savefig(output_dir / "pcamix_indiv_plot.png")
        plt.close()
        logger.info("Projection individus PCAmix F1-F2 enregistrée")

    # ─── Projection individus 3D ──────────────────────────────────────
    if {"F1", "F2", "F3"}.issubset(row_coords.columns):
        fig = plt.figure(figsize=(12, 6), dpi=200)
        ax = fig.add_subplot(111, projection="3d")
        if df_active is not None and "Statut commercial" in df_active.columns:
            codes = df_active.loc[row_coords.index, "Statut commercial"].astype(
                "category"
            ).cat.codes
        else:
            codes = None
        sc = ax.scatter(
            row_coords["F1"],
            row_coords["F2"],
            row_coords["F3"],
            c=codes,
            s=10,
            alpha=0.7,
        )
        ax.set_xlabel("F1")
        ax.set_ylabel("F2")
        ax.set_zlabel("F3")
        ax.set_title("PCAmix – individus (3D)")
        if codes is not None:
            fig.colorbar(sc, label="Statut commercial")
        plt.tight_layout()
        plt.savefig(output_dir / "pcamix_indiv_plot_3D.png")
        plt.close()
        logger.info("Projection individus PCAmix 3D enregistrée")

    # ─── Cercle des corrélations ─────────────────────────────────────
    if {"F1", "F2"}.issubset(col_coords.columns):
        qcoords = col_coords.loc[[v for v in quant_vars if v in col_coords.index]]
        if not qcoords.empty:
            plt.figure(figsize=(12, 6), dpi=200)
            circle = plt.Circle((0, 0), 1, color="grey", fill=False)
            ax = plt.gca()
            ax.add_patch(circle)
            ax.axhline(0, color="grey", lw=0.5)
            ax.axvline(0, color="grey", lw=0.5)
            for var in qcoords.index:
                x, y = qcoords.loc[var, ["F1", "F2"]]
                ax.arrow(0, 0, x, y, head_width=0.02, length_includes_head=True)
                ax.text(x, y, var, fontsize=8)
            ax.set_xlim(-1.1, 1.1)
            ax.set_ylim(-1.1, 1.1)
            ax.set_xlabel("F1")
            ax.set_ylabel("F2")
            ax.set_title("PCAmix – cercle des corrélations")
            ax.set_aspect("equal")
            plt.tight_layout()
            plt.savefig(output_dir / "pcamix_correlation_circle.png")
            plt.close()
            logger.info("Cercle des corrélations PCAmix enregistré")

    # ─── Modalités qualitatives ──────────────────────────────────────
    if {"F1", "F2"}.issubset(col_coords.columns):
        modalities = col_coords.drop(index=[v for v in quant_vars if v in col_coords.index], errors="ignore")
        if not modalities.empty:
            plt.figure(figsize=(12, 6), dpi=200)
            plt.scatter(modalities["F1"], modalities["F2"], marker="o", alpha=0.7)
            for mod in modalities.index:
                label = mod
                if "_" in mod:
                    var, val = mod.split("_", 1)
                    label = f"{var}={val}"
                plt.text(modalities.loc[mod, "F1"], modalities.loc[mod, "F2"], label, fontsize=8)
            plt.xlabel("F1")
            plt.ylabel("F2")
            plt.title("PCAmix – modalités (F1–F2)")
            plt.tight_layout()
            plt.savefig(output_dir / "pcamix_modalities_plot.png")
            plt.close()
            logger.info("Modalités PCAmix enregistrées")

    # ─── Contributions ───────────────────────────────────────────────
    fig, axes_plot = plt.subplots(1, 2, figsize=(12, 6), dpi=200)
    for i, axis in enumerate(["F1", "F2"]):
        if axis in contrib.columns:
            top = contrib[axis].sort_values(ascending=False).head(10)
            axes_plot[i].bar(top.index.astype(str), top.values)
            axes_plot[i].set_title(f"Contributions PCAmix – Axe {axis[-1]}")
            axes_plot[i].set_ylabel("% contribution")
            axes_plot[i].tick_params(axis="x", rotation=45)
        else:
            axes_plot[i].axis("off")
    plt.tight_layout()
    plt.savefig(output_dir / "pcamix_contributions.png")
    plt.close()
    logger.info("Contributions PCAmix enregistrées")

    # ─── Exports CSV ─────────────────────────────────────────────────
    var_df = pd.DataFrame({
        "axe": axes,
        "variance_%": [v * 100 for v in mdpca_inertia],
    })
    var_df["variance_cum_%"] = var_df["variance_%"].cumsum()
    var_df.to_csv(output_dir / "pcamix_explained_variance.csv", index=False)
    logger.info("CSV variance expliqué PCAmix enregistré")

    row_coords.to_csv(output_dir / "pcamix_indiv_coords.csv", index=True)
    logger.info("CSV coordonnées individus PCAmix enregistré")

    vars_coords = col_coords.loc[[v for v in quant_vars if v in col_coords.index]]
    vars_coords.to_csv(output_dir / "pcamix_variables_coords.csv", index=True)
    logger.info("CSV coordonnées variables PCAmix enregistré")

    mods_coords = col_coords.drop(index=[v for v in quant_vars if v in col_coords.index], errors="ignore")
    mods_coords.to_csv(output_dir / "pcamix_modalities_coords.csv", index=True)
    logger.info("CSV coordonnées modalités PCAmix enregistré")

    contrib.to_csv(output_dir / "pcamix_contributions.csv", index=True)
    logger.info("CSV contributions PCAmix enregistré")

    logger.info("Export des résultats PCAmix terminé")


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

        for cj in unique[i + 1:]:
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
        df_active: pd.DataFrame | None = None,
        quant_vars: Sequence[str] | None = None,
        qual_vars: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Compare diverses méthodes de réduction de dimension."""

    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from sklearn.manifold import trustworthiness
    import seaborn as sns
    from sklearn.preprocessing import StandardScaler, OneHotEncoder

    rows = []
    for method, info in results_dict.items():
        inertias = info.get("inertia", [])
        if inertias is None:
            inertias = []
        if isinstance(inertias, pd.Series):
            inertias = inertias.tolist()
        kaiser = sum(1 for eig in inertias if eig > 1)
        cum_inertia = sum(inertias) if len(inertias) > 0 else None

        X = info["embeddings"].values
        labels = KMeans(n_clusters=n_clusters, random_state=0).fit_predict(X)
        sil = silhouette_score(X, labels)
        dunn = dunn_index(X, labels)
        runtime = info.get("runtime")

        T = None
        C = None
        if df_active is not None and quant_vars is not None and qual_vars is not None:
            # Build the high-dimensional representation aligned on the embedding index
            try:
                enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
            except TypeError:  # pragma: no cover - older scikit-learn
                enc = OneHotEncoder(sparse=False, handle_unknown="ignore")
            X_num = StandardScaler().fit_transform(df_active.loc[info["embeddings"].index, quant_vars])
            X_cat = enc.fit_transform(df_active.loc[info["embeddings"].index, qual_vars])
            X_high = np.hstack([X_num, X_cat])
            X_low = X
            T = trustworthiness(X_high, X_low, n_neighbors=10)
            C = trustworthiness(X_low, X_high, n_neighbors=10)

        rows.append(
            {
                "method": method,
                "kaiser": kaiser,
                "cum_inertia": cum_inertia,
                "silhouette": sil,
                "dunn": dunn,
                "trustworthiness": T,
                "continuity": C,
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

    plt.figure(figsize=(12, 6), dpi=200)
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

    import copy
    effective_config = copy.deepcopy(config)

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

    optimize_params = config.get("optimize_params", False)
    results: Dict[str, Dict[str, Any]] = {}

    # 2. FAMD
    if "famd" in config.get("methods", []):
        t0 = time.time()
        famd_model, famd_inertia, famd_rows, famd_cols, famd_contrib = run_famd(
            df_active,
            quant_vars,
            qual_vars,
            optimize=optimize_params,
            **config.get("famd", {})
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
        effective_config.setdefault("famd", {})["n_components"] = famd_model.n_components
        output_dir_famd = output_dir / "FAMD"
        export_famd_results(
            famd_model,
            famd_inertia,
            famd_rows,
            famd_cols,
            famd_contrib,
            quant_vars,
            qual_vars,
            output_dir_famd,
            df_active=df_active,
        )

    # 3. PCA (quantitatives seulement)
    if "pca" in config.get("methods", []):
        t0 = time.time()
        output_dir_pca = output_dir / "PCA"
        pca_model, pca_inertia, pca_rows, pca_cols, pca_contrib = run_pca(
            df_active,
            quant_vars,
            qual_vars,
            output_dir_pca,
            optimize=optimize_params,
            **config.get("pca", {})
        )
        rt = time.time() - t0
        results["PCA"] = {
            "embeddings": pca_rows,
            "inertia": list(pca_inertia),
            "runtime": rt,
        }
        logger.info(
            "PCA : %d composantes, %.1f%% variance cumulée, %.1fs",
            pca_model.n_components,
            sum(pca_inertia) * 100,
            rt,
        )
        effective_config.setdefault("pca", {})["n_components"] = pca_model.n_components
        export_pca_results(
            pca_model,
            pca_inertia,
            pca_rows,
            pca_cols,
            output_dir_pca,
            quant_vars,
            df_active=df_active,
        )

    # 4. MCA (qualitatives seulement)
    if "mca" in config.get("methods", []):
        t0 = time.time()
        output_dir_mca = output_dir / "MCA"
        mca_model, mca_inertia, mca_rows, mca_cols, mca_contrib = run_mca(
            df_active,
            quant_vars,
            qual_vars,
            output_dir_mca,
            optimize=optimize_params,
            **config.get("mca", {})
        )
        rt = time.time() - t0
        results["MCA"] = {
            "embeddings": mca_rows,
            "inertia": list(mca_inertia),
            "runtime": rt,
        }
        logger.info(
            "MCA : %d dimensions, %.1f%% inertie cumulée, %.1fs",
            mca_model.n_components,
            sum(mca_inertia) * 100,
            rt,
        )
        effective_config.setdefault("mca", {})["n_components"] = mca_model.n_components
        export_mca_results(
            mca_model,
            mca_inertia,
            mca_rows,
            mca_cols,
            output_dir_mca,
            qual_vars,
            df_active=df_active,
        )

    # 5. MFA
    if "mfa" in config.get("methods", []):
        t0 = time.time()
        output_dir_mfa = output_dir / "MFA"
        mfa_model, mfa_rows = run_mfa(
            df_active,
            quant_vars,
            qual_vars,
            output_dir_mfa,
            optimize=optimize_params,
            **config.get("mfa", {})
        )
        rt = time.time() - t0
        results["MFA"] = {
            "embeddings": mfa_rows,
            "inertia": list(mfa_model.explained_inertia_),
            "runtime": rt,
        }
        logger.info(
            "MFA : %d composantes, %.1f%% variance cumulée, %.1fs",
            mfa_model.n_components,
            mfa_model.explained_inertia_.sum() * 100,
            rt,
        )
        effective_config.setdefault("mfa", {})["n_components"] = mfa_model.n_components
        export_mfa_results(
            mfa_model,
            mfa_rows,
            output_dir_mfa,
            quant_vars,
            qual_vars,
            df_active=df_active,
        )

    # 6. PCAmix
    if "pcamix" in config.get("methods", []):
        t0 = time.time()
        output_dir_pcamix = output_dir / "PCAmix"
        mdpca_model, mdpca_inertia, mdpca_rows, mdpca_cols = run_pcamix(
            df_active,
            quant_vars,
            qual_vars,
            output_dir_pcamix,
            optimize=optimize_params,
            **config.get("pcamix", {})
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
        effective_config.setdefault("pcamix", {})["n_components"] = mdpca_model.n_components
        export_pcamix_results(
            mdpca_model,
            mdpca_inertia,
            mdpca_rows,
            mdpca_cols,
            output_dir_pcamix,
            quant_vars,
            qual_vars,
            df_active=df_active,
        )

    # 7. UMAP
    if "umap" in config.get("methods", []):
        t0 = time.time()
        output_dir_umap = output_dir / "UMAP"
        umap_model, umap_df = run_umap(
            df_active,
            quant_vars,
            qual_vars,
            output_dir_umap,
            optimize=optimize_params,
            **config.get("umap", {}),
        )
        export_umap_results(umap_df, df_active, output_dir_umap)
        rt = time.time() - t0
        results["UMAP"] = {
            "embeddings": umap_df,
            "inertia": None,
            "runtime": rt,
        }
        logger.info(
            "UMAP : %d composantes, n_neighbors=%d, min_dist=%.2f, %.1fs",
            umap_model.n_components,
            getattr(umap_model, "n_neighbors", -1),
            getattr(umap_model, "min_dist", 0.0),
            rt,
        )
        effective_config.setdefault("umap", {}).update({
            "n_components": umap_model.n_components,
            "n_neighbors": getattr(umap_model, "n_neighbors", None),
            "min_dist": getattr(umap_model, "min_dist", None),
        })

    # 8. PaCMAP
    if "pacmap" in config.get("methods", []):
        t0 = time.time()
        output_dir_pacmap = output_dir / "PaCMAP"
        pacmap_model, pacmap_df = run_pacmap(
            df_active,
            quant_vars,
            qual_vars,
            output_dir_pacmap,
            **config.get("pacmap", {}),
        )
        if pacmap_model is not None:
            export_pacmap_results(pacmap_df, df_active, output_dir_pacmap)
            rt = time.time() - t0
            results["PaCMAP"] = {
                "embeddings": pacmap_df,
                "inertia": None,
                "runtime": rt,
            }
            logger.info(
                f"PaCMAP : r\u00e9alis\u00e9 en {rt:.1f}s (n_neighbors={pacmap_model.n_neighbors})"
            )
            effective_config.setdefault("pacmap", {}).update({
                "n_components": pacmap_model.n_components,
                "n_neighbors": pacmap_model.n_neighbors,
                "MN_ratio": pacmap_model.MN_ratio,
                "FP_ratio": pacmap_model.FP_ratio,
            })

    # 9. PHATE
    if "phate" in config.get("methods", []):
        t0 = time.time()
        output_dir_phate = output_dir / "PHATE"
        phate_op, phate_df = run_phate(
            df_active,
            quant_vars,
            qual_vars,
            output_dir_phate,
            **config.get("phate", {}),
        )
        if phate_op is not None:
            export_phate_results(phate_df, df_active, output_dir_phate)
            rt = time.time() - t0
            results["PHATE"] = {
                "embeddings": phate_df,
                "inertia": None,
                "runtime": rt,
            }
            logger.info(f"PHATE : r\u00e9alis\u00e9 en {rt:.1f}s")
            effective_config.setdefault("phate", {})["n_components"] = phate_op.n_components

    # 10. t-SNE
    if "tsne" in config.get("methods", []):
        if "FAMD" not in results:
            raise RuntimeError("t-SNE requires FAMD embeddings")
        t0 = time.time()
        output_dir_tsne = output_dir / "TSNE"
        tsne_model, tsne_df = run_tsne(
            results["FAMD"]["embeddings"],
            df_active,
            output_dir_tsne,
            optimize=optimize_params,
            **config.get("tsne", {})
        )
        export_tsne_results(tsne_df, df_active, output_dir_tsne)
        rt = time.time() - t0
        results["TSNE"] = {
            "embeddings": tsne_df,
            "inertia": None,
            "runtime": rt,
        }
        logger.info(
            "t-SNE : perplexity=%s, %.1fs",
            getattr(tsne_model, "perplexity", "?"),
            rt,
        )
        effective_config.setdefault("tsne", {})["perplexity"] = getattr(tsne_model, "perplexity", None)

    # 8. Évaluation croisée
    comp_df = evaluate_methods(
        results,
        output_dir,
        n_clusters=3,
        df_active=df_active,
        quant_vars=quant_vars,
        qual_vars=qual_vars,
    )

    # 9. PDF comparatif
    pdf_path = generate_pdf(output_dir)
    logger.info(f"Rapport PDF généré : {pdf_path}")

    # Sauvegarde de la configuration effective
    cfg_path = output_dir / "phase4_effective_config.json"
    with open(cfg_path, "w", encoding="utf-8") as fh:
        json.dump(effective_config, fh, indent=2, ensure_ascii=False)
    logger.info("Configuration effective exportée : %s", cfg_path.name)

    # 10. Index CSV des fichiers produits
    create_index_file(output_dir)


def generate_report_pdf(output_dir: Path) -> Path:
    """Assemble un PDF de synthèse à partir des figures générées.

    The PDF will include the following images if present in ``output_dir``:
    ``MFA/mfa_scree_plot.png``, ``phase4_pcamix_scree_plot.png``,
    ``UMAP/umap_scatter.png``, ``phase4_tsne_scatter.png`` and
    ``methods_heatmap.png``.
    """
    logger = logging.getLogger(__name__)
    pdf_path = output_dir / "phase4_report.pdf"
    figures = [
        str(Path("MFA") / "mfa_scree_plot.png"),
        "phase4_pcamix_scree_plot.png",
        str(Path("UMAP") / "umap_scatter.png"),
        "phase4_tsne_scatter.png",
        "methods_heatmap.png",
    ]
    with PdfPages(pdf_path) as pdf:
        for name in figures:
            img_path = output_dir / Path(name)
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


def generate_pdf(output_dir: Path, pdf_name: str = "phase4_figures.pdf") -> Path:
    logger = logging.getLogger(__name__)
    """Assemble toutes les figures PNG en un PDF multi-pages classé par méthode."""

    methods = ["FAMD", "PCA", "MCA", "MFA", "PCAmix", "UMAP", "PaCMAP", "PHATE", "TSNE"]
    pdf_path = output_dir / pdf_name

    with PdfPages(pdf_path) as pdf:
        for method in methods:
            m_dir = output_dir / method
            if not m_dir.exists():
                continue

            # Page de titre de la section
            fig, ax = plt.subplots(figsize=(8.27, 11.69), dpi=200)
            ax.text(0.5, 0.5, method, fontsize=24, ha="center", va="center")
            ax.axis("off")
            pdf.savefig(fig, dpi=300)
            plt.close(fig)

            for img_path in sorted(m_dir.glob("*.png")):
                img = plt.imread(img_path)
                fig, ax = plt.subplots(figsize=(12, 6), dpi=200)
                ax.imshow(img)
                ax.axis("off")
                pdf.savefig(fig, dpi=300)
                plt.close(fig)

        # Figures comparatives globales
        global_imgs = [
            "multi_scree.png",
            "multi_scatter.png",
            "multi_heatmap.png",
            "methods_heatmap.png",
        ]
        existing = [output_dir / f for f in global_imgs if (output_dir / f).exists()]
        if existing:
            fig, ax = plt.subplots(figsize=(8.27, 11.69), dpi=200)
            ax.text(
                0.5,
                0.5,
                "Comparaisons multi-méthodes",
                fontsize=24,
                ha="center",
                va="center",
            )
            ax.axis("off")
            pdf.savefig(fig, dpi=300)
            plt.close(fig)

            for img_path in existing:
                img = plt.imread(img_path)
                fig, ax = plt.subplots(figsize=(12, 6), dpi=200)
                ax.imshow(img)
                ax.axis("off")
                pdf.savefig(fig, dpi=300)
                plt.close(fig)

    logger.info(f"PDF des figures Phase 4 généré : {pdf_path.name}")
    return pdf_path


def create_index_file(output_dir: Path) -> Path:
    """Génère un index CSV de tous les fichiers produits en phase 4."""
    logger = logging.getLogger(__name__)

    rows: List[Dict[str, str]] = []

    def guess_description(file_name: str, method: str) -> str:
        name = file_name.lower()
        if name.startswith("segment_"):
            base = Path(file_name).stem[len("segment_"):]
            if file_name.endswith(".csv"):
                return f"Répartition par {base} (données)"
            return f"Répartition par {base}"

        mapping = {
            "scree_plot.png": f"Éboulis {method}",
            "indiv_plot.png": f"Projection {method} 2D (Statut commercial)",
            "indiv_plot_3d.png": f"Projection {method} 3D (Statut commercial)",
            "correlation_circle.png": f"Cercle des corrélations {method}",
            "modalities_plot.png": f"Projection modalités {method}",
            "contributions.png": f"Top contributions {method}",
            "explained_variance.csv": f"Variance expliquée {method}",
            "indiv_coords.csv": f"Coordonnées individus {method}",
            "variables_coords.csv": f"Coordonnées variables {method}",
            "modalities_coords.csv": f"Coordonnées modalités {method}",
            "contributions.csv": f"Contributions {method}",
            "embeddings.csv": f"Embeddings {method}",
            "scatter.png": f"Projection {method} 2D (Statut commercial)",
            "scatter_3d.png": f"Projection {method} 3D (Statut commercial)",
        }
        for key, desc in mapping.items():
            if name.endswith(key):
                return desc

        other = {
            "methods_heatmap.png": "Heatmap comparaison méthodes",
            "methods_comparison.csv": "Tableau comparatif des méthodes",
            "multi_scree.png": "Éboulis comparatif",
            "multi_scatter.png": "Projection comparée des méthodes",
            "multi_heatmap.png": "Heatmap multi-méthodes",
            "phase4_figures.pdf": "PDF de toutes les figures",
            "phase4_report.pdf": "PDF synthèse",
        }
        if file_name in other:
            return other[file_name]

        # Fallback: nom lisible
        desc = Path(file_name).stem
        desc = desc.replace(method.lower() + "_", "")
        desc = desc.replace("phase4_", "")
        desc = desc.replace("_", " ").strip().capitalize()
        if method != "Global" and method not in desc:
            desc = f"{desc} {method}".strip()
        return desc

    for file_path in sorted(output_dir.rglob("*")):
        if not file_path.is_file():
            continue
        if file_path.name in {"phase4.log", "phase4_index.csv"}:
            continue
        ext = file_path.suffix.lower()
        if ext not in {".png", ".csv", ".pdf"}:
            continue

        rel = file_path.relative_to(output_dir)
        method = "Global"
        if len(rel.parts) > 1 and rel.parts[0] in {"FAMD", "PCA", "MCA", "MFA", "PCAmix", "UMAP", "PaCMAP", "PHATE", "TSNE"}:
            method = rel.parts[0]

        file_type = {
            ".png": "figure",
            ".csv": "data",
            ".pdf": "report",
        }[ext]

        desc = guess_description(file_path.name, method)

        rows.append({
            "Méthode": method,
            "Fichier": str(rel),
            "Type": file_type,
            "Description": desc,
        })

    df = pd.DataFrame(rows)
    index_path = output_dir / "phase4_index.csv"
    df.to_csv(index_path, index=False, encoding="utf-8")
    logger.info(f"Index des fichiers exporté -> {index_path.name}")

    txt_path = output_dir / "phase4_output_files.txt"
    with open(txt_path, "w", encoding="utf-8") as fh:
        for rel in sorted(output_dir.rglob("*")):
            if rel.is_file() and rel.name not in {"phase4.log", "phase4_index.csv"}:
                fh.write(str(rel.relative_to(output_dir)) + "\n")
    logger.info("Liste de fichiers enregistrée -> %s", txt_path.name)

    return index_path


if __name__ == "__main__":
    main()
