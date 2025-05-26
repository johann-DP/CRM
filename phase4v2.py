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
import matplotlib
# Use a non-interactive backend to avoid Tkinter cleanup errors in CLI usage
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import ListedColormap
from typing import List, Optional, Tuple, Sequence, Dict, Any
import prince
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.manifold import TSNE
import logging
import os
import numpy as np
import umap
import time
from concurrent.futures import ThreadPoolExecutor
import seaborn as sns
import datetime
import phate
import pacmap
import json
import ast

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

# Principal CRM segmentation columns used to generate variant scatters
SEGMENT_COLUMNS = [
    "Catégorie",
    "Entité opérationnelle",
    "Pilier",
    "Sous-catégorie",
    "Statut commercial",
    "Statut production",
    "Type opportunité",
]


def load_best_params(csv_file: Path) -> Dict[str, Dict[str, Any]]:
    """Load best parameter values from a CSV created by the fine-tuning step."""
    if not csv_file.exists():
        return {}
    df = pd.read_csv(csv_file)
    params: Dict[str, Dict[str, Any]] = {}
    for method, group in df.groupby("method"):
        d: Dict[str, Any] = {}
        for _, row in group.iterrows():
            val = row["value"]
            try:
                val = json.loads(val)
            except Exception:
                try:
                    val = ast.literal_eval(val)
                except Exception:
                    pass
            d[str(row["param"])] = val
        params[method.lower()] = d
    return params


def plot_correlation_circle(
        ax,
        coords: pd.DataFrame,
        title: str,
        colors: Optional[Dict[str, Any]] = None,
) -> None:
    """Plot a correlation circle with slight label offset to reduce overlap."""
    circle = plt.Circle((0, 0), 1, color="grey", fill=False, linestyle="dashed")
    ax.add_patch(circle)
    ax.axhline(0, color="grey", lw=0.5)
    ax.axvline(0, color="grey", lw=0.5)
    for var in coords.index:
        x, y = coords.loc[var, ["F1", "F2"]]
        color = colors.get(var) if colors else None
        ax.arrow(0, 0, x, y, head_width=0.02, length_includes_head=True, color=color)
        offset_x = x * 1.15 + 0.03 * np.sign(x)
        offset_y = y * 1.15 + 0.03 * np.sign(y)
        ax.text(offset_x, offset_y, var, fontsize=8, ha="center", va="center", color=color)
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_xlabel("F1")
    ax.set_ylabel("F2")
    ax.set_title(title)
    ax.set_aspect("equal")


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
    plt.savefig(output_path, dpi=200)
    plt.close()


def load_data(file_path: str) -> pd.DataFrame:
    """Charge les données CRM brutes depuis un fichier.

    Le fichier peut être au format Excel (``.xls`` ou ``.xlsx``) ou CSV.
    Quelques contrôles de base sont effectués après la lecture.

    Parameters
    ----------
    file_path : str
        Chemin vers le fichier d'export Everwin.

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
    path = Path(file_path)
    try:
        if path.suffix.lower() == ".csv":
            df = pd.read_csv(file_path)
        else:
            df = pd.read_excel(file_path, engine="openpyxl")
        logger.info(
            "Données chargées : %s lignes × %s colonnes",
            df.shape[0],
            df.shape[1],
        )
    except FileNotFoundError:
        logger.error("Fichier introuvable : %s", file_path)
        raise
    except Exception as e:
        logger.error("Erreur lors du chargement de '%s' : %s", file_path, e)
        raise ValueError(f"Impossible de charger le fichier : {e}") from e

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


def handle_missing_values(
    df: pd.DataFrame, quant_vars: List[str], qual_vars: List[str]
) -> pd.DataFrame:
    """Impute missing values and remove remaining invalid entries."""

    logger = logging.getLogger(__name__)

    # Replace infinite values by NA so they can be handled uniformly
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    na_count = int(df.isna().sum().sum())
    if na_count > 0:
        logger.info("Imputation des %s valeurs manquantes restantes", na_count)
        if quant_vars:
            df[quant_vars] = df[quant_vars].fillna(df[quant_vars].median())
        for col in qual_vars:
            if (
                df[col].dtype.name == "category"
                and "Non renseigné" not in df[col].cat.categories
            ):
                df[col] = df[col].cat.add_categories("Non renseigné")
            df[col] = df[col].fillna("Non renseigné").astype("category")

        # Second pass in case inf values were introduced by coercion above
        df.replace([np.inf, -np.inf], np.nan, inplace=True)

        remaining_na = int(df.isna().sum().sum())
        if remaining_na > 0:
            logger.warning(
                "%s NA subsistent après imputation → suppression des lignes concernées",
                remaining_na,
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
        plt.savefig(png_path, dpi=200)
        plt.close()

        logger.info(f"Rapport segmentation '{col}' → {csv_path.name}, {png_path.name}")


def get_segment_columns(df: pd.DataFrame) -> List[str]:
    """Return configured segmentation columns present in ``df``."""
    cols = [c for c in SEGMENT_COLUMNS if c in df.columns]
    # Backward compatibility: also include columns containing 'segment'
    cols += [c for c in df.columns if "segment" in c.lower() and c not in cols]
    return cols


def scatter_all_segments(
        emb_df: pd.DataFrame,
        df_active: pd.DataFrame,
        output_dir: Path,
        prefix: str,
) -> None:
    """Generate scatter plots colored by each segment column.

    For each segmentation variable present in ``df_active`` the function
    creates two figures:
        - ``{prefix}_{segment}.png`` colored by the segment categories.
        - ``{prefix}_{segment}_clusters.png`` where colors correspond to
          ``k`` clusters, with ``k`` equal to the number of modalities of the
          segment.
    """
    seg_cols = get_segment_columns(df_active)
    if not seg_cols:
        return

    for col in seg_cols:
        categories = df_active.loc[emb_df.index, col].astype("category")
        palette = sns.color_palette("tab10", len(categories.cat.categories))
        plt.figure(figsize=(12, 6), dpi=200)
        for cat, color in zip(categories.cat.categories, palette):
            mask = categories == cat
            plt.scatter(
                emb_df.loc[mask, emb_df.columns[0]],
                emb_df.loc[mask, emb_df.columns[1]],
                s=10,
                alpha=0.7,
                color=color,
                label=str(cat),
            )
        plt.xlabel(emb_df.columns[0])
        plt.ylabel(emb_df.columns[1])
        plt.title(f"{prefix} – {col}")
        plt.legend(title=col, bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        fname = f"{prefix.lower()}_{col}.png"
        plt.savefig(output_dir / fname)
        plt.close()

        # Version clustered with k equal to number of modalities
        from sklearn.cluster import KMeans

        k = len(categories.cat.categories)
        if k >= 2:
            labels = KMeans(n_clusters=k, random_state=0).fit_predict(
                emb_df.values
            )
            palette = sns.color_palette("tab10", k)
            plt.figure(figsize=(12, 6), dpi=200)
            sc = plt.scatter(
                emb_df.iloc[:, 0],
                emb_df.iloc[:, 1],
                c=labels,
                cmap=ListedColormap(palette),
                s=10,
                alpha=0.7,
            )
            plt.xlabel(emb_df.columns[0])
            plt.ylabel(emb_df.columns[1])
            plt.title(f"{prefix} – {col} clusters")
            plt.colorbar(sc, label="cluster")
            plt.tight_layout()
            fname = f"{prefix.lower()}_{col}_clusters.png"
            plt.savefig(output_dir / fname)
            plt.close()


def scatter_cluster_variants(
        emb_df: pd.DataFrame,
        ks: Sequence[int],
        output_dir: Path,
        prefix: str,
) -> None:
    """Generate KMeans scatter plots for each ``k`` in ``ks``."""
    from sklearn.cluster import KMeans

    uniq = sorted({int(k) for k in ks if k >= 2})
    if not uniq:
        return

    for k in uniq:
        labels = KMeans(n_clusters=k, random_state=0).fit_predict(emb_df.values)
        palette = sns.color_palette("tab10", k)
        plt.figure(figsize=(12, 6), dpi=200)
        sc = plt.scatter(
            emb_df.iloc[:, 0],
            emb_df.iloc[:, 1],
            c=labels,
            cmap=ListedColormap(palette),
            s=10,
            alpha=0.7,
        )
        plt.xlabel(emb_df.columns[0])
        plt.ylabel(emb_df.columns[1])
        plt.title(f"{prefix} – k={k}")
        plt.colorbar(sc, label="cluster")
        plt.tight_layout()
        fname = f"{prefix.lower()}_k{k}.png"
        plt.savefig(output_dir / fname)
        plt.close()


def run_mfa(
        df_active: pd.DataFrame,
        quant_vars: List[str],
        qual_vars: List[str],
        output_dir: Path,
        n_components: Optional[int] = None,
        *,
        groups: Optional[Dict[str, Sequence[str]]] = None,
        optimize: bool = False,
        normalize: bool = True,
        weights: Optional[Dict[str, float]] = None,
        n_iter: int = 3,
) -> Tuple[prince.MFA, pd.DataFrame]:
    """Exécute une MFA sur le jeu de données mixte.

    Args:
        df_active: DataFrame contenant uniquement les colonnes des variables actives.
        quant_vars: Liste des noms de colonnes quantitatives.
        qual_vars: Liste des noms de colonnes qualitatives.
        output_dir: Répertoire où sauver les graphiques.
        n_components: Nombre de composantes factorielles à extraire.
        groups: Dictionnaire ``{nom_groupe: [variables]}`` définissant les
            groupes MFA. Les variables qualitatives sont automatiquement
            "one-hot" encodées.
        optimize: Si ``True`` et ``n_components`` est ``None``, choisit
            automatiquement le nombre d'axes (90 % de variance cumulée).
        normalize: Active la normalisation par groupe pour équilibrer leur
            contribution.
        weights: Pondération facultative ``{groupe: poids}`` appliquée après la
            normalisation.
        n_iter: Nombre d'itérations pour l'algorithme de l'implémentation
            prince.

    Returns:
        - L’objet prince.MFA entraîné.
        - DataFrame des coordonnées des individus dans l'espace MFA.
    """
    import prince
    import matplotlib.pyplot as plt

    # One-hot encode qualitative variables
    df_dummies = pd.get_dummies(df_active[qual_vars].astype(str))

    # Build groups dictionary
    if groups is None:
        groups = {"Quantitatives": quant_vars}
        for var in qual_vars:
            cols = [c for c in df_dummies.columns if c.startswith(f"{var}_")]
            if cols:
                groups[var] = cols
        used_cols = quant_vars + list(df_dummies.columns)
    else:
        new_groups = {}
        used_cols = []
        for gname, vars_list in groups.items():
            cols = []
            for v in vars_list:
                if v in quant_vars:
                    cols.append(v)
                elif v in qual_vars:
                    cols.extend([c for c in df_dummies.columns if c.startswith(f"{v}_")])
                elif v in df_dummies.columns:
                    cols.append(v)
                elif v in df_active.columns:
                    cols.append(v)
            if cols:
                new_groups[gname] = cols
                used_cols.extend(cols)

        # Automatically include variables not listed in any group
        all_cols = quant_vars + list(df_dummies.columns)
        remaining = [c for c in all_cols if c not in used_cols]
        if remaining:
            new_groups["Autres"] = remaining
            used_cols.extend(remaining)

        groups = new_groups
    # Combine numeric columns with the dummy-encoded qualitative variables
    df_mfa = pd.concat([df_active[quant_vars], df_dummies], axis=1)[used_cols]

    if normalize:
        from sklearn.preprocessing import StandardScaler
        for g, cols in groups.items():
            if not cols:
                continue
            scaler = StandardScaler()
            df_mfa[cols] = scaler.fit_transform(df_mfa[cols])

    if weights:
        for g, w in weights.items():
            if g in groups:
                df_mfa[groups[g]] = df_mfa[groups[g]] * float(w)

    logger = logging.getLogger(__name__)

    n_comp = n_components
    if optimize and n_components is None:
        n_init = df_mfa.shape[1]
        tmp = prince.MFA(n_components=n_init).fit(df_mfa, groups=groups)
        eigenvalues = getattr(tmp, "eigenvalues_", None)
        if eigenvalues is None:
            eigenvalues = (tmp.percentage_of_variance_ / 100) * n_init
        n_comp = select_axes(eigenvalues, threshold=0.9)
        logger.info("MFA auto: %d composantes retenues", n_comp)

    n_comp = n_comp or 5

    mfa = prince.MFA(n_components=n_comp, n_iter=n_iter)
    mfa = mfa.fit(df_mfa, groups=groups)
    mfa.df_encoded_ = df_mfa
    mfa.groups_input_ = groups
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
    """Exécute une analyse de type PCAmix via :class:`prince.FAMD`.

    Parameters
    ----------
    df_active : pd.DataFrame
        Données préfiltrées contenant uniquement les variables retenues.
    quant_vars : list of str
        Noms des colonnes quantitatives à standardiser.
    qual_vars : list of str
        Noms des colonnes qualitatives.
    output_dir : Path
        Répertoire où sauvegarder les figures de diagnostic.
    n_components : int, optional
        Nombre de dimensions à conserver. Si ``None`` (par défaut) et que
        ``optimize`` est vrai, le nombre d'axes est déterminé
        automatiquement en combinant le critère de Kaiser (valeurs propres
        supérieures à 1) et l'inertie cumulée (90 %). Sans optimisation, la
        valeur par défaut est 5.
    optimize : bool
        Active la sélection automatique du nombre d'axes quand
        ``n_components`` est omis.
    """

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
        eigenvalues = getattr(tmp, "eigenvalues_", None)
        if eigenvalues is None:
            eigenvalues = np.array(get_explained_inertia(tmp)) * n_init
        n_comp = select_axes(eigenvalues, threshold=0.9)
        logger.info(
            "PCAmix auto: %d composantes retenues", n_comp
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
        eigenvalues = getattr(tmp, "eigenvalues_", None)
        if eigenvalues is None:
            eigenvalues = np.array(get_explained_inertia(tmp)) * n_init
        n_comp = select_axes(eigenvalues, threshold=0.9)
        logger.info("PCA auto: %d composantes retenues", n_comp)

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
        eigenvalues = getattr(tmp, "eigenvalues_", None)
        if eigenvalues is None:
            eigenvalues = np.array(get_explained_inertia(tmp)) * max_dim
        n_comp = select_axes(eigenvalues, threshold=0.9)
        logger.info("MCA auto: %d dimensions retenues", n_comp)

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
) -> Tuple[TSNE, pd.DataFrame, Dict[str, float]]:
    """Applique t-SNE sur des coordonnées factorielles existantes."""

    logger = logging.getLogger(__name__)
    perpl = perplexity if perplexity is not None else 30

    def _fit_tsne(p: int) -> Tuple[TSNE, np.ndarray]:
        try:
            t = TSNE(
                n_components=n_components,
                perplexity=p,
                learning_rate=learning_rate,
                max_iter=n_iter,
                random_state=random_state,
                init="pca",
                n_jobs=-1
            )
        except TypeError:  # pragma: no cover - older scikit-learn
            t = TSNE(
                n_components=n_components,
                perplexity=p,
                learning_rate=learning_rate,
                n_iter=n_iter,
                random_state=random_state,
                init="pca",
                n_jobs=-1
            )
        emb = t.fit_transform(embeddings.values)
        return t, emb

    if optimize and perplexity is None:
        from sklearn.manifold import trustworthiness

        grid = perplexity_grid or [25, 30, 35]
        best = None
        for p in grid:
            if p >= embeddings.shape[0] / 3:
                logger.warning(
                    "Perplexity %d ignorée (trop grande pour %d points)",
                    p,
                    embeddings.shape[0],
                )
                continue
            try:
                t, emb = _fit_tsne(p)
                score = trustworthiness(embeddings.values, emb)
            except Exception as exc:  # pragma: no cover - t-SNE may fail
                logger.warning("t-SNE échec avec perplexity=%d: %s", p, exc)
                continue
            if best is None or score > best[0]:
                best = (score, p, t, emb)
        if best is None:  # fallback on default perplexity
            tsne, tsne_results = _fit_tsne(perpl)
            best_score = None
        else:
            best_score, perpl, tsne, tsne_results = best
            logger.info(
                "t-SNE optimal: perplexity=%d (trustworthiness=%.3f)",
                perpl,
                best_score,
            )
    else:
        tsne, tsne_results = _fit_tsne(perpl)

    cols = [f"TSNE{i + 1}" for i in range(n_components)]
    tsne_df = pd.DataFrame(tsne_results, columns=cols, index=embeddings.index)

    metrics = {
        "perplexity": float(perpl),
        "learning_rate": float(learning_rate),
        "n_iter": int(n_iter),
        "n_components": int(n_components),
        "kl_divergence": float(getattr(tsne, "kl_divergence_", float("nan"))),
    }

    try:
        from sklearn.manifold import trustworthiness

        metrics["trustworthiness"] = float(
            trustworthiness(embeddings.values, tsne_results)
        )
    except Exception:  # pragma: no cover - trustworthiness may fail
        metrics["trustworthiness"] = float("nan")

    return tsne, tsne_df, metrics


def export_tsne_results(
        tsne_df: pd.DataFrame,
        df_active: pd.DataFrame,
        output_dir: Path,
        metrics: Dict[str, float] | None = None,
) -> None:
    """Save scatter plot(s) and embeddings for t-SNE."""
    logger = logging.getLogger(__name__)
    output_dir.mkdir(parents=True, exist_ok=True)

    if {"TSNE1", "TSNE2"}.issubset(tsne_df.columns):
        plt.figure(figsize=(12, 6), dpi=200)
        cats = df_active.loc[tsne_df.index, "Statut commercial"].astype("category")
        palette = sns.color_palette("tab10", len(cats.cat.categories))
        for cat, color in zip(cats.cat.categories, palette):
            mask = cats == cat
            plt.scatter(
                tsne_df.loc[mask, "TSNE1"],
                tsne_df.loc[mask, "TSNE2"],
                s=15,
                alpha=0.7,
                color=color,
                label=str(cat),
            )
        plt.xlabel("TSNE1")
        plt.ylabel("TSNE2")
        plt.title("t-SNE sur axes factoriels (FAMD)")
        plt.legend(title="Statut commercial", bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        fig_path = output_dir / "tsne_scatter.png"
        plt.savefig(fig_path)
        plt.close()
        logger.info(f"Export t-SNE -> {fig_path}")

        scatter_all_segments(
            tsne_df[["TSNE1", "TSNE2"]],
            df_active,
            output_dir,
            "tsne_scatter",
        )

    if {"TSNE1", "TSNE2", "TSNE3"}.issubset(tsne_df.columns):
        fig = plt.figure(figsize=(12, 6), dpi=200)
        ax = fig.add_subplot(111, projection="3d")
        cats = df_active.loc[tsne_df.index, "Statut commercial"].astype("category")
        palette = sns.color_palette("tab10", len(cats.cat.categories))
        for cat, color in zip(cats.cat.categories, palette):
            mask = cats == cat
            ax.scatter(
                tsne_df.loc[mask, "TSNE1"],
                tsne_df.loc[mask, "TSNE2"],
                tsne_df.loc[mask, "TSNE3"],
                s=15,
                alpha=0.7,
                color=color,
                label=str(cat),
            )
        ax.set_xlabel("TSNE1")
        ax.set_ylabel("TSNE2")
        ax.set_zlabel("TSNE3")
        ax.set_title("t-SNE 3D (axes factoriels)")
        ax.legend(title="Statut commercial", bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        fig3d_path = output_dir / "tsne_scatter_3D.png"
        plt.savefig(fig3d_path)
        plt.close()
        logger.info(f"Export t-SNE -> {fig3d_path}")

    csv_path = output_dir / "tsne_embeddings.csv"
    tsne_df.to_csv(csv_path, index=True)
    logger.info(f"Export t-SNE -> {csv_path}")

    if metrics is not None:
        metrics_path = output_dir / "tsne_metrics.txt"
        with open(metrics_path, "w", encoding="utf-8") as fh:
            for k, v in metrics.items():
                fh.write(f"{k}: {v}\n")
        logger.info(f"Export t-SNE -> {metrics_path}")



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
        metric: str = "euclidean",
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
        metric: Fonction de distance à utiliser ("euclidean", "manhattan",
            "cosine", ...).
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
            nj = -1
        return umap.UMAP(
            n_neighbors=nn,
            min_dist=md,
            n_components=n_components,
            random_state=random_state,
            n_jobs=nj,
            metric=metric,
        )

    if optimize and (n_neighbors is None or min_dist is None):
        from joblib import Parallel, delayed
        from sklearn.manifold import trustworthiness

        neigh_grid = [15, 35] if n_neighbors is None else [n_neighbors]
        dist_grid = [0.05, 0.1, 0.5, 0.8] if min_dist is None else [min_dist]

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
        cats = df_active.loc[umap_df.index, "Statut commercial"].astype("category")
        palette = sns.color_palette("tab10", len(cats.cat.categories))
        for cat, color in zip(cats.cat.categories, palette):
            mask = cats == cat
            plt.scatter(
                umap_df.loc[mask, "UMAP1"],
                umap_df.loc[mask, "UMAP2"],
                s=10,
                alpha=0.7,
                color=color,
                label=str(cat),
            )
        plt.xlabel("UMAP1")
        plt.ylabel("UMAP2")
        plt.title("Projection UMAP")
        plt.legend(title="Statut commercial", bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        fig_path = output_dir / "umap_scatter.png"
        plt.savefig(fig_path)
        plt.close()
        logger.info("Projection UMAP 2D enregistrée: %s", fig_path)

        scatter_all_segments(
            umap_df[["UMAP1", "UMAP2"]],
            df_active,
            output_dir,
            "umap_scatter",
        )

    # Scatter 3D
    if {"UMAP1", "UMAP2", "UMAP3"}.issubset(umap_df.columns):
        fig = plt.figure(figsize=(12, 6), dpi=200)
        ax = fig.add_subplot(111, projection="3d")
        cats = df_active.loc[umap_df.index, "Statut commercial"].astype("category")
        palette = sns.color_palette("tab10", len(cats.cat.categories))
        for cat, color in zip(cats.cat.categories, palette):
            mask = cats == cat
            ax.scatter(
                umap_df.loc[mask, "UMAP1"],
                umap_df.loc[mask, "UMAP2"],
                umap_df.loc[mask, "UMAP3"],
                s=10,
                alpha=0.7,
                color=color,
                label=str(cat),
            )
        ax.set_xlabel("UMAP1")
        ax.set_ylabel("UMAP2")
        ax.set_zlabel("UMAP3")
        ax.set_title("Projection UMAP 3D")
        ax.legend(title="Statut commercial", bbox_to_anchor=(1.05, 1), loc="upper left")
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
        n_neighbors: Optional[int] = None,
        MN_ratio: float = 0.5,
        FP_ratio: float = 2.0,
        random_state: int | None = 42,
        optimize: bool = False,
        neighbor_grid: Sequence[int] | None = None,
) -> Tuple[Any | None, pd.DataFrame]:
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
        "Paramètres PaCMAP: n_components=%d, n_neighbors=%s, MN_ratio=%.2f, FP_ratio=%.2f",
        n_components,
        n_neighbors,
        MN_ratio,
        FP_ratio,
    )

    def _build_pacmap(nn: int):
        params = dict(
            n_components=n_components,
            n_neighbors=nn,
            MN_ratio=MN_ratio,
            FP_ratio=FP_ratio,
            random_state=random_state,
        )
        try:
            from inspect import signature

            init_param = None
            sig = signature(pacmap.PaCMAP.__init__)
            if "init" in sig.parameters:
                init_param = "init"
            elif "initialization" in sig.parameters:
                init_param = "initialization"
            if init_param:
                params[init_param] = "pca"
        except Exception:  # pragma: no cover - very old versions
            params["init"] = "pca"

        return pacmap.PaCMAP(**params)

    if optimize and n_neighbors is None:
        from sklearn.manifold import trustworthiness

        grid = neighbor_grid or [10, 15, 30]
        best = None
        for nn in grid:
            reducer = _build_pacmap(nn)
            emb = reducer.fit_transform(X)
            score = trustworthiness(X, emb)
            if best is None or score > best[0]:
                best = (score, reducer, emb, nn)
        best_score, pacmap_model, embedding, n_neighbors = best
        logger.info(
            "PaCMAP optimal: n_neighbors=%d (trustworthiness=%.3f)",
            n_neighbors,
            best_score,
        )
    else:
        nn = n_neighbors or 10
        pacmap_model = _build_pacmap(nn)
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
        cats = df_active.loc[pacmap_df.index, "Statut commercial"].astype("category")
        palette = sns.color_palette("tab10", len(cats.cat.categories))
        for cat, color in zip(cats.cat.categories, palette):
            mask = cats == cat
            plt.scatter(
                pacmap_df.loc[mask, "PACMAP1"],
                pacmap_df.loc[mask, "PACMAP2"],
                s=10,
                alpha=0.7,
                color=color,
                label=str(cat),
            )
        plt.xlabel("PACMAP1")
        plt.ylabel("PACMAP2")
        plt.title("PaCMAP \u2013 individus (dim 1 vs dim 2)")
        plt.legend(title="Statut commercial", bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        plt.savefig(output_dir / "pacmap_scatter.png")
        plt.close()
        logger.info("Projection PaCMAP 2D enregistr\u00e9e")

        scatter_all_segments(
            pacmap_df[["PACMAP1", "PACMAP2"]],
            df_active,
            output_dir,
            "pacmap_scatter",
        )

    if {"PACMAP1", "PACMAP2", "PACMAP3"}.issubset(pacmap_df.columns):
        fig = plt.figure(figsize=(12, 6), dpi=200)
        ax = fig.add_subplot(111, projection="3d")
        cats = df_active.loc[pacmap_df.index, "Statut commercial"].astype("category")
        palette = sns.color_palette("tab10", len(cats.cat.categories))
        for cat, color in zip(cats.cat.categories, palette):
            mask = cats == cat
            ax.scatter(
                pacmap_df.loc[mask, "PACMAP1"],
                pacmap_df.loc[mask, "PACMAP2"],
                pacmap_df.loc[mask, "PACMAP3"],
                s=10,
                alpha=0.7,
                color=color,
                label=str(cat),
            )
        ax.set_xlabel("PACMAP1")
        ax.set_ylabel("PACMAP2")
        ax.set_zlabel("PACMAP3")
        ax.set_title("PaCMAP \u2013 individus (3D)")
        ax.legend(title="Statut commercial", bbox_to_anchor=(1.05, 1), loc="upper left")
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
        knn: int | str | None = None,
        t: int | str | None = "auto",
        random_state: int | None = 42,
        optimize: bool = False,
) -> Tuple[Any | None, pd.DataFrame]:
    """Exécute PHATE sur les données CRM pour détecter des trajectoires potentielles.

    PHATE est particulièrement adapté pour révéler des évolutions progressives,
    par exemple le passage de prospect à client fidèle. Il s'appuie sur un
    graphe de voisins et une diffusion pour préserver la structure globale. Les
    valeurs par défaut (``knn=5``, ``t='auto'``) conviennent généralement aux
    volumes CRM ; ``n_jobs=-1`` exploite tous les cœurs et ``random_state=42``
    assure la reproductibilité. Lorsque ``optimize`` est activé et qu'aucun
    ``knn`` n'est fourni, une recherche sur grille utilise le nombre de
    modalités des variables de segmentation comme valeurs candidates.
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

    if t is None:
        t = "auto"

    def _fit_phate(nn: int | None) -> tuple[Any, np.ndarray]:
        op = phate.PHATE(
            n_components=n_components,
            knn=nn if nn is not None else 5,
            t=t,
            n_jobs=-1,
            random_state=random_state,
        )
        emb = op.fit_transform(X_mix)
        return op, emb

    best_knn = knn
    if optimize and knn is None:
        from sklearn.manifold import trustworthiness

        candidate_vars = [
            "Catégorie",
            "Entité opérationnelle",
            "Pilier",
            "Sous-catégorie",
            "Statut commercial",
            "Statut production",
            "Type opportunité",
        ]

        counts = [
            df_active[v].nunique()
            for v in candidate_vars
            if v in df_active.columns
        ]

        grid = sorted(set(counts)) or [5, 10, 20]

        scores: list[tuple[float, int, Any, np.ndarray]] = []
        for nn in grid:
            op, emb = _fit_phate(nn)
            score = trustworthiness(X_mix, emb)
            scores.append((score, nn, op, emb))
        best = max(scores, key=lambda x: x[0])
        best_score, best_knn, phate_operator, embedding = best
        logger.info(
            "PHATE optimal: knn=%d (trustworthiness=%.3f)",
            best_knn,
            best_score,
        )
    else:
        phate_operator, embedding = _fit_phate(knn)
        best_knn = knn if knn is not None else 5

    logger.info(
        "Paramètres PHATE: n_components=%d, knn=%s, t=%s",
        n_components,
        best_knn,
        t,
    )

    cols = [f"PHATE{i + 1}" for i in range(n_components)]
    phate_df = pd.DataFrame(embedding, index=df_active.index, columns=cols)

    return phate_operator, phate_df


def export_phate_results(
        phate_df: pd.DataFrame,
        df_active: pd.DataFrame,
        output_dir: Path,
) -> None:
    """Génère les visualisations et CSV pour PHATE.

    Cette version détermine d'abord un nombre de clusters ``k`` en appliquant
    plusieurs algorithmes de regroupement sur les coordonnées PHATE. Le nombre
    de groupes ainsi obtenu est ensuite comparé au nombre de modalités des
    principales variables de segmentation du CRM afin de choisir
    automatiquement la variable la plus pertinente pour colorer le scatter.
    """

    logger = logging.getLogger(__name__)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1) Choix automatique de la variable de coloration en fonction du nombre
    #    de clusters détectés dans l'espace PHATE.
    labels, _, _ = best_clustering_labels(phate_df.values)
    unique_labels = set(labels)
    if -1 in unique_labels:
        unique_labels.remove(-1)
    k = len(unique_labels) if unique_labels else 1

    candidate_vars = [
        "Catégorie",
        "Entité opérationnelle",
        "Pilier",
        "Sous-catégorie",
        "Statut commercial",
        "Statut production",
        "Type opportunité",
    ]

    var_counts = {
        var: df_active[var].nunique()
        for var in candidate_vars
        if var in df_active.columns
    }

    if var_counts:
        color_var = min(var_counts.items(), key=lambda t: abs(t[1] - k))[0]
    else:
        color_var = "Statut commercial"

    logger.info(
        "PHATE clustering -> k=%d, variable coloration choisie: %s",
        k,
        color_var,
    )

    codes = (
        df_active.loc[phate_df.index, color_var]
        .astype("category")
        .cat.codes
    )

    if {"PHATE1", "PHATE2"}.issubset(phate_df.columns):
        plt.figure(figsize=(12, 6), dpi=200)
        cats = df_active.loc[phate_df.index, color_var].astype("category")
        palette = sns.color_palette("tab10", len(cats.cat.categories))
        for cat, color in zip(cats.cat.categories, palette):
            mask = cats == cat
            plt.scatter(
                phate_df.loc[mask, "PHATE1"],
                phate_df.loc[mask, "PHATE2"],
                s=10,
                alpha=0.7,
                color=color,
                label=str(cat),
            )
        plt.xlabel("PHATE1")
        plt.ylabel("PHATE2")
        plt.title("Projection PHATE des individus")
        plt.legend(title=color_var, bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        plt.savefig(output_dir / "phate_scatter.png")
        plt.close()
        logger.info("Projection PHATE 2D enregistrée")

        scatter_all_segments(
            phate_df[["PHATE1", "PHATE2"]],
            df_active,
            output_dir,
            "phate_scatter",
        )

        seg_cols = get_segment_columns(df_active)
        if seg_cols:
            k_values = [df_active[c].nunique() for c in seg_cols]
            scatter_cluster_variants(
                phate_df[["PHATE1", "PHATE2"]],
                k_values,
                output_dir,
                "phate_clusters",
            )

    if {"PHATE1", "PHATE2", "PHATE3"}.issubset(phate_df.columns):
        fig = plt.figure(figsize=(12, 6), dpi=200)
        ax = fig.add_subplot(111, projection="3d")
        cats = df_active.loc[phate_df.index, color_var].astype("category")
        palette = sns.color_palette("tab10", len(cats.cat.categories))
        for cat, color in zip(cats.cat.categories, palette):
            mask = cats == cat
            ax.scatter(
                phate_df.loc[mask, "PHATE1"],
                phate_df.loc[mask, "PHATE2"],
                phate_df.loc[mask, "PHATE3"],
                s=10,
                alpha=0.7,
                color=color,
                label=str(cat),
            )
        ax.set_xlabel("PHATE1")
        ax.set_ylabel("PHATE2")
        ax.set_zlabel("PHATE3")
        ax.set_title("Projection PHATE des individus (3D)")
        ax.legend(title=color_var, bbox_to_anchor=(1.05, 1), loc="upper left")
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


def select_n_components(
        eigenvalues: Sequence[float],
        method: str = "variance",
        threshold: float = 0.9,
) -> int:
    """Select the number of components according to a rule.

    Parameters
    ----------
    eigenvalues : Sequence[float]
        Eigenvalues of the decomposition.
    method : {"variance", "kaiser", "elbow"}
        Selection rule. ``"variance"`` uses the cumulative explained inertia
        threshold, ``"kaiser"`` keeps axes with eigenvalue >= 1 and
        ``"elbow"`` uses the point of maximum curvature of the scree plot.
    threshold : float, optional
        Cumulative inertia threshold when ``method='variance'``.
    """
    ev = np.asarray(eigenvalues)
    if method == "kaiser":
        return max(1, int((ev >= 1).sum()))
    if method == "elbow":
        if len(ev) <= 2:
            return len(ev)
        diff2 = np.diff(np.diff(ev))
        return int(np.argmin(diff2) + 2)
    # default: variance threshold
    ratio = ev / ev.sum()
    cum = np.cumsum(ratio)
    for i, v in enumerate(cum, start=1):
        if v >= threshold:
            return i
    return len(ev)


def select_axes(
        explained_variance: Sequence[float],
        *,
        threshold: float = 0.8,
) -> int:
    """Return the recommended number of axes to keep.

    Parameters
    ----------
    explained_variance : Sequence[float]
        Eigenvalues or fractions of explained inertia for each axis.
    threshold : float, optional
        Cumulative inertia threshold for the variance rule (default 0.8).

    Notes
    -----
    The function computes three classical criteria: Kaiser (eigenvalues > 1),
    cumulative inertia and elbow (change of slope). The returned value is the
    maximum of the Kaiser and inertia thresholds to avoid discarding
    potentially important axes. All intermediate values are logged.
    """
    logger = logging.getLogger(__name__)
    ev = np.asarray(explained_variance, dtype=float)

    if ev.sum() <= 1.0:
        eigenvalues = ev * len(ev)
    else:
        eigenvalues = ev

    n_kaiser = max(1, int((eigenvalues >= 1).sum()))

    ratio = eigenvalues / eigenvalues.sum()
    cum = np.cumsum(ratio)
    n_inertia = next((i + 1 for i, v in enumerate(cum) if v >= threshold), len(ev))

    if len(eigenvalues) <= 2:
        n_elbow = len(eigenvalues)
    else:
        diff2 = np.diff(np.diff(eigenvalues))
        n_elbow = int(np.argmin(diff2) + 2)

    logger.info(
        "Axes recommandés : Kaiser=%d ; %.0f%% inertie=%d ; Coude=%d",
        n_kaiser,
        threshold * 100,
        n_inertia,
        n_elbow,
    )

    return max(n_kaiser, n_inertia)


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

    rule = cfg.get('n_components_rule')
    thresh = float(cfg.get('variance_threshold', 0.9))

    if (optimize or rule) and n_components is None:
        n_init = df_for_famd.shape[1]
        tmp = prince.FAMD(
            n_components=n_init,
            n_iter=3,
            copy=True,
            check_input=True,
            engine='sklearn'
        ).fit(df_for_famd)
        eigenvalues = getattr(tmp, 'eigenvalues_', None)
        if eigenvalues is None:
            eigenvalues = np.array(get_explained_inertia(tmp)) * n_init
        if rule in {"variance", "kaiser", "elbow"}:
            n_comp = select_n_components(eigenvalues, method=rule, threshold=thresh)
            logger.info(
                "FAMD auto: %d composantes retenues via %s",
                n_comp,
                rule,
            )
        else:
            n_comp = select_axes(eigenvalues, threshold=thresh)
            logger.info(
                "FAMD auto: %d composantes retenues (mix de critères)", n_comp
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
    *,
    scree_methods: Optional[Sequence[str]] = None,
    scatter_methods: Optional[Sequence[str]] = None,
) -> None:
    """Visualisations comparatives pour plusieurs méthodes factorielles."""

    if scree_methods is None:
        scree_methods = list(results_dict.keys())
    if scatter_methods is None:
        scatter_methods = list(results_dict.keys())

    # ─── Scree-plots ────────────────────────────────────────────
    methods_inertia = {}
    for m in scree_methods:
        inertia = results_dict.get(m, {}).get("inertia")
        if inertia is not None and len(inertia) > 0:
            methods_inertia[m] = inertia
    if methods_inertia:
        n = len(methods_inertia)
        fig, axes = plt.subplots(1, n, figsize=(12, 6), dpi=200)
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

    # ─── Scatter F1–F2 comparés ─────────────────────────────────
    emb_methods = [m for m in scatter_methods if m in results_dict]
    if emb_methods:
        n = len(emb_methods)
        ncols = 3
        nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(12, 6), dpi=200)
        axes = np.atleast_1d(axes).reshape(-1)
        for ax, m in zip(axes, emb_methods):
            emb = results_dict[m]["embeddings"]
            if emb.shape[1] < 2:
                ax.axis("off")
                continue
            x, y = emb.iloc[:, 0], emb.iloc[:, 1]
            codes = (
                df_active.loc[emb.index, "Statut commercial"].astype("category").cat.codes
            )
            ax.scatter(x, y, c=codes, s=15, alpha=0.6)
            ax.set_title(m)
            ax.set_xlabel(emb.columns[0])
            ax.set_ylabel(emb.columns[1])
        for j in range(len(emb_methods), len(axes)):
            axes[j].axis("off")
        plt.tight_layout()
        plt.savefig(output_dir / "multi_scatter.png")
        plt.close()

    # ─── Heatmap d'évaluation ──────────────────────────────────
    # Cette visualisation est désormais générée uniquement par
    # ``evaluate_methods`` et exportée sous le nom
    # ``methods_heatmap.png``. L'ancienne heatmap "multi_heatmap.png"
    # était redondante et a été retirée.


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
    plt.plot(ax_idx, np.cumsum(inertia) * 100, "-o", color="red", label="Cumul")
    plt.xlabel("Composante")
    plt.ylabel("% Inertie expliquée")
    plt.title("Éboulis FAMD")
    plt.xticks(ax_idx, [f"F{i}" for i in ax_idx])
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "famd_scree_plot.png")
    plt.close()
    logger.info("Scree plot enregistré")

    # ─── Projection individus 2D ──────────────────────────────────────
    if {"F1", "F2"}.issubset(row_coords.columns):
        plt.figure(figsize=(12, 6), dpi=200)
        if df_active is not None and "Statut commercial" in df_active.columns:
            cats = df_active.loc[row_coords.index, "Statut commercial"].astype("category")
            categories = cats.cat.categories
            palette = sns.color_palette("tab10", len(categories))
            for cat, color in zip(categories, palette):
                mask = cats == cat
                plt.scatter(
                    row_coords.loc[mask, "F1"],
                    row_coords.loc[mask, "F2"],
                    s=10,
                    alpha=0.7,
                    color=color,
                    label=str(cat),
                )
            plt.legend(title="Statut commercial", bbox_to_anchor=(1.05, 1), loc="upper left")
        else:
            plt.scatter(row_coords["F1"], row_coords["F2"], s=10, alpha=0.7)
        plt.xlabel(f"F1 ({inertia[0]*100:.1f}% inertie)")
        plt.ylabel(f"F2 ({inertia[1]*100:.1f}% inertie)")
        plt.title("FAMD – individus (F1 vs F2)")
        plt.tight_layout()
        plt.savefig(output_dir / "famd_indiv_plot.png")
        plt.close()
        logger.info("Projection F1-F2 enregistrée")

        scatter_all_segments(
            row_coords[["F1", "F2"]],
            df_active,
            output_dir,
            "famd_indiv",
        )

    # ─── Projection individus 3D ──────────────────────────────────────
    if {"F1", "F2", "F3"}.issubset(row_coords.columns):
        fig = plt.figure(figsize=(12, 6), dpi=200)
        ax = fig.add_subplot(111, projection="3d")
        if df_active is not None and "Statut commercial" in df_active.columns:
            cats = df_active.loc[row_coords.index, "Statut commercial"].astype("category")
            categories = cats.cat.categories
            palette = sns.color_palette("tab10", len(categories))
            for cat, color in zip(categories, palette):
                mask = cats == cat
                ax.scatter(
                    row_coords.loc[mask, "F1"],
                    row_coords.loc[mask, "F2"],
                    row_coords.loc[mask, "F3"],
                    s=10,
                    alpha=0.7,
                    color=color,
                    label=str(cat),
                )
            ax.legend(title="Statut commercial", bbox_to_anchor=(1.05, 1), loc="upper left")
        else:
            ax.scatter(row_coords["F1"], row_coords["F2"], row_coords["F3"], s=10, alpha=0.7)
        ax.set_xlabel("F1")
        ax.set_ylabel("F2")
        ax.set_zlabel("F3")
        ax.set_title("FAMD – individus (3D)")
        plt.tight_layout()
        plt.savefig(output_dir / "famd_indiv_plot_3D.png")
        plt.close()
        logger.info("Projection 3D enregistrée")

    # ─── Cercle des corrélations ─────────────────────────────────────
    if {"F1", "F2"}.issubset(col_coords.columns):
        qcoords = col_coords.loc[quant_vars].dropna(how="any")
        plt.figure(figsize=(12, 6), dpi=200)
        ax = plt.gca()
        plot_correlation_circle(ax, qcoords, "FAMD – cercle des corrélations (F1–F2)")
        plt.tight_layout()
        plt.savefig(output_dir / "famd_correlation_circle.png")
        plt.close()
        logger.info("Cercle des corrélations enregistré")

    # ─── Modalités qualitatives ──────────────────────────────────────
    if {"F1", "F2"}.issubset(col_coords.columns):
        modalities = col_coords.drop(index=quant_vars, errors="ignore")
        plt.figure(figsize=(12, 6), dpi=200)
        var_names = [m.split("=")[0] for m in modalities.index]
        palette = sns.color_palette("tab10", len(set(var_names)))
        color_map = {v: palette[i] for i, v in enumerate(sorted(set(var_names)))}
        for mod, var in zip(modalities.index, var_names):
            plt.scatter(
                modalities.loc[mod, "F1"],
                modalities.loc[mod, "F2"],
                marker="D",
                s=20,
                alpha=0.7,
                color=color_map[var],
            )
            plt.text(
                modalities.loc[mod, "F1"] * 1.02,
                modalities.loc[mod, "F2"] * 1.02,
                mod,
                fontsize=8,
            )
        handles = [
            plt.Line2D([0], [0], marker='D', linestyle='', color=color_map[v], label=v)
            for v in sorted(set(var_names))
        ]
        plt.legend(handles=handles, title="Variable", bbox_to_anchor=(1.05, 1), loc="upper left")
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
            for tick in axes_plot[i].get_xticklabels():
                tick.set_horizontalalignment("right")
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

    row_cos2 = (row_coords ** 2).div((row_coords ** 2).sum(axis=1), axis=0)
    row_cos2.to_csv(output_dir / "famd_row_cos2.csv")
    col_cos2 = (col_coords ** 2).div((col_coords ** 2).sum(axis=1), axis=0)
    col_cos2.to_csv(output_dir / "famd_col_cos2.csv")
    logger.info("CSV cos2 enregistré")

    logger.info("Export des résultats FAMD terminé")


def export_mfa_results(
        mfa_model,
        row_coords: pd.DataFrame,
        output_dir: Path,
        quant_vars: List[str],
        qual_vars: List[str],
        df_active: Optional[pd.DataFrame] = None,
        *,
        segment_col: str = "Statut commercial",
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

    groups_dict = getattr(mfa_model, "groups_input_", getattr(mfa_model, "groups_", {}))
    group_map = {col: g for g, cols in groups_dict.items() for col in cols}
    palette_groups = sns.color_palette("tab20", len(groups_dict))
    group_colors = {g: palette_groups[i] for i, g in enumerate(groups_dict)}

    # Coloration selon la segmentation CRM si disponible
    codes = None
    if df_active is not None and segment_col in df_active.columns:
        codes = df_active.loc[row_coords.index, segment_col].astype("category").cat.codes
        palette = sns.color_palette("tab10", len(pd.unique(codes)))
    else:
        palette = sns.color_palette("tab10", 1)

    # ─── Projection individus 2D ──────────────────────────────────────
    if {"F1", "F2"}.issubset(row_coords.columns):
        plt.figure(figsize=(12, 6), dpi=200)
        if codes is not None:
            cats = df_active.loc[row_coords.index, segment_col].astype("category")
            palette = sns.color_palette("tab10", len(cats.cat.categories))
            for cat, color in zip(cats.cat.categories, palette):
                mask = cats == cat
                plt.scatter(
                    row_coords.loc[mask, "F1"],
                    row_coords.loc[mask, "F2"],
                    s=10,
                    alpha=0.7,
                    color=color,
                    label=str(cat),
                )
            plt.legend(title=segment_col, bbox_to_anchor=(1.05, 1), loc="upper left")
        else:
            plt.scatter(row_coords["F1"], row_coords["F2"], color=palette[0], s=10, alpha=0.7)
        plt.xlabel("F1")
        plt.ylabel("F2")
        plt.title("MFA – individus (F1–F2)")
        plt.tight_layout()
        plt.savefig(output_dir / "mfa_indiv_plot.png")
        plt.close()
        logger.info("Projection MFA F1-F2 enregistrée")

        scatter_all_segments(
            row_coords[["F1", "F2"]],
            df_active,
            output_dir,
            "mfa_indiv",
        )

    # ─── Projection individus 3D ──────────────────────────────────────
    if {"F1", "F2", "F3"}.issubset(row_coords.columns):
        fig = plt.figure(figsize=(12, 6), dpi=200)
        ax = fig.add_subplot(111, projection="3d")
        if codes is not None:
            cats = df_active.loc[row_coords.index, segment_col].astype("category")
            palette = sns.color_palette("tab10", len(cats.cat.categories))
            for cat, color in zip(cats.cat.categories, palette):
                mask = cats == cat
                ax.scatter(
                    row_coords.loc[mask, "F1"],
                    row_coords.loc[mask, "F2"],
                    row_coords.loc[mask, "F3"],
                    s=10,
                    alpha=0.7,
                    color=color,
                    label=str(cat),
                )
            ax.legend(title=segment_col, bbox_to_anchor=(1.05, 1), loc="upper left")
        else:
            ax.scatter(row_coords["F1"], row_coords["F2"], row_coords["F3"], color=palette[0], s=10, alpha=0.7)
        ax.set_xlabel("F1")
        ax.set_ylabel("F2")
        ax.set_zlabel("F3")
        ax.set_title("MFA – individus (3D)")
        plt.tight_layout()
        plt.savefig(output_dir / "mfa_indiv_plot_3D.png")
        plt.close()
        logger.info("Projection MFA 3D enregistrée")

    # ─── Cercle des corrélations ─────────────────────────────────────
    if {"F1", "F2"}.issubset(col_coords.columns):
        qcoords = col_coords.loc[[v for v in quant_vars if v in col_coords.index]]
        if not qcoords.empty:
            plt.figure(figsize=(12, 6), dpi=200)
            ax = plt.gca()
            colors = {v: group_colors.get(group_map.get(v, "")) for v in qcoords.index}
            plot_correlation_circle(ax, qcoords, "MFA – cercle des corrélations", colors)
            plt.tight_layout()
            plt.savefig(output_dir / "mfa_correlation_circle.png")
            plt.close()
            logger.info("Cercle des corrélations MFA enregistré")

    # ─── Modalités qualitatives ──────────────────────────────────────
    if {"F1", "F2"}.issubset(col_coords.columns):
        modalities = col_coords.drop(index=[v for v in quant_vars if v in col_coords.index], errors="ignore")
        if not modalities.empty:
            plt.figure(figsize=(12, 6), dpi=200)
            var_groups = [group_map.get(m, "") for m in modalities.index]
            color_map = {g: group_colors.get(g, "grey") for g in groups_dict}
            markers = ['o', 's', '^', 'D', 'v', 'P', 'X', '*', '+', 'x']
            marker_map = {g: markers[i % len(markers)] for i, g in enumerate(groups_dict)}
            for mod, grp in zip(modalities.index, var_groups):
                plt.scatter(
                    modalities.loc[mod, "F1"],
                    modalities.loc[mod, "F2"],
                    color=color_map.get(grp, "grey"),
                    marker=marker_map.get(grp, 'o'),
                    s=20,
                    alpha=0.7,
                )

            radius = np.sqrt(modalities["F1"] ** 2 + modalities["F2"] ** 2)
            thresh = radius.quantile(0.7)
            for mod, r in radius.items():
                if r >= thresh:
                    label = mod
                    if "_" in mod:
                        var, val = mod.split("_", 1)
                        label = f"{var}={val}"
                    plt.text(modalities.loc[mod, "F1"], modalities.loc[mod, "F2"], label, fontsize=8)

            handles = [
                plt.Line2D(
                    [0],
                    [0],
                    marker=marker_map[g],
                    linestyle='',
                    color=color_map[g],
                    label=g,
                )
                for g in groups_dict
            ]
            plt.legend(handles=handles, title="Variable", bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.xlabel("F1")
            plt.ylabel("F2")
            plt.title("MFA – modalités (F1–F2)")
            plt.tight_layout()
            plt.savefig(output_dir / "mfa_modalities_plot.png")
            plt.close()
            logger.info("Modalités MFA enregistrées")

            # Zoom on near-origin modalities with annotations
            near = modalities[radius < thresh]
            if not near.empty:
                plt.figure(figsize=(12, 6), dpi=200)
                for mod in near.index:
                    grp = group_map.get(mod, "")
                    plt.scatter(
                        near.loc[mod, "F1"],
                        near.loc[mod, "F2"],
                        color=color_map.get(grp, "grey"),
                        marker=marker_map.get(grp, 'o'),
                        s=20,
                        alpha=0.7,
                    )
                    label = mod
                    if "_" in mod:
                        var, val = mod.split("_", 1)
                        label = f"{var}={val}"
                    plt.text(near.loc[mod, "F1"], near.loc[mod, "F2"], label, fontsize=8)

                plt.legend(handles=handles, title="Variable", bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.xlabel("F1")
                plt.ylabel("F2")
                plt.title("MFA – modalités proches (zoom)")
                plt.tight_layout()
                plt.savefig(output_dir / "mfa_modalities_zoom.png")
                plt.close()

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
                for tick in axes_plot[i].get_xticklabels():
                    tick.set_horizontalalignment("right")
            else:
                axes_plot[i].axis("off")
        plt.tight_layout()
        plt.savefig(output_dir / "mfa_contributions.png")
        plt.close()
        logger.info("Contributions MFA enregistrées")

        group_contrib = {
            g: contrib.loc[[c for c in cols if c in contrib.index]].sum()
            for g, cols in groups_dict.items()
        }
        if group_contrib:
            df_gc = pd.DataFrame(group_contrib).T
            df_gc = df_gc[contrib.columns]
            df_gc.plot(kind="bar", stacked=True, figsize=(8, 6), colormap="tab20")
            plt.ylabel("% contribution")
            plt.title("Contribution par groupe")
            plt.tight_layout()
            plt.savefig(output_dir / "mfa_group_contributions.png")
            plt.close()
            df_gc.to_csv(output_dir / "mfa_group_contributions.csv", index=True)
            logger.info("CSV contributions groupes MFA enregistré")

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

    if hasattr(mfa_model, "df_encoded_"):
        part = mfa_model.partial_row_coordinates(mfa_model.df_encoded_)
        part.columns = [
            (grp, f"F{i+1}") for grp, i in part.columns
        ]
        part.to_csv(output_dir / "mfa_individus_partial_coords.csv", index=True)
        logger.info("CSV coordonnées partielles individus MFA enregistré")

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
    row_coords = row_coords.copy();
    row_coords.columns = axes
    col_coords = col_coords.copy();
    col_coords.columns = axes[:col_coords.shape[1]]

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
            cats = df_active.loc[row_coords.index, "Statut commercial"].astype("category")
            palette = sns.color_palette("tab10", len(cats.cat.categories))
            for cat, color in zip(cats.cat.categories, palette):
                mask = cats == cat
                plt.scatter(
                    row_coords.loc[mask, "F1"],
                    row_coords.loc[mask, "F2"],
                    s=10,
                    alpha=0.7,
                    color=color,
                    label=str(cat),
                )
            plt.legend(title="Statut commercial", bbox_to_anchor=(1.05, 1), loc="upper left")
        else:
            plt.scatter(row_coords["F1"], row_coords["F2"], s=10, alpha=0.7)
        plt.xlabel("F1"); plt.ylabel("F2")
        plt.title("PCA – individus (F1–F2)")
        plt.tight_layout()
        plt.savefig(output_dir / "pca_indiv_plot.png")
        plt.close()

        scatter_all_segments(
            row_coords[["F1", "F2"]],
            df_active,
            output_dir,
            "pca_indiv",
        )

    # Individuals 3D
    if {"F1", "F2", "F3"}.issubset(row_coords.columns):
        fig = plt.figure(figsize=(12, 6), dpi=200)
        ax = fig.add_subplot(111, projection="3d")
        if df_active is not None and "Statut commercial" in df_active.columns:
            cats = df_active.loc[row_coords.index, "Statut commercial"].astype("category")
            palette = sns.color_palette("tab10", len(cats.cat.categories))
            for cat, color in zip(cats.cat.categories, palette):
                mask = cats == cat
                ax.scatter(
                    row_coords.loc[mask, "F1"],
                    row_coords.loc[mask, "F2"],
                    row_coords.loc[mask, "F3"],
                    s=10,
                    alpha=0.7,
                    color=color,
                    label=str(cat),
                )
            ax.legend(title="Statut commercial", bbox_to_anchor=(1.05, 1), loc="upper left")
        else:
            ax.scatter(row_coords["F1"], row_coords["F2"], row_coords["F3"], s=10, alpha=0.7)
        ax.set_xlabel("F1"); ax.set_ylabel("F2"); ax.set_zlabel("F3")
        ax.set_title("PCA – individus (3D)")
        plt.tight_layout()
        plt.savefig(output_dir / "pca_indiv_plot_3D.png")
        plt.close()

    # Correlation circle
    if {"F1", "F2"}.issubset(col_coords.columns):
        plt.figure(figsize=(12, 6), dpi=200)
        ax = plt.gca()
        plot_correlation_circle(ax, col_coords, "PCA – cercle des corrélations (F1–F2)")
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
    row_coords = row_coords.copy();
    row_coords.columns = axes
    col_coords = col_coords.copy();
    col_coords.columns = axes[:col_coords.shape[1]]

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
            cats = df_active.loc[row_coords.index, "Statut commercial"].astype("category")
            palette = sns.color_palette("tab10", len(cats.cat.categories))
            for cat, color in zip(cats.cat.categories, palette):
                mask = cats == cat
                plt.scatter(
                    row_coords.loc[mask, "F1"],
                    row_coords.loc[mask, "F2"],
                    s=10,
                    alpha=0.7,
                    color=color,
                    label=str(cat),
                )
            plt.legend(title="Statut commercial", bbox_to_anchor=(1.05, 1), loc="upper left")
        else:
            plt.scatter(row_coords["F1"], row_coords["F2"], s=10, alpha=0.7)
        plt.xlabel("F1"); plt.ylabel("F2")
        plt.title("MCA – individus (F1–F2)")
        plt.tight_layout()
        plt.savefig(output_dir / "mca_indiv_plot.png")
        plt.close()

    # Individuals 3D
    if {"F1", "F2", "F3"}.issubset(row_coords.columns):
        fig = plt.figure(figsize=(12, 6), dpi=200)
        ax = fig.add_subplot(111, projection="3d")
        if df_active is not None and "Statut commercial" in df_active.columns:
            cats = df_active.loc[row_coords.index, "Statut commercial"].astype("category")
            palette = sns.color_palette("tab10", len(cats.cat.categories))
            for cat, color in zip(cats.cat.categories, palette):
                mask = cats == cat
                ax.scatter(
                    row_coords.loc[mask, "F1"],
                    row_coords.loc[mask, "F2"],
                    row_coords.loc[mask, "F3"],
                    s=10,
                    alpha=0.7,
                    color=color,
                    label=str(cat),
                )
            ax.legend(title="Statut commercial", bbox_to_anchor=(1.05, 1), loc="upper left")
        else:
            ax.scatter(row_coords["F1"], row_coords["F2"], row_coords["F3"], s=10, alpha=0.7)
        ax.set_xlabel("F1"); ax.set_ylabel("F2"); ax.set_zlabel("F3")
        ax.set_title("MCA – individus (3D)")
        plt.tight_layout()
        plt.savefig(output_dir / "mca_indiv_plot_3D.png")
        plt.close()

    # Modalities
    if {"F1", "F2"}.issubset(col_coords.columns):
        plt.figure(figsize=(12, 6), dpi=200)
        var_names = [m.split("_", 1)[0] if "_" in m else m for m in col_coords.index]
        palette = sns.color_palette("tab10", len(set(var_names)))
        markers = ['o', 's', '^', 'D', 'v', 'P', 'X', '*', '+', 'x']
        color_map = {v: palette[i] for i, v in enumerate(sorted(set(var_names)))}
        marker_map = {v: markers[i % len(markers)] for i, v in enumerate(sorted(set(var_names)))}
        for mod, var in zip(col_coords.index, var_names):
            plt.scatter(
                col_coords.loc[mod, "F1"],
                col_coords.loc[mod, "F2"],
                color=color_map[var],
                marker=marker_map[var],
                s=20,
                alpha=0.7,
            )

        radius = np.sqrt(col_coords["F1"] ** 2 + col_coords["F2"] ** 2)
        thresh = radius.quantile(0.7)
        for mod, r in radius.items():
            if r >= thresh:
                label = mod
                if "_" in mod:
                    var, val = mod.split("_", 1)
                    label = f"{var}={val}"
                plt.text(col_coords.loc[mod, "F1"], col_coords.loc[mod, "F2"], label, fontsize=8)

        handles = [
            plt.Line2D(
                [0],
                [0],
                marker=marker_map[v],
                linestyle='',
                color=color_map[v],
                label=v,
            )
            for v in sorted(set(var_names))
        ]
        plt.legend(handles=handles, title="Variable", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xlabel("F1"); plt.ylabel("F2")
        plt.title("MCA – modalités (F1–F2)")
        plt.tight_layout()
        plt.savefig(output_dir / "mca_modalities_plot.png")
        plt.close()

        # Zoom on modalities near the origin
        near = col_coords[radius < thresh]
        if not near.empty:
            plt.figure(figsize=(12, 6), dpi=200)
            for mod, var in zip(near.index, [m.split("_",1)[0] if "_" in m else m for m in near.index]):
                plt.scatter(
                    near.loc[mod, "F1"],
                    near.loc[mod, "F2"],
                    color=color_map[var],
                    marker=marker_map[var],
                    s=20,
                    alpha=0.7,
                )
                label = mod
                if "_" in mod:
                    var, val = mod.split("_", 1)
                    label = f"{var}={val}"
                plt.text(near.loc[mod, "F1"], near.loc[mod, "F2"], label, fontsize=8)

            plt.legend(handles=handles, title="Variable", bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.xlabel("F1")
            plt.ylabel("F2")
            plt.title("MCA – modalités proches (zoom)")
            plt.tight_layout()
            plt.savefig(output_dir / "mca_modalities_zoom.png")
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
            for tick in axes_plot[i].get_xticklabels():
                tick.set_horizontalalignment("right")
        else:
            axes_plot[i].axis("off")
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.25)
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
            cats = df_active.loc[row_coords.index, "Statut commercial"].astype("category")
            palette = sns.color_palette("tab10", len(cats.cat.categories))
            for cat, color in zip(cats.cat.categories, palette):
                mask = cats == cat
                plt.scatter(
                    row_coords.loc[mask, "F1"],
                    row_coords.loc[mask, "F2"],
                    s=10,
                    alpha=0.7,
                    color=color,
                    label=str(cat),
                )
            plt.legend(title="Statut commercial", bbox_to_anchor=(1.05, 1), loc="upper left")
        else:
            plt.scatter(row_coords["F1"], row_coords["F2"], s=10, alpha=0.7)
        plt.xlabel("F1")
        plt.ylabel("F2")
        plt.title("PCAmix – individus (F1–F2)")
        plt.tight_layout()
        plt.savefig(output_dir / "pcamix_indiv_plot.png")
        plt.close()
        logger.info("Projection individus PCAmix F1-F2 enregistrée")

        scatter_all_segments(
            row_coords[["F1", "F2"]],
            df_active,
            output_dir,
            "pcamix_indiv",
        )

    # ─── Projection individus 3D ──────────────────────────────────────
    if {"F1", "F2", "F3"}.issubset(row_coords.columns):
        fig = plt.figure(figsize=(12, 6), dpi=200)
        ax = fig.add_subplot(111, projection="3d")
        if df_active is not None and "Statut commercial" in df_active.columns:
            cats = df_active.loc[row_coords.index, "Statut commercial"].astype("category")
            palette = sns.color_palette("tab10", len(cats.cat.categories))
            for cat, color in zip(cats.cat.categories, palette):
                mask = cats == cat
                ax.scatter(
                    row_coords.loc[mask, "F1"],
                    row_coords.loc[mask, "F2"],
                    row_coords.loc[mask, "F3"],
                    s=10,
                    alpha=0.7,
                    color=color,
                    label=str(cat),
                )
            ax.legend(title="Statut commercial", bbox_to_anchor=(1.05, 1), loc="upper left")
        else:
            ax.scatter(row_coords["F1"], row_coords["F2"], row_coords["F3"], s=10, alpha=0.7)
        ax.set_xlabel("F1")
        ax.set_ylabel("F2")
        ax.set_zlabel("F3")
        ax.set_title("PCAmix – individus (3D)")
        plt.tight_layout()
        plt.savefig(output_dir / "pcamix_indiv_plot_3D.png")
        plt.close()
        logger.info("Projection individus PCAmix 3D enregistrée")

    # ─── Cercle des corrélations ─────────────────────────────────────
    if {"F1", "F2"}.issubset(col_coords.columns):
        qcoords = col_coords.loc[[v for v in quant_vars if v in col_coords.index]]
        if not qcoords.empty:
            plt.figure(figsize=(12, 6), dpi=200)
            ax = plt.gca()
            plot_correlation_circle(ax, qcoords, "PCAmix – cercle des corrélations")
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
            for tick in axes_plot[i].get_xticklabels():
                tick.set_horizontalalignment("right")
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


def best_clustering_labels(X: np.ndarray) -> tuple[np.ndarray, str, Any]:
    """Compare several clustering approaches and return the best labels.

    Evaluates KMeans, AgglomerativeClustering and DBSCAN using the silhouette
    score. The configuration chosen is the one yielding the highest score.
    """
    from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
    from sklearn.metrics import silhouette_score

    candidates: list[tuple[np.ndarray, str, Any, float]] = []

    for k in range(2, min(10, len(X))):
        labels = KMeans(n_clusters=k, random_state=0).fit_predict(X)
        score = silhouette_score(X, labels)
        candidates.append((labels, f"kmeans_{k}", k, score))

    for k in range(2, min(10, len(X))):
        labels = AgglomerativeClustering(n_clusters=k).fit_predict(X)
        score = silhouette_score(X, labels)
        candidates.append((labels, f"agg_{k}", k, score))

    for eps in (0.3, 0.5, 0.7, 1.0):
        db = DBSCAN(eps=eps, min_samples=5).fit(X)
        labels = db.labels_
        if len(set(labels)) > 1 and (labels != -1).sum() > 1:
            mask = labels != -1
            score = silhouette_score(X[mask], labels[mask])
            candidates.append((labels, f"dbscan_{eps}", eps, score))

    if not candidates:
        return np.zeros(len(X), dtype=int), "none", None

    best = max(candidates, key=lambda c: c[3])
    return best[0], best[1], best[2]


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
        n_features = None
        if df_active is not None and quant_vars is not None and qual_vars is not None:
            n_features = len(quant_vars) + len(qual_vars)
        elif isinstance(info["embeddings"], pd.DataFrame):
            n_features = info["embeddings"].shape[1]
        if n_features is not None:
            kaiser = sum(1 for eig in np.array(inertias) * n_features if eig > 1)
        else:
            kaiser = sum(1 for eig in inertias if eig > 1)
        cum_inertia = sum(inertias) * 100 if len(inertias) > 0 else None

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
    ax = plt.gca()
    sns.heatmap(
        df_norm,
        annot=df_comp,
        fmt=".2f",
        cmap="coolwarm",
        ax=ax,
        vmin=0,
        vmax=1,
    )
    ax.set_title("Évaluation des méthodes")
    plt.yticks(rotation=0)
    plt.tight_layout()
    (output_dir / "methods_heatmap.png").parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "methods_heatmap.png")
    plt.close()

    return df_comp


def compare_method_clusters(
    results_dict: Dict[str, Dict[str, Any]],
    output_dir: Path,
    *,
    n_clusters: int = 3,
) -> pd.DataFrame:
    """Compute clustering consistency between methods using ARI."""

    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score

    if not results_dict:
        return pd.DataFrame()

    index = next(iter(results_dict.values()))["embeddings"].index
    labels_df = pd.DataFrame(index=index)
    for method, info in results_dict.items():
        emb = info["embeddings"]
        labels = KMeans(n_clusters=n_clusters, random_state=0).fit_predict(emb.values)
        labels_df[method] = labels

    labels_df.to_csv(output_dir / "cluster_labels.csv")

    methods = labels_df.columns.tolist()
    ari = pd.DataFrame(np.eye(len(methods)), index=methods, columns=methods)
    for i, m1 in enumerate(methods):
        for j in range(i + 1, len(methods)):
            m2 = methods[j]
            score = adjusted_rand_score(labels_df[m1], labels_df[m2])
            ari.loc[m1, m2] = score
            ari.loc[m2, m1] = score

    ari.to_csv(output_dir / "methods_similarity.csv")

    fig, ax = plt.subplots(figsize=(12, 6), dpi=200)
    sns.heatmap(ari, annot=True, vmin=0, vmax=1, cmap="coolwarm", ax=ax)
    ax.set_title("Corrélations entre méthodes")
    plt.yticks(rotation=0)
    fig.tight_layout()
    plt.savefig(output_dir / "methods_similarity_heatmap.png")
    plt.close(fig)

    return ari


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
    parser.add_argument("--best-params", help="CSV des meilleurs paramètres")
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

    if args.best_params:
        bp = load_best_params(Path(args.best_params))
        for meth, params in bp.items():
            config.setdefault(meth, {}).update(params)
            effective_config.setdefault(meth, {}).update(params)
        config["optimize_params"] = False

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
    n_jobs = int(config.get("n_jobs", 2))
    logger.info("Parallel executor using %d workers", n_jobs)
    methods = config.get("methods", [])
    results: Dict[str, Dict[str, Any]] = {}

    start: Dict[str, float] = {}
    futures: Dict[str, Any] = {}
    with ThreadPoolExecutor(max_workers=16) as executor:
        if "famd" in methods:
            start["FAMD"] = time.time()
            famd_cfg = config.get("famd", {})
            futures["FAMD"] = executor.submit(
                run_famd,
                df_active,
                quant_vars,
                qual_vars,
                famd_cfg.get("n_components"),
                famd_cfg,
                optimize=optimize_params,
            )
        if "pca" in methods:
            start["PCA"] = time.time()
            futures["PCA"] = executor.submit(
                run_pca,
                df_active,
                quant_vars,
                qual_vars,
                output_dir / "PCA",
                optimize=optimize_params,
                **config.get("pca", {})
            )
        if "mca" in methods:
            start["MCA"] = time.time()
            futures["MCA"] = executor.submit(
                run_mca,
                df_active,
                quant_vars,
                qual_vars,
                output_dir / "MCA",
                optimize=optimize_params,
                **config.get("mca", {})
            )
        if "mfa" in methods:
            start["MFA"] = time.time()
            mfa_cfg = config.get("mfa", {})
            futures["MFA"] = executor.submit(
                run_mfa,
                df_active,
                quant_vars,
                qual_vars,
                output_dir / "MFA",
                n_components=mfa_cfg.get("n_components"),
                groups=mfa_cfg.get("groups"),
                optimize=optimize_params,
                normalize=mfa_cfg.get("normalize", True),
            )
        if "pcamix" in methods:
            start["PCAmix"] = time.time()
            futures["PCAmix"] = executor.submit(
                run_pcamix,
                df_active,
                quant_vars,
                qual_vars,
                output_dir / "PCAmix",
                optimize=optimize_params,
                **config.get("pcamix", {})
            )

        for name, fut in futures.items():
            res = fut.result()
            rt = time.time() - start[name]
            if name == "FAMD":
                famd_model, famd_inertia, famd_rows, famd_cols, famd_contrib = res
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
                export_famd_results(
                    famd_model,
                    famd_inertia,
                    famd_rows,
                    famd_cols,
                    famd_contrib,
                    quant_vars,
                    qual_vars,
                    output_dir / "FAMD",
                    df_active=df_active,
                )
                del famd_model, famd_rows, famd_cols, famd_contrib
            elif name == "PCA":
                pca_model, pca_inertia, pca_rows, pca_cols, pca_contrib = res
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
                    output_dir / "PCA",
                    quant_vars,
                    df_active=df_active,
                )
                del pca_model, pca_rows, pca_cols
            elif name == "MCA":
                mca_model, mca_inertia, mca_rows, mca_cols, mca_contrib = res
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
                    output_dir / "MCA",
                    qual_vars,
                    df_active=df_active,
                )
                del mca_model, mca_rows, mca_cols
            elif name == "MFA":
                mfa_model, mfa_rows = res
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
                    output_dir / "MFA",
                    quant_vars,
                    qual_vars,
                    df_active=df_active,
                    segment_col=config.get("mfa", {}).get("segment_col", "Statut commercial"),
                )
                del mfa_model, mfa_rows
            elif name == "PCAmix":
                mdpca_model, mdpca_inertia, mdpca_rows, mdpca_cols = res
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
                    output_dir / "PCAmix",
                    quant_vars,
                    qual_vars,
                    df_active=df_active,
                )
                del mdpca_model, mdpca_rows, mdpca_cols

    futures.clear();
    start.clear()
    with ThreadPoolExecutor(max_workers=16) as executor:
        if "umap" in methods:
            start["UMAP"] = time.time()
            futures["UMAP"] = executor.submit(
                run_umap,
                df_active,
                quant_vars,
                qual_vars,
                output_dir / "UMAP",
                optimize=optimize_params,
                **config.get("umap", {}),
            )
        if "pacmap" in methods:
            start["PaCMAP"] = time.time()
            futures["PaCMAP"] = executor.submit(
                run_pacmap,
                df_active,
                quant_vars,
                qual_vars,
                output_dir / "PaCMAP",
                optimize=optimize_params,
                **config.get("pacmap", {}),
            )
        if "phate" in methods:
            start["PHATE"] = time.time()
            futures["PHATE"] = executor.submit(
                run_phate,
                df_active,
                quant_vars,
                qual_vars,
                output_dir / "PHATE",
                optimize=optimize_params,
                **config.get("phate", {}),
            )
        if "tsne" in methods and "FAMD" in results:
            start["TSNE"] = time.time()
            futures["TSNE"] = executor.submit(
                run_tsne,
                results["FAMD"]["embeddings"],
                df_active,
                output_dir / "TSNE",
                optimize=optimize_params,
                **config.get("tsne", {}),
            )
        elif "tsne" in methods:
            logger.warning("t-SNE ignoré : embeddings FAMD indisponibles")

        for name, fut in futures.items():
            res = fut.result()
            rt = time.time() - start[name]
            if name == "UMAP":
                umap_model, umap_df = res
                export_umap_results(umap_df, df_active, output_dir / "UMAP")
                results["UMAP"] = {"embeddings": umap_df, "inertia": None, "runtime": rt}
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
                    "metric": getattr(umap_model, "metric", None),
                })
                del umap_model, umap_df
            elif name == "PaCMAP":
                pacmap_model, pacmap_df = res
                if pacmap_model is not None:
                    export_pacmap_results(pacmap_df, df_active, output_dir / "PaCMAP")
                    results["PaCMAP"] = {"embeddings": pacmap_df, "inertia": None, "runtime": rt}
                    logger.info(
                        f"PaCMAP : réalisé en {rt:.1f}s (n_neighbors={pacmap_model.n_neighbors})"
                    )
                    effective_config.setdefault("pacmap", {}).update({
                        "n_components": pacmap_model.n_components,
                        "n_neighbors": pacmap_model.n_neighbors,
                        "MN_ratio": pacmap_model.MN_ratio,
                        "FP_ratio": pacmap_model.FP_ratio,
                    })
                del pacmap_model, pacmap_df
            elif name == "PHATE":
                phate_op, phate_df = res
                if phate_op is not None:
                    export_phate_results(phate_df, df_active, output_dir / "PHATE")
                    results["PHATE"] = {"embeddings": phate_df, "inertia": None, "runtime": rt}
                    logger.info(f"PHATE : réalisé en {rt:.1f}s")
                    effective_config.setdefault("phate", {}).update({
                        "n_components": phate_op.n_components,
                        "knn": getattr(phate_op, "knn", None),
                        "t": getattr(phate_op, "t", None),
                    })
                del phate_op, phate_df
            elif name == "TSNE":
                tsne_model, tsne_df, tsne_metrics = res
                export_tsne_results(tsne_df, df_active, output_dir / "TSNE", tsne_metrics)
                results["TSNE"] = {"embeddings": tsne_df, "inertia": None, "runtime": rt}
                logger.info(
                    "t-SNE : perplexity=%s, %.1fs",
                    getattr(tsne_model, "perplexity", "?"),
                    rt,
                )
                effective_config.setdefault("tsne", {})["perplexity"] = getattr(tsne_model, "perplexity", None)
                del tsne_model, tsne_df
    # 8. Évaluation croisée
    comp_df = evaluate_methods(
        results,
        output_dir,
        n_clusters=3,
        df_active=df_active,
        quant_vars=quant_vars,
        qual_vars=qual_vars,
    )

    # 8b. Comparaisons multi-méthodes
    scree_methods = [
        m
        for m in (
            "FAMD",
            "PCA",
            "MCA",
            "MFA",
            "PCAmix",
        )
        if m in results
    ]
    scatter_methods = [
        m
        for m in (
            "FAMD",
            "PCA",
            "MCA",
            "MFA",
            "PCAmix",
            "UMAP",
            "TSNE",
            "PaCMAP",
            "PHATE",
        )
        if m in results
    ]
    method_order = []
    for m in scree_methods + scatter_methods:
        if m not in method_order:
            method_order.append(m)
    plot_multimethod_results(
        {m: results[m] for m in method_order},
        df_active,
        comp_df.loc[comp_df.index.intersection(method_order)],
        output_dir,
        scree_methods=scree_methods,
        scatter_methods=scatter_methods,
    )

    compare_method_clusters(results, output_dir, n_clusters=3)

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
    ``PCA/pca_scree_plot.png``, ``MCA/mca_scree_plot.png``,
    ``MFA/mfa_scree_plot.png``, ``PCAmix/pcamix_scree_plot.png``,
    ``UMAP/umap_scatter.png``, ``TSNE/tsne_scatter.png``,
    ``PHATE/phate_scatter.png``, ``PaCMAP/pacmap_scatter.png`` and
    ``methods_heatmap.png``.
    """
    logger = logging.getLogger(__name__)
    pdf_path = output_dir / "phase4_report.pdf"
    figures = [
        str(Path("PCA") / "pca_scree_plot.png"),
        str(Path("MCA") / "mca_scree_plot.png"),
        str(Path("MFA") / "mfa_scree_plot.png"),
        str(Path("PCAmix") / "pcamix_scree_plot.png"),
        str(Path("UMAP") / "umap_scatter.png"),
        str(Path("TSNE") / "tsne_scatter.png"),
        str(Path("PHATE") / "phate_scatter.png"),
        str(Path("PaCMAP") / "pacmap_scatter.png"),
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
            fig.tight_layout()
            fig.text(
                0.99,
                0.01,
                f"{img_path.resolve().parent} | {img_path.name}",
                ha="right",
                va="bottom",
                fontsize=6,
                color="gray",
            )
            pdf.savefig(fig)
            plt.close(fig)
    return pdf_path


def generate_pdf(output_dir: Path, pdf_name: str = "phase4_rapport_complet.pdf") -> Path:
    logger = logging.getLogger(__name__)
    """Assemble toutes les figures PNG en un PDF multi-pages classé par méthode."""

    methods = [
        "FAMD",
        "PCA",
        "MCA",
        "MFA",
        "PCAmix",
        "UMAP",
        "TSNE",
        "PHATE",
        "PaCMAP",
    ]
    pdf_path = output_dir / pdf_name

    with PdfPages(pdf_path) as pdf:
        # Page de garde
        fig, ax = plt.subplots(figsize=(8.27, 11.69), dpi=200)
        today = datetime.datetime.now().strftime("%Y-%m-%d")
        ax.text(
            0.5,
            0.6,
            "Analyse Factorielle – Rapport Phase 4",
            fontsize=20,
            ha="center",
            va="center",
        )
        ax.text(
            0.5,
            0.4,
            f"Généré le {today}",
            fontsize=12,
            ha="center",
            va="center",
        )
        ax.axis("off")
        pdf.savefig(fig, dpi=300)
        plt.close(fig)

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
                fig_w, fig_h = 12, 6
                fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=200)
                ax.imshow(img)
                ax.axis("off")
                fig.tight_layout()
                fig.text(
                    0.99,
                    0.01,
                    f"{img_path.resolve().parent} | {img_path.name}",
                    ha="right",
                    va="bottom",
                    fontsize=6,
                    color="gray",
                )
                pdf.savefig(fig, dpi=300)
                plt.close(fig)

        # Figures segmentation
        seg_dir = output_dir / "segments"
        seg_imgs = sorted(seg_dir.glob("segment_*.png")) if seg_dir.exists() else []
        if seg_imgs:
            fig, ax = plt.subplots(figsize=(8.27, 11.69), dpi=200)
            ax.text(0.5, 0.5, "Segments", fontsize=24, ha="center", va="center")
            ax.axis("off")
            pdf.savefig(fig, dpi=300)
            plt.close(fig)

            for img_path in seg_imgs:
                img = plt.imread(img_path)
                fig, ax = plt.subplots(figsize=(12, 6), dpi=200)
                ax.imshow(img)
                ax.axis("off")
                fig.tight_layout()
                fig.text(
                    0.99,
                    0.01,
                    f"{img_path.resolve().parent} | {img_path.name}",
                    ha="right",
                    va="bottom",
                    fontsize=6,
                    color="gray",
                )
                pdf.savefig(fig, dpi=300)
                plt.close(fig)

        # Figures comparatives globales
        global_imgs = [
            "multi_scree.png",
            "multi_scatter.png",
            "methods_similarity_heatmap.png",
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
                fig.tight_layout()
                fig.text(
                    0.99,
                    0.01,
                    f"{img_path.resolve().parent} | {img_path.name}",
                    ha="right",
                    va="bottom",
                    fontsize=6,
                    color="gray",
                )
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
            "methods_heatmap.png": "Heatmap évaluation des méthodes",
            "methods_comparison.csv": "Tableau comparatif des méthodes",
            "multi_scree.png": "Éboulis comparatif",
            "multi_scatter.png": "Projection comparée des méthodes",
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
        if len(rel.parts) > 1 and rel.parts[0] in {"FAMD", "PCA", "MCA", "MFA", "PCAmix", "UMAP", "TSNE", "PaCMAP", "PHATE"}:
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
