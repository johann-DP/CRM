import sys
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from itertools import combinations
import seaborn as sns
import matplotlib.pyplot as plt
import io
import unicodedata
import re
from PIL import Image

# ─── 1) Harmonisation graphique ───────────────────────────
sns.set_theme(style="whitegrid", font_scale=1.3)
plt.rcParams.update({
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "figure.dpi": 300
})

# ─── 2) Définition des chemins ────────────────────────────
RAW_PATH = r"D:\DATAPREDICT\DATAPREDICT 2024\Missions\Digora\export_everwin (19).xlsx"
OUTPUT_DIR = Path(r"D:\DATAPREDICT\DATAPREDICT 2024\Missions\Digora\phase3_output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = OUTPUT_DIR / "phase3.log"

# ─── 3) Configuration du logger ────────────────────────────
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Handler console
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.INFO)
ch.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(ch)

# Handler fichier
fh = logging.FileHandler(LOG_PATH, mode='w', encoding='utf-8')
fh.setLevel(logging.INFO)
fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(fh)

logger.info(f"Dossier d'export Phase 3 prêt : {OUTPUT_DIR}")

# ─── 4) Chargement des données ─────────────────────────────
try:
    df = pd.read_excel(RAW_PATH)
    logger.info(f"Données chargées : {df.shape[0]} lignes × {df.shape[1]} colonnes")
except FileNotFoundError:
    logger.error(f"Fichier introuvable : {RAW_PATH}")
    sys.exit(1)
except Exception as e:
    logger.error(f"Erreur lors du chargement : {e}")
    sys.exit(1)

# ─── Analyse amont détaillée des données ───────────────────
logger.info("Démarrage de l’analyse amont détaillée des données")

# 1) Informations générales
logger.info(f"Colonnes du DataFrame : {list(df.columns)}")
# Types et non-null counts (capturés dans un buffer)
logger.info("Types et non-null counts :")
buf = io.StringIO()
df.info(buf=buf)
info_str = buf.getvalue()
buf.close()
logger.info("\n" + info_str)

# 2) Résumé numérique enrichi
num = df.select_dtypes(include='number')
desc_num = num.describe().T
desc_num['missing'] = num.isna().sum()
desc_num['missing_pct'] = (desc_num['missing'] / len(df) * 100).round(2)
desc_num['nunique'] = num.nunique()
desc_num['skew'] = num.skew().round(2)
desc_num['kurtosis'] = num.kurtosis().round(2)
logger.info("Résumé numérique enrichi :\n" + desc_num.to_string())
desc_num.to_csv(OUTPUT_DIR / "phase3_numeric_overview.csv")

# 3) Résumé catégoriel
cat = df.select_dtypes(include=['object', 'category', 'bool'])
cat_overview = []
for col in cat:
    ser = df[col]
    vc = ser.value_counts(dropna=False)
    top = vc.index[0] if not vc.empty else None
    freq = vc.iloc[0] if not vc.empty else 0
    missing = ser.isna().sum()
    cat_overview.append({
        'variable': col,
        'dtype': ser.dtype,
        'nunique': ser.nunique(dropna=False),
        'top': top,
        'freq_top': int(freq),
        'missing': int(missing),
        'missing_pct': round(missing / len(df) * 100, 2)
    })
cat_df = pd.DataFrame(cat_overview)
logger.info("Résumé catégoriel :\n" + cat_df.to_string(index=False))
cat_df.to_csv(OUTPUT_DIR / "phase3_categorical_overview.csv", index=False)


# ─── Détection de signaux faibles dans les variables catégorielles ─────────────
def detecter_categories_rares(df_all,
                              seuil_modalite=5,
                              seuil_combo=3,
                              couples_logiques=(('Type', 'Pilier'),
                                                ('Statut commercial', 'Statut production')),
                              out_dir=OUTPUT_DIR,
                              fig_container=None):
    """
    1) Marque les modalités rares (< seuil_modalite occurrences) -> flag_cat_rare
    2) Marque les combinaisons rares (< seuil_combo) sur des couples_logiques
       -> flag_combo_rare
    3) Exporte deux CSV et produit un barplot des modalités les plus rares.
    """
    cat_cols = df_all.select_dtypes(include=['object', 'category', 'bool']).columns
    rares_list = []

    # ---- Modèles rares --------------------------------------------------------
    df_all['flag_cat_rare'] = False
    for col in cat_cols:
        vc = df_all[col].value_counts(dropna=False)
        rares = vc[vc < seuil_modalite]
        if not rares.empty:
            rares_list.append(pd.DataFrame({
                'variable': col,
                'modalite': rares.index.astype(str),
                'freq': rares.values
            }))
            df_all.loc[df_all[col].isin(rares.index), 'flag_cat_rare'] = True

    # ---- Combinaisons rares --------------------------------------------------
    df_all['flag_combo_rare'] = False
    combo_list = []

    for col1, col2 in couples_logiques:
        # calcul des fréquences de chaque couple de modalités
        vc_combo = df_all.groupby([col1, col2]).size()
        rares_combo = vc_combo[vc_combo < seuil_combo]
        if not rares_combo.empty:
            for (val1, val2), freq in rares_combo.items():
                combo_list.append({
                    'variable1': col1,
                    'variable2': col2,
                    'modalite1': val1,
                    'modalite2': val2,
                    'freq': int(freq)
                })
                # marquer les lignes correspondantes
                mask = (df_all[col1] == val1) & (df_all[col2] == val2)
                df_all.loc[mask, 'flag_combo_rare'] = True

    # export du détail des combinaisons rares
    if combo_list:
        combo_df = pd.DataFrame(combo_list)
        combo_df.to_csv(out_dir / "phase3_rare_combinations.csv", index=False)
        logger.info("Export combinaisons rares : phase3_rare_combinations.csv")
    else:
        logger.info("Aucune combinaison rare détectée (seuil < %d)", seuil_combo)

    # ---- Nouvelle synthèse visuelle des raretés ------------------------------
    if rares_list:
        rares_df_all = pd.concat(rares_list, ignore_index=True)

        # Filtre métier : on écarte les variables où la rareté est normale
        vars_exclues_rarete = {
            # identifiants ou clés quasi uniques
            'Code', 'Titre', 'SharePoint', 'idSALES',
            # coordonnées et infos contact
            'Adresse société', 'Code postal société', 'Ville société',
            'SIREN société', 'SIRET société', 'Téléphone société', 'E-mail société',
            # champs texte ou commentaires
            'Description', 'Commentaire Admin'
        }

        rares_df_all = rares_df_all[~rares_df_all['variable'].isin(vars_exclues_rarete)]
        if rares_df_all.empty:
            logger.info("Toutes les raretés appartiennent à des champs exclus (identifiants ou texte libre)")
            return

        # A) combien de modalités rares pour chaque variable ?
        count_per_var = (
                            rares_df_all.groupby('variable')['modalite']
                            .nunique()
                            .sort_values(ascending=False)
                        )[:16]

        # Bar-chart horizontal (# modalités rares par variable)
        fig, ax = plt.subplots(figsize=(12, 8), dpi=200)
        sns.barplot(
            x=count_per_var.values,
            y=count_per_var.index,
            color=sns.color_palette('deep')[0],
            ax=ax
        )
        for patch in ax.patches:
            patch.set_edgecolor('black')
        ax.set_xlabel(f"Nombre de modalités rares (< {seuil_modalite} occ.)")
        ax.set_ylabel("Variable catégorielle")
        ax.set_title("Variables contenant le plus de modalités rares")
        plt.tight_layout()
        bar_var_path = out_dir / "phase3_bar_rare_variables.png"
        fig.savefig(bar_var_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        if fig_container is not None:
            fig_container.append(bar_var_path)
        logger.info("Bar-chart rare variables généré : phase3_bar_rare_variables.png")

        # B) extrait qualitatif : 5 modalités les plus rares pour les 3 variables en tête
        top_vars = count_per_var.head(3).index.tolist()
        excerpt_rows = []
        for var in top_vars:
            excerpt_rows.append(
                rares_df_all.loc[rares_df_all['variable'] == var]
                .sort_values('freq')
                .head(5)
            )
        excerpt_df = pd.concat(excerpt_rows, ignore_index=True)
        # libellé tronqué à 40 caractères pour visibilité
        excerpt_df['modalite_tronquee'] = (
                excerpt_df['modalite'].astype(str).str.slice(0, 40) +
                np.where(excerpt_df['modalite'].str.len() > 40, '…', '')
        )
        excerpt_df.to_csv(out_dir / "phase3_rare_modalities_excerpt.csv",
                          index=False)
        logger.info("Export extrait modalités rares : phase3_rare_modalities_excerpt.csv")


# 4) Dates : min / max
date_cols = [c for c in df.columns if 'date' in c.lower()]
date_overview = []
for c in date_cols:
    try:
        s = pd.to_datetime(df[c], errors='coerce')
        date_overview.append({
            'variable': c,
            'min': s.min(),
            'max': s.max(),
            'missing': int(s.isna().sum())
        })
    except Exception:
        continue
date_df = pd.DataFrame(date_overview)
if not date_df.empty:
    logger.info("Aperçu des dates :\n" + date_df.to_string(index=False))
    date_df.to_csv(OUTPUT_DIR / "phase3_date_overview.csv", index=False)
else:
    logger.warning("Aucune colonne date valide détectée pour l’aperçu.")

logger.info("Analyse amont terminée – fichiers CSV générés dans phase3_output")

# ─── Détection des anomalies de dates hors plage 1990–2050 ─────────────────
import pandas as _pd

# Plage de validité
min_date = _pd.Timestamp('1990-01-01')
max_date = _pd.Timestamp('2050-12-31')

# Initialisation du flag global
df['flag_date_anomalie'] = False


def sanitize_filename(name: str) -> str:
    """
    Transforme un nom de colonne en un identifiant sûr pour les fichiers :
    remplace les caractères non-alphanumériques par un underscore.
    """
    return re.sub(r'[^0-9a-zA-Z]+', '_', name).strip('_').lower()


for c in date_cols:
    # reconvertit en datetime pour être sûr
    s = _pd.to_datetime(df[c], errors='coerce')
    # masque anomalies
    mask = s.lt(min_date) | s.gt(max_date)
    # crée un flag par colonne si besoin
    safe = sanitize_filename(c)
    df[f'flag_date_anomalie_{safe}'] = mask
    # cumul global
    df['flag_date_anomalie'] |= mask

# Export des enregistrements ayant au moins une date anormale
anom_df = df[df['flag_date_anomalie']].copy()
if not anom_df.empty:
    anom_df.to_csv(OUTPUT_DIR / "phase3_date_anomalies.csv", index=False)
    logger.info(
        f"Export anomalies dates (hors 1990–2050) : phase3_date_anomalies.csv "
        f"({len(anom_df)} enregistrements)"
    )
else:
    logger.info("Aucune anomalie de date détectée hors plage 1990–2050")

# ─── Étape A – Statistiques descriptives et typologie ───────
from scipy.stats import skew  # (Pandas est déjà importé plus haut)

logger.info("Démarrage de l’étape A : statistiques descriptives et typologie")

# Logger spécifique pour les statistiques détaillées
stats_log = OUTPUT_DIR / "phase3_stats.log"
slogger = logging.getLogger("phase3_stats")
slogger.setLevel(logging.INFO)
sh = logging.FileHandler(stats_log, mode='w', encoding='utf-8')
sh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
slogger.addHandler(sh)

# Sélection des variables numériques (hors booléens)
num_cols = df.select_dtypes(include=[int, float]).columns.tolist()
slogger.info(f"{len(num_cols)} variables numériques détectées")

# Calcul des statistiques pour chaque variable numérique
stats = []
for col in num_cols:
    ser = df[col].dropna()
    stats.append({
        "variable": col,
        "count": ser.count(),
        "nunique": ser.nunique(),
        "min": ser.min(),
        "25%": ser.quantile(0.25),
        "50%": ser.median(),
        "75%": ser.quantile(0.75),
        "max": ser.max(),
        "skewness": float(skew(ser)) if ser.count() > 2 else None
    })
    # Typologie simple des variables numériques
    if ser.nunique() <= 2:
        slogger.warning(f"{col} → faible variabilité ({ser.nunique()} valeurs uniques)")
    if ser.nunique() > 2 and abs(skew(ser)) > 1:
        slogger.info(f"{col} → forte asymétrie (skewness={skew(ser):.2f})")

# Export du résumé des stats
stats_df = pd.DataFrame(stats)
stats_df.to_csv(OUTPUT_DIR / "phase3_var_stats.csv", index=False)
slogger.info("CSV des stats descriptives généré : phase3_var_stats.csv")

# Vérification de la présence de colonnes critiques (dates, montants, marges, durées)
critical = {
    "dates": [c for c in df.columns if "date" in c.lower()],
    "montants": [c for c in df.columns if any(k in c.lower() for k in ["recette", "budget", "montant"])],
    "marges": [c for c in df.columns if "marge" in c.lower()],
    "durées": [c for c in df.columns if "durée" in c.lower()]
}
for key, cols in critical.items():
    if not cols:
        slogger.error(f"Aucune colonne détectée pour : {key}")
    else:
        slogger.info(f"{key.capitalize()} détectées : {cols}")

logger.info("Étape A terminée. Vérifiez phase3_var_stats.csv et phase3_stats.log")

# ─── Détection univariée des outliers ───────────────────────
logger.info("Démarrage de la détection univariée des outliers")
univ_counts_list = []  # Pour suivre le nombre d'outliers par variable
outliers_parts = []
for col in num_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    mask = (df[col] < lower) | (df[col] > upper)
    count = int(mask.sum())
    if count > 0:
        count_low = int((df[col] < lower).sum())
        count_high = int((df[col] > upper).sum())
        logger.info(
            f"{col:30s} → {count:5d} outliers univariés (dont {count_low} sous le min, {count_high} au-delà du max)")
    else:
        logger.info(f"{col:30s} → {count:5d} outliers univariés")
    # Ajouter un flag dans le DataFrame
    df.loc[mask, f"flag_univ_{col}"] = True
    # Collecter les lignes outliers pour export (toutes variables numériques + flag)
    tmp = df.loc[mask, num_cols + [f"flag_univ_{col}"]].copy()
    tmp['variable'] = col
    # Indiquer si outlier inférieur ou supérieur
    tmp['outlier_side'] = np.where(tmp[col] < lower, 'lower', 'upper')
    univ_counts_list.append({"variable": col, "count": count})
    outliers_parts.append(tmp)

# Concaténation et export des outliers univariés
outliers_univ = pd.concat(outliers_parts, ignore_index=True).drop_duplicates()
outliers_univ.to_csv(OUTPUT_DIR / "phase3_outliers_univariate.csv", index=False)

# Jeu de données nettoyé sans outliers univariés
cleaned_univ = df[~df.filter(like="flag_univ_").any(axis=1)].copy()
cleaned_univ.to_csv(OUTPUT_DIR / "phase3_cleaned_univ.csv", index=False)
logger.info(
    f"Export univ terminé : {outliers_univ.shape[0]} outliers univariés exportés, {cleaned_univ.shape[0]} lignes restantes (jeu nettoyé)")

# ─── Test de sensibilité univariée : plusieurs seuils IQR ────────────────
thresholds = [1.5, 2.5, 3.0]
sens_list = []
for col in num_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    row = {"variable": col}
    for thr in thresholds:
        low_thr = Q1 - thr * IQR
        up_thr = Q3 + thr * IQR
        mask_thr = (df[col] < low_thr) | (df[col] > up_thr)
        # nombre d’outliers pour ce seuil
        row[f"count_{thr}xIQR"] = int(mask_thr.sum())
    sens_list.append(row)

# Export CSV de comparaison des seuils
sens_df = pd.DataFrame(sens_list)
sens_df.to_csv(OUTPUT_DIR / "phase3_univ_threshold_comparison.csv",
               index=False)
logger.info("Export comparaison des seuils univariés : "
            "phase3_univ_threshold_comparison.csv")

# ─── Détection multivariée des outliers ────────────────────
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

logger.info("Démarrage détection multivariée des outliers")

# Variables multivariées pour IF/LOF
outliers_cols = [
    'Total recette actualisé',
    'Total recette réalisé',
    'Total recette produit',
    'Budget client estimé'
]

# 1) Référence brute (NA→0)
X_raw = df[outliers_cols].fillna(0)
# Isolation Forest brute
iforest_raw = IsolationForest(n_estimators=100, contamination='auto', random_state=42)
y_if_raw = iforest_raw.fit_predict(X_raw)
raw_if_count = int((y_if_raw == -1).sum())
# Local Outlier Factor brut
lof_raw = LocalOutlierFactor(n_neighbors=20, contamination='auto')
y_lof_raw = lof_raw.fit_predict(X_raw)
raw_lof_count = int((y_lof_raw == -1).sum())

logger.info(f"[Brut] IF={raw_if_count} outliers, LOF={raw_lof_count} outliers")

# 2) Prétraitement : standardisation + imputation médiane
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[outliers_cols])
imputer = SimpleImputer(strategy='median')
X_proc = imputer.fit_transform(X_scaled)
X_proc_df = pd.DataFrame(X_proc, columns=outliers_cols, index=df.index)

logger.info("Prétraitement multivarié appliqué : StandardScaler + imputation médiane")

# 3) Détection sur données prétraitées

# Isolation Forest prétraité
iforest = IsolationForest(n_estimators=100, contamination='auto', random_state=42)
y_if = iforest.fit_predict(X_proc_df)
new_if_count = int((y_if == -1).sum())
df['flag_iforest'] = (y_if == -1)
diff_if = int(np.sum((y_if == -1) != (y_if_raw == -1)))
logger.info(f"[Prétraité] IF={new_if_count} outliers "
            f"({diff_if} labels modifiés vs brut {raw_if_count})")
out_if = df[df['flag_iforest']]
out_if.to_csv(OUTPUT_DIR / "phase3_outliers_iforest.csv", index=False)

# Local Outlier Factor prétraité
lof = LocalOutlierFactor(n_neighbors=20, contamination='auto')
y_lof = lof.fit_predict(X_proc_df)
new_lof_count = int((y_lof == -1).sum())
df['flag_lof'] = (y_lof == -1)
diff_lof = int(np.sum((y_lof == -1) != (y_lof_raw == -1)))
logger.info(f"[Prétraité] LOF={new_lof_count} outliers "
            f"({diff_lof} labels modifiés vs brut {raw_lof_count})")
out_lof = df[df['flag_lof']]
out_lof.to_csv(OUTPUT_DIR / "phase3_outliers_lof.csv", index=False)

# 4) Union des méthodes et export final
df['flag_multivariate'] = df['flag_iforest'] | df['flag_lof']
out_multi = df[df['flag_multivariate']]
out_multi.to_csv(OUTPUT_DIR / "phase3_outliers_multivariate.csv", index=False)
logger.info(f"Multivarié (IF ∪ LOF) → {out_multi.shape[0]} outliers exportés")

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np

# --- 1) PCA sur les données numériques standardisées ---
# outliers_cols = liste de vos variables numériques utilisées pour IF et LOF
X = df[outliers_cols].fillna(0).values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)
df['PC1'] = X_pca[:, 0]
df['PC2'] = X_pca[:, 1]

pca_files = []

# --- 2) Trace des loadings (directions des variables) ---
loadings = pca.components_.T  # shape (n_vars, 2)
plt.figure(figsize=(12, 6), dpi=200)
plt.axhline(0, color='grey', linewidth=1)
plt.axvline(0, color='grey', linewidth=1)
for i, (var, comp) in enumerate(zip(outliers_cols, loadings)):
    # segment simple du centre vers les coordonnées du loading
    plt.plot(
        [0, comp[0]], [0, comp[1]],
        color=sns.color_palette('deep')[i], linewidth=2
    )
    va_align = 'bottom' if var == 'Total recette actualisé' else 'top'
    plt.text(
        comp[0] * 1.1, comp[1] * 1.1,
        var,
        ha='left',
        va=va_align,
        fontsize=8
    )
pc1_logscale_param = 10 ** (-1.2)
pc2_logscale_param = 10 ** (-0.5)
plt.xscale('symlog', linthresh=pc1_logscale_param)
plt.yscale('symlog', linthresh=pc2_logscale_param)
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f} %)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f} %)")
plt.title("PCA – Loadings des variables")
plt.tight_layout()
file_raw = OUTPUT_DIR / "phase3_pca_loadings.png"
plt.savefig(file_raw, dpi=300, bbox_inches="tight")
plt.close()
pca_files.append(file_raw)

# --- 3) Scatter PCA + cercles d’outliers IF / LOF ---
# détermine qui est IF seul, LOF seul ou les deux
flags = df[['flag_iforest', 'flag_lof']].copy()
flags['type'] = np.where(flags['flag_iforest'] & flags['flag_lof'], 'IF ∩ LOF',
                         np.where(flags['flag_iforest'], 'IF seul',
                                  np.where(flags['flag_lof'], 'LOF seul',
                                           'normal')))
df['outlier_type'] = flags['type']

palette = {
    'normal': 'lightgray',
    'IF seul': 'blue',
    'LOF seul': 'orange',
    'IF ∩ LOF': 'purple'
}

plt.figure(figsize=(12, 6), dpi=200)
# tracé manuel par type, sans « normal » en légende
fig, ax = plt.subplots(figsize=(12, 6), dpi=200)
# points normaux (pas de label !)
ax.scatter(
    df.loc[df.outlier_type == 'normal', 'PC1'],
    df.loc[df.outlier_type == 'normal', 'PC2'],
    color=palette['normal'], alpha=0.6, s=30,
    label='_nolegend_'
)
# IF seul
ax.scatter(
    df.loc[df.outlier_type == 'IF seul', 'PC1'],
    df.loc[df.outlier_type == 'IF seul', 'PC2'],
    color=palette['IF seul'], alpha=0.6, s=30,
    label='IF seul'
)
# LOF seul
ax.scatter(
    df.loc[df.outlier_type == 'LOF seul', 'PC1'],
    df.loc[df.outlier_type == 'LOF seul', 'PC2'],
    color=palette['LOF seul'], alpha=0.6, s=30,
    label='LOF seul'
)
# IF ∩ LOF
ax.scatter(
    df.loc[df.outlier_type == 'IF ∩ LOF', 'PC1'],
    df.loc[df.outlier_type == 'IF ∩ LOF', 'PC2'],
    color=palette['IF ∩ LOF'], alpha=0.6, s=30,
    label='IF ∩ LOF'
)

ax.set_xscale('symlog', linthresh=pc1_logscale_param)
ax.set_yscale('symlog', linthresh=pc2_logscale_param)
ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f} %)")
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f} %)")
ax.set_title("Projection PCA – Outliers IsolationForest et LOF")
ax.legend(loc='best', framealpha=0.7)
plt.tight_layout()
fig_path = OUTPUT_DIR / "phase3_pca_outliers.png"
fig.savefig(fig_path, dpi=300, bbox_inches="tight")
plt.close(fig)
pca_files.append(fig_path)

# 4) Jeu de données nettoyé multivarié
cleaned_multi = df[~df['flag_multivariate']].copy()
cleaned_multi.to_csv(OUTPUT_DIR / "phase3_cleaned_multivariate.csv", index=False)
logger.info(f"Jeu nettoyé multivarié : {cleaned_multi.shape[0]} lignes restantes")

# ─── Synthèse et export des outliers – phase3 ───────────────
logger.info("Synthèse et export des outliers – phase3")

# Ajout d'un flag global pour tout outlier (univarié ou multivarié)
df['flag_any_outlier'] = df.filter(like="flag_univ_").any(axis=1) | df['flag_multivariate']

# Construction du résumé global des outliers
common_if_lof = int((df['flag_iforest'] & df['flag_lof']).sum())
summary = {
    "total_records": len(df),
    "outliers_univariés": outliers_univ.shape[0],
    "outliers_iforest": out_if.shape[0],
    "outliers_lof": out_lof.shape[0],
    "outliers_multivariés": out_multi.shape[0],
    "outliers_if_and_lof": common_if_lof,
    "outliers_total": int(df['flag_any_outlier'].sum())
}
summary_df = pd.DataFrame([summary])
summary_df.to_csv(OUTPUT_DIR / "phase3_outlier_summary.csv", index=False)
logger.info("Export du résumé des outliers : phase3_outlier_summary.csv")


# ─── Consolidation globale des outliers ─────────────────────
def consolider_outliers_global(df_all, outliers_cols, output_dir,
                               fig_list_container):
    """
    1) Calcule la répartition % par famille d'algorithme et trace un barplot.
    2) Extrait les 20 enregistrements les plus extrêmes (max écart absolu
       vis-à-vis de la médiane sur les variables financières) et les exporte.
    """
    # ---- 1. Répartition des familles d'outliers --------------------------------
    # mapping exclusif : IF∩LOF > IF seul > LOF seul > Univarié
    cond_if_and_lof = df_all['flag_iforest'] & df_all['flag_lof']
    cond_if_only = df_all['flag_iforest'] & ~df_all['flag_lof']
    cond_lof_only = df_all['flag_lof'] & ~df_all['flag_iforest']
    cond_univ_only = df_all.filter(like="flag_univ_").any(axis=1) & \
                     ~(df_all['flag_iforest'] | df_all['flag_lof'])

    fam_counts = {
        "IF ∩ LOF": int(cond_if_and_lof.sum()),
        "IF seul": int(cond_if_only.sum()),
        "LOF seul": int(cond_lof_only.sum()),
        "Univarié": int(cond_univ_only.sum())
    }
    fam_df = pd.DataFrame(
        [{"famille": k, "count": v,
          "pct": round(v / df_all['flag_any_outlier'].sum() * 100, 2)}
         for k, v in fam_counts.items()]
    )
    fam_df.to_csv(output_dir / "phase3_outlier_families.csv", index=False)

    # ── Barplot % d’outliers par famille (Seaborn) ─────────────────────
    fam_df_sorted = fam_df.sort_values('pct', ascending=False)
    fig, ax = plt.subplots(figsize=(12, 5), dpi=200)
    sns.barplot(
        x='pct', y='famille',
        data=fam_df_sorted,
        edgecolor="black",
        color=sns.color_palette('deep')[0],
        ax=ax
    )
    ax.set_xlabel("Pourcentage d'outliers (%)")
    ax.set_ylabel("Famille d'algorithme")
    ax.set_title("Répartition des outliers par famille d'algorithme")
    for bar in ax.patches:
        w = bar.get_width()
        ax.text(w + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{w:.1f}%",
                va='center')
    plt.tight_layout()
    bar_path = output_dir / "phase3_bar_outlier_families.png"
    fig.savefig(bar_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    fig_list_container.append(bar_path)

    # ---- 2. Top-20 des enregistrements extrêmes --------------------------------
    outliers = df_all[df_all['flag_any_outlier']].copy()
    if not outliers.empty:
        # écart absolu à la médiane (robuste aux valeurs extrêmes)
        med = df_all[outliers_cols].median()
        dev = (outliers[outliers_cols] - med).abs()
        outliers['max_abs_dev'] = dev.max(axis=1)
        top20 = (
            outliers
            .sort_values('max_abs_dev', ascending=False)
            .head(20)
            .drop(columns=['max_abs_dev'])  # la garder ? -> retirer si inutile
        )
        top20.to_csv(output_dir / "phase3_outliers_top20.csv", index=False)
        logger.info("Export Top 20 des outliers extrêmes : phase3_outliers_top20.csv")

        # ─── 2.b) Génération d’un commentaire texte pour chaque Top 20 ─────────────
        commentary_lines = []
        for _, row in top20.iterrows():
            commentary_lines.append(
                f"ID {row['Code']}: Client={row['Client']}, "
                f"Montant={row['Total recette réalisé']}, "
                f"Durée={row['Durée engagement (mois)']}, "
                f"Statut={row['Statut production']}, "
                f"Catégorie={row['Catégorie']}"
            )
        comment_path = output_dir / "phase3_top20_commentary.txt"
        with open(comment_path, "w", encoding="utf-8") as f:
            f.write("\n".join(commentary_lines))
        logger.info("Export commentaires Top 20 : phase3_top20_commentary.txt")

    else:
        logger.warning("Aucun outlier trouvé pour constituer un Top 20.")


logger.info("Consolidation globale des outliers terminée (barplot + Top 20).")

# ─── Typologie multivariée des outliers – abandon du clustering ─────────────
logger.info(
    "Tentatives de typologie (clustering) des outliers multivariés non concluantes – étape abandonnée"
)

# Sauvegarde finale des jeux de données (complet+flags, outliers seuls, jeu nettoyé)
# (On nettoie d'abord les colonnes temporaires de plot si présentes)
drop_cols = [col for col in ["Avant_vente_plot", "duree_bucket"] if col in df.columns]
if drop_cols:
    df.drop(columns=drop_cols, inplace=True, errors='ignore')
outliers_all = df[df['flag_any_outlier']].copy()
cleaned_all = df[~df['flag_any_outlier']].copy()
df.to_csv(OUTPUT_DIR / "phase3_all_flags.csv", index=False)
outliers_all.to_csv(OUTPUT_DIR / "phase3_outliers_all.csv", index=False)
cleaned_all.to_csv(OUTPUT_DIR / "phase3_cleaned_all.csv", index=False)
logger.info("Exports terminés : phase3_all_flags.csv, phase3_outliers_all.csv, phase3_cleaned_all.csv")

# ─── Impact statistique de l’exclusion des outliers ────────────────────────
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
impact = pd.DataFrame(index=numeric_cols)

# statistiques avant exclusion
impact['mean_complet'] = df[numeric_cols].mean()
impact['median_complet'] = df[numeric_cols].median()
impact['std_complet'] = df[numeric_cols].std()

# statistiques après exclusion (cleaned_all)
impact['mean_epure'] = cleaned_all[numeric_cols].mean()
impact['median_epure'] = cleaned_all[numeric_cols].median()
impact['std_epure'] = cleaned_all[numeric_cols].std()

impact.to_csv(OUTPUT_DIR / "phase3_cleaning_impact.csv")
logger.info("Export impact statistique de l'exclusion : phase3_cleaning_impact.csv")

# ─── 6) Génération des visualisations ───────────────────────
logger.info("Génération des visualisations")

# Préparation des listes de fichiers de figures
boxplot_files = []
hist_files = []
special_files = []
scatter_files = []
other_files = []  # Autres figures (venn, etc.)

# 1) Boxplots univariés (échelle log pour données asymétriques)
# variables ciblées
log_cols = [
    'Total recette actualisé',
    'Total recette réalisé',
    'Total recette produit',
    'Budget client estimé'
]

# a) BOXPLOTS SUR LES DONNÉES BRUTES
# ----------------------------------
# on filtre les valeurs positives et met NaN ailleurs
df_pos = df[log_cols].where(df[log_cols] > 0)

fig, ax = plt.subplots(figsize=(12, 6), dpi=200)
sns.boxplot(
    data=df_pos,
    whis=[5, 95],
    fliersize=3,
    flierprops=dict(marker='o', markeredgecolor='black',
                    markerfacecolor='lightgray', alpha=0.7, markersize=3),
    boxprops=dict(edgecolor='black'),
    medianprops=dict(color='black'),
    ax=ax
)
ax.set_yscale('log')
ax.set_xticklabels(log_cols, rotation=0, ha='right')
ax.set_title("Boxplots (log) des variables brutes", pad=12)
ax.grid(axis='y', linestyle='--', alpha=0.5)

plt.tight_layout()
file_raw = OUTPUT_DIR / "phase3_boxplots_all_log.png"
fig.savefig(file_raw, dpi=300, bbox_inches="tight")
plt.close(fig)
boxplot_files.append(file_raw)

# b) BOXPLOTS SUR LES DONNÉES NETTOYÉES
# ------------------------------------
if not cleaned_all.empty:
    df_clean_pos = cleaned_all[log_cols].where(cleaned_all[log_cols] > 0)
    # vérifier qu'il y a au moins 2 valeurs non-null par colonne
    if df_clean_pos.count().min() >= 2:
        fig, ax = plt.subplots(figsize=(12, 6), dpi=200)
        sns.boxplot(
            data=df_clean_pos,
            whis=[5, 95],
            fliersize=3,
            flierprops=dict(marker='o', markeredgecolor='black',
                            markerfacecolor='lightgray', alpha=0.7, markersize=3),
            boxprops=dict(edgecolor='black'),
            medianprops=dict(color='black'),
            ax=ax
        )
        ax.set_yscale('log')
        ax.set_xticklabels(log_cols, rotation=0, ha='right')
        ax.set_title("Boxplots (log) des variables nettoyées", pad=12)
        ax.grid(axis='y', linestyle='--', alpha=0.5)

        plt.tight_layout()
        file_clean = OUTPUT_DIR / "phase3_boxplots_all_log_cleaned.png"
        fig.savefig(file_clean, dpi=300, bbox_inches="tight")
        plt.close(fig)
        boxplot_files.append(file_clean)

# 2) Histogrammes univariés en échelle log
for col in num_cols:
    data = df[col].dropna()
    if data.empty:
        logger.warning(f"Skip histogramme {col} : pas de données")
        continue
    if col in ("Durée engagement (mois)", "Avant_vente"):
        logger.info(f"Skip histogramme {col} : variable non pertinente pour histogramme")
        continue

    # on ne trace que si on a des valeurs > 0 pour la log
    pos_data = data[data > 0]
    if pos_data.empty:
        logger.info(f"Skip hist log {col} : pas de valeurs > 0")
        continue

    # figure brute
    fig, ax = plt.subplots(figsize=(12, 6), dpi=200)
    sns.histplot(
        data,
        kde=False,
        ax=ax,
        edgecolor="black",
        linewidth=1,
        color=sns.color_palette('deep')[0]
    )
    ax.set_xscale('log')
    ax.set_title(f"Histogramme (échelle log) de {col}")
    ax.set_xlabel(col)
    plt.tight_layout()
    col_ascii = ''.join(c for c in unicodedata.normalize('NFD', col)
                        if unicodedata.category(c) != 'Mn')
    col_norm = re.sub(r'[()/%]', '', col_ascii.lower().replace(' ', '_'))
    col_norm = re.sub('_+', '_', col_norm).strip('_')
    file_path = OUTPUT_DIR / f"phase3_hist_log_{col_norm}.png"
    plt.savefig(file_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    hist_files.append(file_path)

    # idem sur le jeu nettoyé
    if not cleaned_all.empty and col in cleaned_all.columns:
        data_clean = cleaned_all[col].dropna()
        pos_clean = data_clean[data_clean > 0]
        if not pos_clean.empty:
            fig, ax = plt.subplots(figsize=(12, 6), dpi=200)
            sns.histplot(
                data_clean,
                kde=False,
                ax=ax,
                edgecolor="black",
                linewidth=1,
                color=sns.color_palette('deep')[0]
            )
            ax.set_xscale('log')
            ax.set_title(f"Histogramme (échelle log) de {col} (nettoyé)")
            ax.set_xlabel(col)
            plt.tight_layout()
            file_path_clean = OUTPUT_DIR / f"phase3_hist_log_{col_norm}_cleaned.png"
            plt.savefig(file_path_clean, dpi=300, bbox_inches="tight")
            plt.close(fig)
            hist_files.append(file_path_clean)

# ─── 2.2 Analyse des distributions – QQ‐plot, KDE cumulative & Hartigan dip ──

from scipy import stats

dist_vars = [
    ('Total recette réalisé', 'log'),
    ('Durée engagement (mois)', 'linear'),
]

for var, scale in dist_vars:
    if var not in df.columns:
        continue
    data = df[var].dropna()
    if scale == 'log':
        data = data[data > 0]
        data = np.log(data)
        label = f"log({var})"
    else:
        label = var

    safe = sanitize_filename(var)

    # 1) QQ-plot vs loi normale
    fig, ax = plt.subplots(figsize=(12, 6), dpi=200)
    stats.probplot(data, dist="norm", plot=ax)
    ax.set_title(f"QQ-plot de {label}")
    qq_name = f"phase3_qq_{safe}.png"
    qq_path = OUTPUT_DIR / qq_name
    fig.savefig(qq_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    hist_files.append(qq_path)
    logger.info(f"QQ-plot généré pour {label} → {qq_name}")

    # 2) KDE cumulative
    fig, ax = plt.subplots(figsize=(12, 6), dpi=200)
    sns.kdeplot(data, cumulative=True, ax=ax)
    ax.set_title(f"KDE cumulative de {label}")
    ax.set_xlabel(label)
    cum_name = f"phase3_kde_cum_{safe}.png"
    cum_path = OUTPUT_DIR / cum_name
    fig.savefig(cum_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    hist_files.append(cum_path)
    logger.info(f"KDE cumulative générée pour {label} → {cum_name}")

    # 3) Hartigan dip test
    try:
        from diptest import diptest

        dip_stat, p_val = diptest(data.values)
        logger.info(
            f"Hartigan dip test pour {label} : dip_stat={dip_stat:.4f}, p={p_val:.4f}"
        )
    except ImportError:
        logger.warning(
            "Package 'diptest' non installé : test de bimodalité ignoré"
        )

# 3) Graphiques spécifiques : Avant_vente et Durée engagement (mois)

# Distribution d'Avant_vente (0/1 manquant)
if 'Avant_vente' in df.columns:
    n_missing = df['Avant_vente'].isna().sum()
    if n_missing > 0:
        logger.warning(f"{n_missing} valeurs manquantes dans Avant_vente")
    # mappe 0.0→"0", 1.0→"1", NaN→"Missing"
    df['Avant_vente_plot'] = (
        df['Avant_vente']
        .map({0.0: '0', 1.0: '1'})
        .fillna('Missing')
        .astype('category')
    )
    fig, ax = plt.subplots(figsize=(12, 6), dpi=200)
    sns.countplot(x='Avant_vente_plot', data=df, order=['0', '1', 'Missing'], ax=ax)
    palette = {
        '0': '#4C72B0',  # choix depuis CRM (exemple)
        '1': '#55A868',
        'Missing': '#C44E52'
    }
    for patch in ax.patches:
        key = str(patch.get_x())  # or use order mapping
        patch.set_facecolor(palette.get(key, 'gray'))
    if len(ax.patches) > 2:
        ax.patches[2].set_facecolor('lightgray')
    if len(ax.patches) > 1:
        ax.patches[1].set_facecolor(sns.color_palette('deep')[0])
    for patch in ax.patches:
        patch.set_edgecolor('black')
    ax.set_title("Distribution d'Avant_vente (0 / 1 / Missing)")
    ax.set_xlabel("Avant_vente")
    ax.set_ylabel("Nombre d’enregistrements")
    plt.tight_layout()
    count_path = OUTPUT_DIR / "phase3_count_avant_vente.png"
    plt.savefig(count_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    special_files.append(count_path)
    logger.info("Countplot Avant_vente généré : phase3_count_avant_vente.png")

# ─── Stripplots Durée engagement – brut vs zoomé ───────────────────────────
if 'Durée engagement (mois)' in df.columns:
    data = df['Durée engagement (mois)'].dropna()
    if not data.empty:
        # 1) Stripplot full – échelle linéaire
        fig, ax = plt.subplots(figsize=(12, 6), dpi=200)
        sns.stripplot(x=data, ax=ax, jitter=0.2, size=3, alpha=0.6)
        ax.set_title("Stripplot Durée engagement – échelle linéaire (brute)")
        ax.set_xlabel("Durée (mois)")
        plt.tight_layout()
        full_lin = OUTPUT_DIR / "phase3_strip_duree_full_lin.png"
        fig.savefig(full_lin, dpi=300, bbox_inches="tight")
        plt.close(fig)
        special_files.append(full_lin)
        logger.info("Stripplot linéaire complet généré : phase3_strip_duree_full_lin.png")

        # 2) Stripplot full – échelle log
        fig, ax = plt.subplots(figsize=(12, 6), dpi=200)
        sns.stripplot(x=data, ax=ax, jitter=0.2, size=3, alpha=0.6)
        ax.set_xscale('log')
        ax.set_title("Stripplot Durée engagement – échelle logarithmique (brute)")
        ax.set_xlabel("Durée (mois)")
        plt.tight_layout()
        full_log = OUTPUT_DIR / "phase3_strip_duree_full_log.png"
        fig.savefig(full_log, dpi=300, bbox_inches="tight")
        plt.close(fig)
        special_files.append(full_log)
        logger.info("Stripplot log complet généré : phase3_strip_duree_full_log.png")

        # 3) Stripplot zoom – P0–P95
        p95 = data.quantile(0.95)
        fig, ax = plt.subplots(figsize=(12, 6), dpi=200)
        sns.stripplot(x=data, ax=ax, jitter=0.2, size=3, alpha=0.6)
        ax.set_xlim(0, p95)
        ax.set_title("Stripplot Durée engagement – zoom 0–95ᵉ percentile")
        ax.set_xlabel("Durée (mois)")
        plt.tight_layout()
        zoom = OUTPUT_DIR / "phase3_strip_duree_zoom.png"
        fig.savefig(zoom, dpi=300, bbox_inches="tight")
        plt.close(fig)
        special_files.append(zoom)
        logger.info("Stripplot zoom P0–95% généré : phase3_strip_duree_zoom.png")

logger.info("Durée d’engagement : missing à 97 % (variable supprimée)")

# 4) Scatterplots (paires de variables numériques avec peu de missings)
vars_scatter = [
    'Total recette actualisé',
    'Total recette réalisé',
    'Total recette produit',
    'Budget client estimé'
]
# toutes les paires possibles
pairs = list(combinations(vars_scatter, 2))

# palette de base
palette = sns.color_palette('deep', 4)
color_normal = palette[0]
color_xaxis = palette[1]
color_yaxis = palette[2]
color_diagonal = palette[3]
color_origin = 'black'


def norm(col):
    ascii_col = ''.join(c for c in unicodedata.normalize('NFD', col)
                        if unicodedata.category(c) != 'Mn')
    s = re.sub(r'[()/%]', '', ascii_col.lower().replace(' ', '_'))
    return re.sub('_+', '_', s).strip('_')


for x_col, y_col in pairs:
    X = df[x_col]
    Y = df[y_col]

    # masques
    mask_origin = (X == 0) & (Y == 0)
    mask_xaxis = (Y == 0) & ~mask_origin
    mask_yaxis = (X == 0) & ~mask_origin
    mask_diag = (X == Y) & ~mask_origin
    mask_normal = ~(mask_origin | mask_xaxis | mask_yaxis | mask_diag)

    x_norm = norm(x_col)
    y_norm = norm(y_col)

    fig, ax = plt.subplots(figsize=(12, 6), dpi=200)

    # tracé de chaque catégorie
    ax.scatter(X[mask_normal], Y[mask_normal],
               color=color_normal, alpha=0.6)
    ax.scatter(X[mask_xaxis], Y[mask_xaxis],
               color=color_xaxis, alpha=0.8, label=f"{y_col} = 0")
    ax.scatter(X[mask_yaxis], Y[mask_yaxis],
               color=color_yaxis, alpha=0.8, label=f"{x_col} = 0")
    ax.scatter(X[mask_diag], Y[mask_diag],
               color=color_diagonal, alpha=0.8, label=f"{x_col} = {y_col}")
    if mask_origin.any():
        # un seul point (0,0)
        ax.scatter(0, 0, color=color_origin, label="Valeurs nulles")

    ax.set_title(f"Scatter {x_col} vs {y_col}")
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(loc='best', framealpha=0.7)
    plt.tight_layout()
    scatter_path = OUTPUT_DIR / f"phase3_scatter_{x_norm}_{y_norm}.png"
    plt.savefig(scatter_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    scatter_files.append(scatter_path)

    # --- log-log (avec coloration complète) ---
    # on reprend les mêmes masques que pour la version linéaire
    mask_origin = (X == 0) & (Y == 0)
    mask_xaxis = (Y == 0) & ~mask_origin
    mask_yaxis = (X == 0) & ~mask_origin
    mask_diag = (X == Y) & ~mask_origin
    mask_norm = ~(mask_origin | mask_xaxis | mask_yaxis | mask_diag)

    # on ne trace en log-log que les points strictement positifs
    mask_origin_plot = mask_origin  # jamais tracé en log, reste faux
    mask_xaxis_plot = mask_xaxis  # toujours vide car Y>0 requis
    mask_yaxis_plot = mask_yaxis  # idem
    mask_diag_plot = mask_diag & (X > 0) & (Y > 0)
    mask_norm_plot = mask_norm & (X > 0) & (Y > 0)

    if mask_norm_plot.sum() + mask_diag_plot.sum() > 0:
        fig, ax = plt.subplots(figsize=(12, 6), dpi=200)

        # Points « normaux »
        if mask_norm_plot.any():
            ax.scatter(
                X[mask_norm_plot], Y[mask_norm_plot],
                color=color_normal, alpha=0.6,
            )
        # Points diagonale (X==Y>0)
        if mask_diag_plot.any():
            ax.scatter(
                X[mask_diag_plot], Y[mask_diag_plot],
                color=color_diagonal, alpha=0.8, label=f"{x_col} = {y_col}"
            )
        # (les masques axes/origine sont vides en log-log strictement positif)

        # Échelles log-log
        ax.set_xscale('log')
        ax.set_yscale('log')

        # Titres et légende
        ax.set_title(f"Scatter {x_col} vs {y_col} (log-log)")
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(loc='best', framealpha=0.7)

        plt.tight_layout()
        scatter_path_log = OUTPUT_DIR / f"phase3_scatter_{x_norm}_{y_norm}_loglog.png"
        plt.savefig(scatter_path_log, dpi=300, bbox_inches="tight")
        plt.close(fig)
        scatter_files.append(scatter_path_log)

# ─── Scatter log-log « nettoyé » (dataset_phase3_cleaned.csv) ────────────────
# même masques et couleurs que pour la version brute, appliqués au jeu cleaned_all
if 'cleaned_all' in globals():
    Xc = cleaned_all['Total recette actualisé'].fillna(0).values
    Yc = cleaned_all['Budget client estimé'].fillna(0).values

    # Reconstruire les masques
    mask_origin_c = (Xc == 0) & (Yc == 0)
    mask_xaxis_c = (Yc == 0) & ~mask_origin_c
    mask_yaxis_c = (Xc == 0) & ~mask_origin_c
    mask_diag_c = (Xc == Yc) & ~mask_origin_c
    mask_norm_c = ~(mask_origin_c | mask_xaxis_c | mask_yaxis_c | mask_diag_c)

    mask_diag_plot_c = mask_diag_c & (Xc > 0) & (Yc > 0)
    mask_norm_plot_c = mask_norm_c & (Xc > 0) & (Yc > 0)

    if mask_norm_plot_c.sum() + mask_diag_plot_c.sum() > 0:
        fig, ax = plt.subplots(figsize=(12, 6), dpi=200)

        # Points « normaux »
        if mask_norm_plot_c.any():
            ax.scatter(
                Xc[mask_norm_plot_c], Yc[mask_norm_plot_c],
                color=color_normal, alpha=0.6, s=30
            )
        # Points diagonale (Xc==Yc>0)
        if mask_diag_plot_c.any():
            ax.scatter(
                Xc[mask_diag_plot_c], Yc[mask_diag_plot_c],
                color=color_diagonal, alpha=0.8,
                label="Total recette actualisé = Budget client estimé"
            )

        # Échelles log-log
        ax.set_xscale('log')
        ax.set_yscale('log')

        # Titres et légende
        ax.set_title("Scatter Total recette actualisé vs Budget client estimé (nettoyé, log-log)")
        ax.set_xlabel("Total recette actualisé")
        ax.set_ylabel("Budget client estimé")
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(loc='best', framealpha=0.7)

        plt.tight_layout()
        cleaned_scatter_path = OUTPUT_DIR / "phase3_scatter_total_recette_actualise_budget_client_estime_loglog_cleaned.png"
        plt.savefig(cleaned_scatter_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        scatter_files.append(cleaned_scatter_path)

# 5) Heatmaps côte-à-côte des corrélations (brut vs nettoyé)
valid_corr_cols = [
    'Total recette actualisé',
    'Total recette réalisé',
    'Total recette produit',
    'Budget client estimé'
]
if valid_corr_cols:
    # calculs
    corr_raw = df[valid_corr_cols].corr()
    corr_clean = cleaned_all[valid_corr_cols].corr() if not cleaned_all.empty else None

    # figure unique
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), dpi=200)

    # heatmap brute
    sns.heatmap(
        corr_raw,
        annot=True, fmt=".2f",
        cmap="coolwarm", vmin=0, vmax=1,
        square=True, linewidths=0.3,
        cbar_kws={"shrink": 0.6},
        ax=axes[0]
    )
    axes[0].set_xticklabels(valid_corr_cols, rotation=45, ha="right")
    axes[0].set_yticklabels(valid_corr_cols, rotation=0, va="center")
    axes[0].set_title("Corrélations – jeu brut", pad=8)

    # heatmap nettoyée (si disponible)
    if corr_clean is not None:
        sns.heatmap(
            corr_clean,
            annot=True, fmt=".2f",
            cmap="coolwarm", vmin=0, vmax=1,
            square=True, linewidths=0.3,
            cbar_kws={"shrink": 0.6},
            ax=axes[1]
        )
        axes[1].set_xticklabels(valid_corr_cols, rotation=45, ha="right")
        axes[1].set_yticklabels(valid_corr_cols, rotation=0, va="center")
        axes[1].set_title("Corrélations – jeu nettoyé", pad=8)
    else:
        axes[1].axis("off")

    plt.tight_layout(pad=2)
    heatmap_path = OUTPUT_DIR / "phase3_heatmap_corr_comparison.png"
    fig.savefig(heatmap_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    logger.info("Heatmaps côte-à-côte générées : phase3_heatmap_corr_comparison.png")
    other_files.append(heatmap_path)

# 6) Visualisations spécifiques sur les outliers (synthèses)
# Barplot du nombre d'outliers univariés par variable
uc_df = pd.DataFrame(univ_counts_list)
uc_df = uc_df[~uc_df['variable'].isin(['Avant_vente', 'Durée engagement (mois)'])]
uc_df = uc_df[uc_df['count'] > 0].sort_values('count', ascending=False)
if not uc_df.empty:
    fig, ax = plt.subplots(figsize=(12, 5), dpi=200)  #(10, max(4, 0.4 * len(uc_df)))
    sns.barplot(y='variable', x='count', data=uc_df, ax=ax, color=sns.color_palette('deep')[0])
    for patch in ax.patches:
        patch.set_edgecolor('black')
    ax.set_title("Outliers univariés par variable")
    ax.set_xlabel("Nombre d'outliers")
    ax.set_ylabel("Variable")
    plt.tight_layout()
    outlier_count_path = OUTPUT_DIR / "phase3_outlier_counts_univariate.png"
    plt.savefig(outlier_count_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    other_files.append(outlier_count_path)
    logger.info("Graphique outliers univariés par variable : phase3_outlier_counts_univariate.png")

# Diagramme de Venn du chevauchement entre IF et LOF
try:
    from matplotlib_venn import venn2
except ImportError:
    logger.warning("matplotlib_venn non disponible, diagramme Venn ignoré")

# calcul du chevauchement
only_if = out_if.shape[0] - common_if_lof
only_lof = out_lof.shape[0] - common_if_lof
both = common_if_lof

venn_colors = {
    '10': 'blue',  # IF seul
    '01': 'orange',  # LOF seul
    '11': 'purple',  # IF ∩ LOF
}

fig, ax = plt.subplots(figsize=(12, 4), dpi=200)
v = venn2(
    subsets=(only_if, only_lof, both),
    set_labels=('IsolationForest', 'LOF'),
    alpha=0.6,
    ax=ax
)
ax.set_title("Overlap des outliers IF vs LOF")

# applique simplement les couleurs avec transparence
for region, color in venn_colors.items():
    patch = v.get_patch_by_id(region)
    if patch:
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

# taille des labels
for text in v.set_labels + v.subset_labels:
    if text:
        text.set_fontsize(12)

plt.tight_layout()
venn_path = OUTPUT_DIR / "phase3_venn_if_lof.png"
plt.savefig(venn_path, dpi=300, bbox_inches="tight")
plt.close(fig)
other_files.append(venn_path)
logger.info("Diagramme Venn IF vs LOF généré : phase3_venn_if_lof.png")

logger.info("Visualisations générées et sauvegardées")

# ─── Crosstab Statut × flag_multivariate + répartition des statuts des outliers ──
# Crosstab et export CSV
status_ct = pd.crosstab(df['Statut production'], df['flag_multivariate'])
status_ct.to_csv(OUTPUT_DIR / "phase3_status_flag_multivariate.csv")
logger.info("Export crosstab Statut production × flag_multivariate : phase3_status_flag_multivariate.csv")

# Barplot des statuts au sein des outliers multivariés
multivar = df[df['flag_multivariate']]
status_pct = multivar['Statut production'].value_counts(normalize=True).mul(100).round(1)

fig, ax = plt.subplots(figsize=(8, 5), dpi=200)
sns.barplot(x=status_pct.index, y=status_pct.values, ax=ax, palette='deep')
ax.set_title("Répartition des statuts des outliers multivariés")
ax.set_xlabel("Statut production")
ax.set_ylabel("Pourcentage (%)")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

stat_bar_path = OUTPUT_DIR / "phase3_status_multivar_outliers.png"
fig.savefig(stat_bar_path, dpi=300, bbox_inches="tight")
plt.close(fig)
other_files.append(stat_bar_path)
logger.info("Barplot des statuts des outliers multivariés généré : phase3_status_multivar_outliers.png")

# Commentaire métier
pct_perdus = status_pct.get('Perdu', 0)
logger.info(f"{pct_perdus:.1f}% des outliers multivariés sont des affaires perdues")

# Ajouts des éléments concernant les outliers à other_files
consolider_outliers_global(
    df_all=df,
    outliers_cols=[
        'Total recette actualisé',
        'Total recette réalisé',
        'Total recette produit',
        'Budget client estimé'
    ],
    output_dir=OUTPUT_DIR,
    fig_list_container=other_files
)

# --- Ajout des signaux faibles dans les variables catégorielles ---
detecter_categories_rares(
    df_all=df,
    seuil_modalite=5,
    seuil_combo=3,
    couples_logiques=(('Type', 'Pilier'),
                      ('Statut commercial', 'Statut production')),
    out_dir=OUTPUT_DIR,
    fig_container=other_files  # sera ajouté plus loin au PDF global
)


# ─── Profil métiers des outliers – Top clients & piliers ──────────────────
def profil_outliers_categoriels(df_all, outliers_all, output_dir, fig_container):
    """
    1) Exporte top 5 clients avec le plus d'outliers
    2) Exporte top 5 piliers avec le plus d'outliers
    3) Barplot empilé (normal vs outliers) par pilier
    """
    # --- 1) Top 5 clients ---
    top_clients = (
        outliers_all['Client']
        .value_counts()
        .head(5)
        .reset_index()
        .rename(columns={'index': 'Client', 'Client': 'outlier_count'})
    )
    top_clients.to_csv(output_dir / "phase3_top5_clients_outliers.csv", index=False)
    logger.info("Export Top 5 clients outliers : phase3_top5_clients_outliers.csv")

    # --- 2) Top 5 piliers ---
    top_piliers = (
        outliers_all['Pilier']
        .value_counts(dropna=False)
        .head(5)
        .reset_index()
        .rename(columns={'index': 'Pilier', 'Pilier': 'outlier_count'})
    )
    top_piliers.to_csv(output_dir / "phase3_top5_piliers_outliers.csv", index=False)
    logger.info("Export Top 5 piliers outliers : phase3_top5_piliers_outliers.csv")

    # --- 3) Barplot empilé outliers vs total par pilier ---
    # Totaux par pilier
    total_pilier = df_all['Pilier'].value_counts(dropna=False)
    outlier_pilier = outliers_all['Pilier'].value_counts(dropna=False)
    df_bar = pd.DataFrame({
        'Pilier': total_pilier.index,
        'total': total_pilier.values,
        'outliers': outlier_pilier.reindex(total_pilier.index, fill_value=0).values
    })
    df_bar['normal'] = df_bar['total'] - df_bar['outliers']
    # passage en format long
    bar_df = df_bar.melt(
        id_vars='Pilier',
        value_vars=['normal', 'outliers'],
        var_name='type',
        value_name='count'
    )
    fig, ax = plt.subplots(figsize=(10, 6), dpi=200)
    sns.barplot(
        data=bar_df,
        x='Pilier', y='count',
        hue='type',
        ax=ax
    )
    ax.set_title("Outliers vs Normaux par pilier")
    ax.set_xlabel("Pilier")
    ax.set_ylabel("Nombre d’enregistrements")
    ax.legend(title="", loc='best', framealpha=0.7)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    bar_path = output_dir / "phase3_bar_outliers_par_pilier.png"
    fig.savefig(bar_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    fig_container.append(bar_path)
    logger.info("Barplot outliers par pilier généré : phase3_bar_outliers_par_pilier.png")


# --- Appel du profil métier ---
profil_outliers_categoriels(
    df_all=df,
    outliers_all=outliers_all,
    output_dir=OUTPUT_DIR,
    fig_container=other_files
)


# 7) PDF des figures et liste des fichiers
# ─── Génération ou mise à jour du PDF récapitulatif des figures ───────────────
def generer_figures_synthese_phase3(out_dir,
                                    boxplot_files,
                                    hist_files,
                                    special_files,
                                    scatter_files,
                                    other_files,
                                    pca_files):
    """
    Concatène toutes les figures PNG pertinentes (y c. clusters d'outliers
    et barplot des modalités rares) dans un unique PDF phase3_figures.pdf.
    """
    pdf_path = out_dir / "phase3_figures.pdf"
    # Concatène dans l'ordre souhaité (boxplots → histos → spéciaux → scatter → autres → PCA)
    final_list = (
            boxplot_files +
            hist_files +
            special_files +
            scatter_files +
            other_files +  # inclut barplot rare + barplot familles d'outliers + venn
            pca_files  # inclut la figure des clusters d'outliers
    )

    # ─── Génération de la page INDEX des figures ─────────────────────────
    # On liste les fichiers et on crée un PNG A4 avec les légendes
    import matplotlib.pyplot as _plt

    # Préparer les lignes d'index
    index_lines = []
    for i, png in enumerate(final_list, start=1):
        index_lines.append(f"Figure {i:02d} – {png.name}")

    # Créer la figure d'index (A4 portrait)
    fig, ax = _plt.subplots(figsize=(8.27, 11.69), dpi=200)
    ax.axis('off')
    ax.text(
        0.01, 0.99,
        "\n".join(index_lines),
        va='top', family='monospace', fontsize=8
    )
    idx_png = out_dir / "phase3_figures_index.png"
    fig.savefig(idx_png, dpi=300, bbox_inches="tight")
    _plt.close(fig)

    # On insère l'index en début de liste pour qu'il soit la première page du PDF
    # elle sera ouverte plus bas
    final_list.insert(0, idx_png)

    if not final_list:
        logger.warning("Aucune figure PNG trouvée pour générer le PDF.")
        return

    images = []
    for png in final_list:
        try:
            img = Image.open(png)
            if img.mode in ("RGBA", "P"):
                img = img.convert("RGB")
            images.append(img)
        except Exception as e:
            logger.error(f"Impossible d’ouvrir {png} ({e}) – figure ignorée")

    # Sauvegarde PDF (append_images ne fonctionne que si au moins 2 images)
    if images:
        try:
            images[0].save(
                pdf_path,
                format="PDF",
                save_all=True,
                append_images=images[1:],
                resolution=300
            )
            logger.info(f"PDF des figures de phase 3 mis à jour : {pdf_path.name}")
        except PermissionError:
            logger.warning(
                f"Permission refusée à l’écriture de {pdf_path.name}, suppression de l’ancien fichier et nouvelle tentative")
            try:
                pdf_path.unlink()
                images[0].save(
                    pdf_path,
                    format="PDF",
                    save_all=True,
                    append_images=images[1:],
                    resolution=300
                )
                logger.info(f"PDF régénéré après suppression de l’ancien : {pdf_path.name}")
            except Exception as e:
                logger.error(f"Échec de la régénération du PDF après suppression ({e})")
    else:
        logger.warning("Aucune image valide pour générer le PDF.")


# ─── Appel immédiat ───
generer_figures_synthese_phase3(
    out_dir=OUTPUT_DIR,
    boxplot_files=boxplot_files,
    hist_files=hist_files,
    special_files=special_files,
    scatter_files=scatter_files,
    other_files=other_files,
    pca_files=pca_files
)

# Liste de tous les fichiers produits
listing_path = OUTPUT_DIR / "phase3_output_files.txt"
with open(listing_path, "w", encoding="utf-8") as list_file:
    for f in sorted(OUTPUT_DIR.iterdir()):
        list_file.write(f.name + "\n")
logger.info(f"Listing des fichiers produit : {listing_path.name}")


# 2025-05-16 12:32:46,665 - INFO - Dossier d'export Phase 3 prêt : D:\DATAPREDICT\DATAPREDICT 2024\Missions\Digora\phase3_output
# 2025-05-16 12:33:04,967 - INFO - Données chargées : 16100 lignes × 75 colonnes
# 2025-05-16 12:33:04,968 - INFO - Démarrage de l’analyse amont détaillée des données
# 2025-05-16 12:33:04,968 - INFO - Colonnes du DataFrame : ['Client', 'Code', 'Titre', 'Contact principal', 'Entité opérationnelle', 'Chef de projet', 'Statut production', 'Avant_vente', 'Banque', 'Commentaire Admin', "Centrale d'achats / Partenaire", 'Commercial sédentaire', 'Total recette actualisé', 'Total recette réalisé', 'Total recette produit', 'Unnamed: 15', 'Unnamed: 16', 'Unnamed: 17', 'Unnamed: 18', 'Affaire principale', 'Directeur de projet', 'Commercial', 'Opération', 'Statut commercial', 'Type', 'Catégorie', 'Sous-catégorie', 'Pilier', 'Charge prévisionnelle projet', 'Unité de durée', 'Date de début initiale', 'Date prévisionnelle début projet', 'Date de fin initiale', 'Date de début actualisée', 'Date de fin actualisée', "Devise de gestion de l'affaire", "Date d'enregistrement", 'Date de modification', 'Dernière modification par', 'Périodicité de Facturation', 'Reconduction tacite', 'Date de reconduction', 'Date de fin réelle', 'Budget client estimé', "Mois de référence de l'indice SYNTEC", 'Motif non conformité', 'Date de fin de Delivery', 'Date de résiliation', 'Durée engagement (mois)', "Reconduite par l'affaire", "En remplacement de l'affaire", 'Autre Partenaire', 'Commentaire renouvellement ', "Est une extension de l'affaire", 'Date annonce Résiliation', 'RC traité', 'Raison de cloture', 'SharePoint', 'Type de document', 'Version du document', 'Version du contrat en cours', 'Description', 'Suivi par le service', 'idSALES', 'Hébergement', 'Code Analytique', 'Type opportunité', 'Date de signature', 'Adresse société', 'Code postal société', 'Ville société', 'Téléphone société', 'SIREN société', 'SIRET société', 'E-mail société']
# 2025-05-16 12:33:04,969 - INFO - Types et non-null counts :
# 2025-05-16 12:33:05,020 - INFO -
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 16100 entries, 0 to 16099
# Data columns (total 75 columns):
#  #   Column                                Non-Null Count  Dtype
# ---  ------                                --------------  -----
#  0   Client                                15968 non-null  object
#  1   Code                                  16100 non-null  object
#  2   Titre                                 16100 non-null  object
#  3   Contact principal                     13420 non-null  object
#  4   Entité opérationnelle                 16100 non-null  object
#  5   Chef de projet                        7316 non-null   object
#  6   Statut production                     16100 non-null  object
#  7   Avant_vente                           15312 non-null  float64
#  8   Banque                                2160 non-null   object
#  9   Commentaire Admin                     3429 non-null   object
#  10  Centrale d'achats / Partenaire        8241 non-null   object
#  11  Commercial sédentaire                 3988 non-null   object
#  12  Total recette actualisé               16100 non-null  float64
#  13  Total recette réalisé                 16100 non-null  float64
#  14  Total recette produit                 16100 non-null  float64
#  15  Unnamed: 15                           0 non-null      float64
#  16  Unnamed: 16                           8857 non-null   object
#  17  Unnamed: 17                           41 non-null     object
#  18  Unnamed: 18                           0 non-null      float64
#  19  Affaire principale                    11 non-null     object
#  20  Directeur de projet                   4492 non-null   object
#  21  Commercial                            15968 non-null  object
#  22  Opération                             2654 non-null   object
#  23  Statut commercial                     16100 non-null  object
#  24  Type                                  16100 non-null  object
#  25  Catégorie                             16100 non-null  object
#  26  Sous-catégorie                        16091 non-null  object
#  27  Pilier                                2486 non-null   object
#  28  Charge prévisionnelle projet          3942 non-null   float64
#  29  Unité de durée                        16100 non-null  object
#  30  Date de début initiale                3383 non-null   datetime64[ns]
#  31  Date prévisionnelle début projet      229 non-null    datetime64[ns]
#  32  Date de fin initiale                  3370 non-null   object
#  33  Date de début actualisée              7454 non-null   datetime64[ns]
#  34  Date de fin actualisée                7858 non-null   object
#  35  Devise de gestion de l'affaire        16100 non-null  object
#  36  Date d'enregistrement                 16099 non-null  datetime64[ns]
#  37  Date de modification                  16100 non-null  datetime64[ns]
#  38  Dernière modification par             16100 non-null  object
#  39  Périodicité de Facturation            1931 non-null   object
#  40  Reconduction tacite                   13896 non-null  object
#  41  Date de reconduction                  3248 non-null   datetime64[ns]
#  42  Date de fin réelle                    7304 non-null   datetime64[ns]
#  43  Budget client estimé                  15640 non-null  float64
#  44  Mois de référence de l'indice SYNTEC  3 non-null      datetime64[ns]
#  45  Motif non conformité                  415 non-null    object
#  46  Date de fin de Delivery               2963 non-null   datetime64[ns]
#  47  Date de résiliation                   128 non-null    datetime64[ns]
#  48  Durée engagement (mois)               476 non-null    float64
#  49  Reconduite par l'affaire              2216 non-null   object
#  50  En remplacement de l'affaire          2100 non-null   object
#  51  Autre Partenaire                      29 non-null     object
#  52  Commentaire renouvellement            0 non-null      float64
#  53  Est une extension de l'affaire        249 non-null    object
#  54  Date annonce Résiliation              83 non-null     datetime64[ns]
#  55  RC traité                             9669 non-null   object
#  56  Raison de cloture                     16 non-null     object
#  57  SharePoint                            8795 non-null   object
#  58  Type de document                      4 non-null      object
#  59  Version du document                   1 non-null      object
#  60  Version du contrat en cours           75 non-null     object
#  61  Description                           1493 non-null   object
#  62  Suivi par le service                  1839 non-null   object
#  63  idSALES                               2088 non-null   object
#  64  Hébergement                           7981 non-null   object
#  65  Code Analytique                       3725 non-null   object
#  66  Type opportunité                      13076 non-null  object
#  67  Date de signature                     15996 non-null  datetime64[ns]
#  68  Adresse société                       15793 non-null  object
#  69  Code postal société                   15766 non-null  object
#  70  Ville société                         15874 non-null  object
#  71  Téléphone société                     13391 non-null  object
#  72  SIREN société                         13362 non-null  object
#  73  SIRET société                         15902 non-null  object
#  74  E-mail société                        1892 non-null   object
# dtypes: datetime64[ns](12), float64(10), object(53)
# memory usage: 9.2+ MB
#
# 2025-05-16 12:33:05,062 - INFO - Résumé numérique enrichi :
#                                 count          mean            std      min     25%       50%       75%         max  missing  missing_pct  nunique   skew  kurtosis
# Avant_vente                   15312.0      0.006792       0.082136     0.00     0.0     0.000      0.00         1.0      788         4.89        2  12.01    142.28
# Total recette actualisé       16100.0  15939.273109  106754.273092 -1636.28     0.0  1180.000   7750.00   9283629.8        0         0.00     5243  48.33   3694.47
# Total recette réalisé         16100.0  15443.026883  105907.332715 -1636.28     0.0  1082.300   7392.25   9283629.8        0         0.00     5079  49.40   3814.07
# Total recette produit         16100.0  14329.807758  101120.427227  -777.00     0.0   416.055   6642.00   9283629.8        0         0.00     5296  54.39   4524.10
# Unnamed: 15                       0.0           NaN            NaN      NaN     NaN       NaN       NaN         NaN    16100       100.00        0    NaN       NaN
# Unnamed: 18                       0.0           NaN            NaN      NaN     NaN       NaN       NaN         NaN    16100       100.00        0    NaN       NaN
# Charge prévisionnelle projet   3942.0      0.000000       0.000000     0.00     0.0     0.000      0.00         0.0    12158        75.52        1   0.00      0.00
# Budget client estimé          15640.0  22373.085843  146921.370656 -5984.00  2934.5  5753.000  15000.00  15012015.0      460         2.86     4385  72.06   7001.04
# Durée engagement (mois)         476.0     12.476891       3.970084     3.00    12.0    12.000     12.00        48.0    15624        97.04        9   7.11     53.89
# Commentaire renouvellement        0.0           NaN            NaN      NaN     NaN       NaN       NaN         NaN    16100       100.00        0    NaN       NaN
# 2025-05-16 12:33:05,187 - INFO - Résumé catégoriel :
#                       variable  dtype  nunique                                                         top  freq_top  missing  missing_pct
#                         Client object     2415                                          LACOSTE OPERATIONS       171      132         0.82
#                           Code object    16100                                                   BOR107665         1        0         0.00
#                          Titre object    13642                    ISR_RENOUVELLEMENT SUPPORT TECHNOLOGIQUE       107        0         0.00
#              Contact principal object     4018                                                         NaN      2680     2680        16.65
#          Entité opérationnelle object       13                                                    RENNES 1      3124        0         0.00
#                 Chef de projet object       97                                                         NaN      8784     8784        54.56
#              Statut production object        9                                                       Perdu      5038        0         0.00
#                         Banque object        6                                                         NaN     13940    13940        86.58
#              Commentaire Admin object     3054                                                         NaN     12671    12671        78.70
# Centrale d'achats / Partenaire object       14                                                         NaN      7859     7859        48.81
#          Commercial sédentaire object       12                                                         NaN     12112    12112        75.23
#                    Unnamed: 16 object        4 Il y a un(des) devis validé(s) ayant un bon de commande lié      8738     7243        44.99
#                    Unnamed: 17 object        4                                                         NaN     16059    16059        99.75
#             Affaire principale object       11                                                         NaN     16089    16089        99.93
#            Directeur de projet object       25                                                         NaN     11608    11608        72.10
#                     Commercial object       66                                                         AUG      1562      132         0.82
#                      Opération object       15                                                         NaN     13446    13446        83.52
#              Statut commercial object        6                                                       Gagné      9348        0         0.00
#                           Type object        4                                                     Externe     15913        0         0.00
#                      Catégorie object       20                                                      PROJET      8013        0         0.00
#                 Sous-catégorie object       50                                                       Régie      6022        9         0.06
#                         Pilier object        5                                                         NaN     13614    13614        84.56
#                 Unité de durée object        2                                                        Jour     16097        0         0.00
#           Date de fin initiale object      898                                                         NaN     12730    12730        79.07
#         Date de fin actualisée object     2416                                                         NaN      8242     8242        51.19
# Devise de gestion de l'affaire object        4                                                         EUR     16093        0         0.00
#      Dernière modification par object      116                                  pierre.sallerin@digora.com      9193        0         0.00
#     Périodicité de Facturation object        5                                                         NaN     14169    14169        88.01
#            Reconduction tacite object        3                                                         Non     11501     2204        13.69
#           Motif non conformité object        8                                                         NaN     15685    15685        97.42
#       Reconduite par l'affaire object     2173                                                         NaN     13884    13884        86.24
#   En remplacement de l'affaire object     2031                                                         NaN     14000    14000        86.96
#               Autre Partenaire object       22                                                         NaN     16071    16071        99.82
# Est une extension de l'affaire object      207                                                         NaN     15851    15851        98.45
#                      RC traité object        3                                                         Non      9237     6431        39.94
#              Raison de cloture object        6                                                         NaN     16084    16084        99.90
#                     SharePoint object     8795                                                         NaN      7305     7305        45.37
#               Type de document object        2                                                         NaN     16096    16096        99.98
#            Version du document object        2                                                         NaN     16099    16099        99.99
#    Version du contrat en cours object        6                                                         NaN     16025    16025        99.53
#                    Description object      641                                                         NaN     14607    14607        90.73
#           Suivi par le service object        7                                                         NaN     14261    14261        88.58
#                        idSALES object     2089                                                         NaN     14012    14012        87.03
#                    Hébergement object       10                                                         NaN      8119     8119        50.43
#                Code Analytique object       11                                                         NaN     12375    12375        76.86
#               Type opportunité object        6                                                        PIPE      7339     3024        18.78
#                Adresse société object     2290                                                         NaN       307      307         1.91
#            Code postal société object     1589                                                         NaN       334      334         2.07
#                  Ville société object     1467                                                       PARIS      1046      226         1.40
#              Téléphone société object     1743                                                         NaN      2709     2709        16.83
#                  SIREN société object     1590                                                         NaN      2738     2738        17.01
#                  SIRET société object     2380                                                         NaN       198      198         1.23
#                 E-mail société object      237                                                         NaN     14208    14208        88.25
# 2025-05-16 12:33:05,273 - INFO - Aperçu des dates :
#                         variable                 min                 max  missing
#           Date de début initiale 1899-12-29 00:00:00 2050-02-01 00:00:00    12717
# Date prévisionnelle début projet 2023-01-01 00:00:00 2025-01-01 00:00:00    15871
#             Date de fin initiale 1970-01-01 00:00:00 2054-06-04 00:00:00    12731
#         Date de début actualisée 1899-12-29 00:00:00 2025-07-27 00:00:00     8646
#           Date de fin actualisée 2010-02-20 00:00:00 2052-04-01 00:00:00     8243
#            Date d'enregistrement 2009-09-15 00:00:00 2025-04-29 17:03:00        1
#             Date de modification 2002-02-08 14:30:00 2025-04-30 10:26:00        0
#             Date de reconduction 2016-12-31 00:00:00 2107-09-12 00:00:00    12852
#               Date de fin réelle 2009-12-16 00:00:00 2025-04-30 00:00:00     8796
#          Date de fin de Delivery 2016-02-25 00:00:00 2025-04-30 00:00:00    13137
#              Date de résiliation 2018-03-31 00:00:00 2025-10-08 00:00:00    15972
#         Date annonce Résiliation 2019-12-02 00:00:00 2025-04-04 00:00:00    16017
#                Date de signature 1899-12-29 00:00:00 2051-12-24 00:00:00      104
# 2025-05-16 12:33:05,275 - INFO - Analyse amont terminée – fichiers CSV générés dans phase3_output
# 2025-05-16 12:33:05,345 - INFO - Export anomalies dates (hors 1990–2050) : phase3_date_anomalies.csv (71 enregistrements)
# 2025-05-16 12:33:05,345 - INFO - Démarrage de l’étape A : statistiques descriptives et typologie
# 2025-05-16 12:33:05,346 - INFO - 10 variables numériques détectées
# 2025-05-16 12:33:05,349 - WARNING - Avant_vente → faible variabilité (2 valeurs uniques)
# 2025-05-16 12:33:05,354 - INFO - Total recette actualisé → forte asymétrie (skewness=48.33)
# 2025-05-16 12:33:05,359 - INFO - Total recette réalisé → forte asymétrie (skewness=49.39)
# 2025-05-16 12:33:05,365 - INFO - Total recette produit → forte asymétrie (skewness=54.39)
# 2025-05-16 12:33:05,366 - WARNING - Unnamed: 15 → faible variabilité (0 valeurs uniques)
# 2025-05-16 12:33:05,367 - WARNING - Unnamed: 18 → faible variabilité (0 valeurs uniques)
# 2025-05-16 12:33:05,370 - WARNING - Charge prévisionnelle projet → faible variabilité (1 valeurs uniques)
# 2025-05-16 12:33:05,376 - INFO - Budget client estimé → forte asymétrie (skewness=72.05)
# 2025-05-16 12:33:05,379 - INFO - Durée engagement (mois) → forte asymétrie (skewness=7.09)
# 2025-05-16 12:33:05,380 - WARNING - Commentaire renouvellement  → faible variabilité (0 valeurs uniques)
# 2025-05-16 12:33:05,381 - INFO - CSV des stats descriptives généré : phase3_var_stats.csv
# 2025-05-16 12:33:05,382 - INFO - Dates détectées : ['Date de début initiale', 'Date prévisionnelle début projet', 'Date de fin initiale', 'Date de début actualisée', 'Date de fin actualisée', "Date d'enregistrement", 'Date de modification', 'Date de reconduction', 'Date de fin réelle', 'Date de fin de Delivery', 'Date de résiliation', 'Date annonce Résiliation', 'Date de signature', 'flag_date_anomalie', 'flag_date_anomalie_date_de_d_but_initiale', 'flag_date_anomalie_date_pr_visionnelle_d_but_projet', 'flag_date_anomalie_date_de_fin_initiale', 'flag_date_anomalie_date_de_d_but_actualis_e', 'flag_date_anomalie_date_de_fin_actualis_e', 'flag_date_anomalie_date_d_enregistrement', 'flag_date_anomalie_date_de_modification', 'flag_date_anomalie_date_de_reconduction', 'flag_date_anomalie_date_de_fin_r_elle', 'flag_date_anomalie_date_de_fin_de_delivery', 'flag_date_anomalie_date_de_r_siliation', 'flag_date_anomalie_date_annonce_r_siliation', 'flag_date_anomalie_date_de_signature']
# 2025-05-16 12:33:05,382 - INFO - Montants détectées : ['Total recette actualisé', 'Total recette réalisé', 'Total recette produit', 'Budget client estimé']
# 2025-05-16 12:33:05,383 - ERROR - Aucune colonne détectée pour : marges
# 2025-05-16 12:33:05,383 - INFO - Durées détectées : ['Unité de durée', 'Durée engagement (mois)']
# 2025-05-16 12:33:05,383 - INFO - Étape A terminée. Vérifiez phase3_var_stats.csv et phase3_stats.log
# 2025-05-16 12:33:05,383 - INFO - Démarrage de la détection univariée des outliers
# 2025-05-16 12:33:05,385 - INFO - Avant_vente                    →   104 outliers univariés (dont 0 sous le min, 104 au-delà du max)
# 2025-05-16 12:33:05,390 - INFO - Total recette actualisé        →  2072 outliers univariés (dont 0 sous le min, 2072 au-delà du max)
# 2025-05-16 12:33:05,397 - INFO - Total recette réalisé          →  2086 outliers univariés (dont 0 sous le min, 2086 au-delà du max)
# 2025-05-16 12:33:05,405 - INFO - Total recette produit          →  2140 outliers univariés (dont 0 sous le min, 2140 au-delà du max)
# 2025-05-16 12:33:05,411 - INFO - Unnamed: 15                    →     0 outliers univariés
# 2025-05-16 12:33:05,414 - INFO - Unnamed: 18                    →     0 outliers univariés
# 2025-05-16 12:33:05,418 - INFO - Charge prévisionnelle projet   →     0 outliers univariés
# 2025-05-16 12:33:05,423 - INFO - Budget client estimé           →  1922 outliers univariés (dont 0 sous le min, 1922 au-delà du max)
# 2025-05-16 12:33:05,430 - INFO - Durée engagement (mois)        →    16 outliers univariés (dont 5 sous le min, 11 au-delà du max)
# 2025-05-16 12:33:05,434 - INFO - Commentaire renouvellement     →     0 outliers univariés
# 2025-05-16 12:33:05,982 - INFO - Export univ terminé : 7690 outliers univariés exportés, 12819 lignes restantes (jeu nettoyé)
# 2025-05-16 12:33:06,003 - INFO - Export comparaison des seuils univariés : phase3_univ_threshold_comparison.csv
# 2025-05-16 12:33:06,221 - INFO - Démarrage détection multivariée des outliers
# 2025-05-16 12:33:06,571 - INFO - [Brut] IF=1624 outliers, LOF=2479 outliers
# 2025-05-16 12:33:06,581 - INFO - Prétraitement multivarié appliqué : StandardScaler + imputation médiane
# 2025-05-16 12:33:06,777 - INFO - [Prétraité] IF=1633 outliers (29 labels modifiés vs brut 1624)
# 2025-05-16 12:33:07,062 - INFO - [Prétraité] LOF=2576 outliers (501 labels modifiés vs brut 2479)
# 2025-05-16 12:33:07,298 - INFO - Multivarié (IF ∪ LOF) → 3915 outliers exportés
# 2025-05-16 12:33:09,944 - INFO - Jeu nettoyé multivarié : 12185 lignes restantes
# 2025-05-16 12:33:09,945 - INFO - Synthèse et export des outliers – phase3
# 2025-05-16 12:33:09,960 - INFO - Export du résumé des outliers : phase3_outlier_summary.csv
# 2025-05-16 12:33:09,960 - INFO - Consolidation globale des outliers terminée (barplot + Top 20).
# 2025-05-16 12:33:09,960 - INFO - Tentatives de typologie (clustering) des outliers multivariés non concluantes – étape abandonnée
# 2025-05-16 12:33:11,226 - INFO - Exports terminés : phase3_all_flags.csv, phase3_outliers_all.csv, phase3_cleaned_all.csv
# 2025-05-16 12:33:11,250 - INFO - Export impact statistique de l'exclusion : phase3_cleaning_impact.csv
# 2025-05-16 12:33:11,250 - INFO - Génération des visualisations
# 2025-05-16 12:33:13,336 - INFO - Skip histogramme Avant_vente : variable non pertinente pour histogramme
# 2025-05-16 12:34:23,607 - WARNING - Skip histogramme Unnamed: 15 : pas de données
# 2025-05-16 12:34:23,608 - WARNING - Skip histogramme Unnamed: 18 : pas de données
# 2025-05-16 12:34:23,608 - INFO - Skip hist log Charge prévisionnelle projet : pas de valeurs > 0
# 2025-05-16 12:34:48,544 - INFO - Skip histogramme Durée engagement (mois) : variable non pertinente pour histogramme
# 2025-05-16 12:34:48,545 - WARNING - Skip histogramme Commentaire renouvellement  : pas de données
# 2025-05-16 12:34:49,106 - INFO - QQ-plot généré pour log(Total recette réalisé) → phase3_qq_total_recette_r_alis.png
# 2025-05-16 12:34:49,728 - INFO - KDE cumulative générée pour log(Total recette réalisé) → phase3_kde_cum_total_recette_r_alis.png
# 2025-05-16 12:34:49,730 - WARNING - Package 'diptest' non installé : test de bimodalité ignoré
# 2025-05-16 12:34:50,209 - INFO - QQ-plot généré pour Durée engagement (mois) → phase3_qq_dur_e_engagement_mois.png
# 2025-05-16 12:34:50,676 - INFO - KDE cumulative générée pour Durée engagement (mois) → phase3_kde_cum_dur_e_engagement_mois.png
# 2025-05-16 12:34:50,677 - WARNING - Package 'diptest' non installé : test de bimodalité ignoré
# 2025-05-16 12:34:50,678 - WARNING - 788 valeurs manquantes dans Avant_vente
# 2025-05-16 12:34:51,372 - INFO - Countplot Avant_vente généré : phase3_count_avant_vente.png
# 2025-05-16 12:34:51,940 - INFO - Stripplot linéaire complet généré : phase3_strip_duree_full_lin.png
# 2025-05-16 12:34:52,593 - INFO - Stripplot log complet généré : phase3_strip_duree_full_log.png
# 2025-05-16 12:34:53,083 - INFO - Stripplot zoom P0–95% généré : phase3_strip_duree_zoom.png
# 2025-05-16 12:34:53,083 - INFO - Durée d’engagement : missing à 97 % (variable supprimée)
# 2025-05-16 12:35:13,420 - INFO - Heatmaps côte-à-côte générées : phase3_heatmap_corr_comparison.png
# 2025-05-16 12:35:13,945 - INFO - Graphique outliers univariés par variable : phase3_outlier_counts_univariate.png
# 2025-05-16 12:35:14,288 - INFO - Diagramme Venn IF vs LOF généré : phase3_venn_if_lof.png
# 2025-05-16 12:35:14,289 - INFO - Visualisations générées et sauvegardées
# 2025-05-16 12:35:14,296 - INFO - Export crosstab Statut production × flag_multivariate : phase3_status_flag_multivariate.csv
# 2025-05-16 12:35:14,846 - INFO - Barplot des statuts des outliers multivariés généré : phase3_status_multivar_outliers.png
# 2025-05-16 12:35:14,846 - INFO - 11.0% des outliers multivariés sont des affaires perdues
# 2025-05-16 12:35:15,472 - INFO - Export Top 20 des outliers extrêmes : phase3_outliers_top20.csv
# 2025-05-16 12:35:15,474 - INFO - Export commentaires Top 20 : phase3_top20_commentary.txt
# 2025-05-16 12:35:15,684 - INFO - Export combinaisons rares : phase3_rare_combinations.csv
# 2025-05-16 12:35:16,513 - INFO - Bar-chart rare variables généré : phase3_bar_rare_variables.png
# 2025-05-16 12:35:16,520 - INFO - Export extrait modalités rares : phase3_rare_modalities_excerpt.csv
# 2025-05-16 12:35:16,524 - INFO - Export Top 5 clients outliers : phase3_top5_clients_outliers.csv
# 2025-05-16 12:35:16,526 - INFO - Export Top 5 piliers outliers : phase3_top5_piliers_outliers.csv
# 2025-05-16 12:35:17,070 - INFO - Barplot outliers par pilier généré : phase3_bar_outliers_par_pilier.png
# 2025-05-16 12:35:23,345 - INFO - PDF des figures de phase 3 mis à jour : phase3_figures.pdf
# 2025-05-16 12:35:23,347 - INFO - Listing des fichiers produit : phase3_output_files.txt


# phase3.log
# phase3_all_flags.csv
# phase3_bar_outlier_families.png
# phase3_bar_outliers_par_pilier.png
# phase3_bar_rare_variables.png
# phase3_boxplots_all_log.png
# phase3_boxplots_all_log_cleaned.png
# phase3_categorical_overview.csv
# phase3_cleaned_all.csv
# phase3_cleaned_multivariate.csv
# phase3_cleaned_univ.csv
# phase3_cleaning_impact.csv
# phase3_count_avant_vente.png
# phase3_date_anomalies.csv
# phase3_date_overview.csv
# phase3_figures.pdf
# phase3_figures_index.png
# phase3_heatmap_corr_comparison.png
# phase3_hist_log_budget_client_estime.png
# phase3_hist_log_budget_client_estime_cleaned.png
# phase3_hist_log_total_recette_actualise.png
# phase3_hist_log_total_recette_actualise_cleaned.png
# phase3_hist_log_total_recette_produit.png
# phase3_hist_log_total_recette_produit_cleaned.png
# phase3_hist_log_total_recette_realise.png
# phase3_hist_log_total_recette_realise_cleaned.png
# phase3_kde_cum_dur_e_engagement_mois.png
# phase3_kde_cum_total_recette_r_alis.png
# phase3_numeric_overview.csv
# phase3_outlier_counts_univariate.png
# phase3_outlier_families.csv
# phase3_outlier_summary.csv
# phase3_outliers_all.csv
# phase3_outliers_iforest.csv
# phase3_outliers_lof.csv
# phase3_outliers_multivariate.csv
# phase3_outliers_top20.csv
# phase3_outliers_univariate.csv
# phase3_output_files.txt
# phase3_pca_loadings.png
# phase3_pca_outliers.png
# phase3_qq_dur_e_engagement_mois.png
# phase3_qq_total_recette_r_alis.png
# phase3_rare_combinations.csv
# phase3_rare_modalities_excerpt.csv
# phase3_scatter_total_recette_actualise_budget_client_estime.png
# phase3_scatter_total_recette_actualise_budget_client_estime_loglog.png
# phase3_scatter_total_recette_actualise_total_recette_produit.png
# phase3_scatter_total_recette_actualise_total_recette_produit_loglog.png
# phase3_scatter_total_recette_actualise_total_recette_realise.png
# phase3_scatter_total_recette_actualise_total_recette_realise_loglog.png
# phase3_scatter_total_recette_produit_budget_client_estime.png
# phase3_scatter_total_recette_produit_budget_client_estime_loglog.png
# phase3_scatter_total_recette_realise_budget_client_estime.png
# phase3_scatter_total_recette_realise_budget_client_estime_loglog.png
# phase3_scatter_total_recette_realise_total_recette_produit.png
# phase3_scatter_total_recette_realise_total_recette_produit_loglog.png
# phase3_stats.log
# phase3_status_flag_multivariate.csv
# phase3_status_multivar_outliers.png
# phase3_strip_duree_full_lin.png
# phase3_strip_duree_full_log.png
# phase3_strip_duree_zoom.png
# phase3_top20_commentary.txt
# phase3_top5_clients_outliers.csv
# phase3_top5_piliers_outliers.csv
# phase3_univ_threshold_comparison.csv
# phase3_var_stats.csv
# phase3_venn_if_lof.png
