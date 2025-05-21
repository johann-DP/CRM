#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 2 – EDA : Analyse exploratoire globale
Étapes 2.1 à 2.5 : KPI globaux, tendance temporelle, répartition par statut,
Top clients, répartition par entité et type d’opportunité
"""
import os
import logging
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns

# Harmonisation graphique avec Phase 1
# cf. phase1.py : sns.set_theme et rcParams.update :contentReference[oaicite:0]{index=0}:contentReference[oaicite:1]{index=1}
sns.set_theme(style="whitegrid", font_scale=1.3)
plt.rcParams.update({
    "axes.titlesize":    16,
    "axes.labelsize":    14,
    "xtick.labelsize":   12,
    "ytick.labelsize":   12,
    "figure.dpi":        200
})

# 1. Définition des chemins
DATA_PATH  = r"D:\DATAPREDICT\DATAPREDICT 2024\Missions\Digora\phase1_output\export_phase1_cleaned.csv"
OUTPUT_DIR = r"D:\DATAPREDICT\DATAPREDICT 2024\Missions\Digora\phase2_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)
logging.info(f"Dossier d'export Phase 2 prêt : {OUTPUT_DIR}")

# 2. Configuration du logger
LOG_PATH = os.path.join(OUTPUT_DIR, "phase2.log")
# Logger racine
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
logger.info(f"Logger configuré : console + fichier ({LOG_PATH})")

# 3. Chargement du jeu de données
try:
    df = pd.read_csv(DATA_PATH)
    logging.info(f"Jeu nettoyé chargé : {df.shape[0]} lignes × {df.shape[1]} colonnes")
except Exception as e:
    logging.error(f"Impossible de charger le fichier : {e}")
    raise

# 3a. Vérifications métier & robustesse – assertions sur colonnes critiques
required_cols = [
    'Statut commercial',
    'Total recette réalisé',
    "Date d'enregistrement",
    'Date de fin réelle',
    'Client',
    'Code'
]
for col in required_cols:
    assert col in df.columns, f"Colonne critique manquante: {col}"
logger.info("Vérification des colonnes critiques terminée.")

# 4a. Bucketing des modalités rares (copié de Phase 1)
to_bucket = [
    "Type opportunité",
    "Pilier",
    "Entité opérationnelle",
    "Statut commercial",
    "Statut production",
    "Catégorie",
    "Sous-catégorie",
]
# Ne bucketer que les colonnes existantes, loguer les autres
existing_bucket = [c for c in to_bucket if c in df.columns]
missing_bucket  = [c for c in to_bucket if c not in df.columns]
if missing_bucket:
    logger.warning(f"Colonnes introuvables pour bucketing, ignorées : {missing_bucket}")
to_bucket = existing_bucket
# seuil de 1 % pour regrouper en "Autres"
seuil = 0.01
for col in to_bucket:
    if col in df.columns:
        freq = df[col].value_counts(normalize=True)
        rares = freq[freq < seuil].index.tolist()
        if rares:
            df[col] = df[col].where(~df[col].isin(rares), other="Autres")
            logger.info(
                f"Bucketing {col} : {len(rares)} modalités rares "
                f"({sum(freq[rares])*100:.2f}% des enregistrements) regroupées en 'Autres'"
            )
    else:
        logger.warning(f"Colonne {col!r} introuvable pour bucketing")

# 4b. Extension du bucketing pour Chef de projet & Motif non conformité
additional = ["Chef de projet", "Motif non conformité"]
existing_add   = [c for c in additional if c in df.columns]
missing_add    = [c for c in additional if c not in df.columns]
if missing_add:
    logger.warning(f"Colonnes introuvables pour export des manquants, ignorées : {missing_add}")
for col in existing_add:
    if col in df.columns:
        mask_na = df[col].isna()
        if mask_na.any():
            fname = f"missing_{col.replace(' ', '_')}.csv"
            df.loc[mask_na].to_csv(os.path.join(OUTPUT_DIR, fname), index=False)
            logger.info(f"{mask_na.sum()} enregistrements sans '{col}' exportés dans {fname}")
    else:
        logger.warning(f"Colonne {col!r} introuvable pour export des manquants")

# 4c. Génération du dictionnaire de données (Phase 1 style) :contentReference[oaicite:0]{index=0}:contentReference[oaicite:1]{index=1}
data_dict = []
for col in df.columns:
    dtype      = df[col].dtype
    n_unique   = df[col].nunique(dropna=False)
    missing    = df[col].isna().sum()
    missing_pct= missing / len(df) * 100
    # Exemples de modalités (jusqu’à 5 valeurs distinctes)
    samples    = df[col].dropna().astype(str).unique()[:5]
    sample_str = "; ".join(samples)
    data_dict.append({
        "column"       : col,
        "dtype"        : str(dtype),
        "n_unique"     : n_unique,
        "missing_count": missing,
        "missing_pct"  : round(missing_pct, 2),
        "sample_values": sample_str
    })
data_dict_df = pd.DataFrame(data_dict)
# Export en Excel pour intégration rapide au rapport
dict_path = os.path.join(OUTPUT_DIR, "phase2_data_dictionary.xlsx")
with pd.ExcelWriter(dict_path, engine="xlsxwriter") as writer:
    data_dict_df.to_excel(writer, sheet_name="Data_Dictionary", index=False)
logger.info(f"Dictionnaire de données exporté vers {dict_path}")

# 4d. Création de variables métier & journal de nettoyage (Phase 1 style) :contentReference[oaicite:2]{index=2}:contentReference[oaicite:3]{index=3}
# 1) Calcul des marges (CA réalisé – coûts)
cost_cols = ['Charge prévisionnelle projet', 'Budget client estimé']
cost_col = next((c for c in cost_cols if c in df.columns), None)

if cost_col:
    df['margin_abs'] = df['Total recette réalisé'] - df[cost_col]
    df['margin_pct'] = df['margin_abs'] / df['Total recette réalisé'].replace(0, np.nan) * 100
    df['flag_margin_low'] = df['margin_pct'] < 10  # vrai si marge < 10 %
    logger.info(f"Marge calculée ({cost_col}) : abs, %, flag faible/negatif")
else:
    logger.warning("Aucun champ de coût détecté pour le calcul de marge")

# 2) Suivi budgétaire (CA réalisé / Budget estimé)
if 'Budget client estimé' in df.columns:
    df['ecart_budget'] = (
        df['Total recette réalisé']
        / df['Budget client estimé'].replace(0, np.nan)
    *100
    )
    logger.info("Variable 'ecart_budget' calculée")
else:
    logger.warning("Colonne 'Budget client estimé' introuvable pour suivi budgétaire")

# 3) Respect des délais (Date de fin prévue vs Date de fin réelle)
expected_end_cols = ['Date de fin initiale', 'Date de fin actualisée']
perf_date = None

for col in expected_end_cols:
    if col in df.columns:
        perf_date = col
        df[col] = pd.to_datetime(df[col], errors='coerce')
        n_bad = df[col].isna().sum()
        logger.info(f"{col} : {n_bad} valeurs non converties (NaT) après to_datetime")
    break

# === Étape de vérification manuelle des dates hors bornes ===
# Initialisation des variables pour le contrôle des dates
date_cols = ["Date d'enregistrement", "Date de fin réelle"]
min_date  = pd.Timestamp("2010-01-01")
max_date  = pd.Timestamp.today()
# 1) Forcer la conversion en datetime sur les deux colonnes avant le filtrage
for col in date_cols:
    df[col] = pd.to_datetime(df[col], errors='coerce')
    n_nat = df[col].isna().sum()
    logger.info(f"{col}: conversion forcée en datetime, {n_nat} NaT après coercion")
for col in date_cols:
    # masque des dates hors bornes potentielles
    mask_bad = df[col].notna() & ((df[col] < min_date) | (df[col] > max_date))
    n_bad = mask_bad.sum()
    if n_bad:
        # 1) Export des cas isolés pour revue
        bad_path = os.path.join(OUTPUT_DIR, f"out_of_bounds_{col.replace(' ', '_')}.csv")
        df.loc[mask_bad, ['Code', col]].to_csv(bad_path, index=False)
        logger.info(f"{n_bad} enregistrements hors bornes pour {col} exportés dans {bad_path}")

        # 2) Affichage d’un échantillon pour valider
        sample = df.loc[mask_bad, col].sort_values().head(5)
        logger.info(f"Exemple de dates hors bornes pour {col} :\n{sample.to_list()}")

        # 3) Stats sur la distribution des dates (min / median / max)
        non_na = df[col].dropna()
        if not non_na.empty:
            dmin = non_na.min()
            dmed = non_na.median()
            dmax = non_na.max()
            logger.info(
                        f"Dates hors bornes pour {col} – "
                        f"min : {dmin}, median : {dmed}, max : {dmax}"
        )
        else:
            logger.info(f"Aucune date valide pour {col}, pas de stats produites")

if perf_date:
    # 1) Forcer la conversion en datetime sur **les deux** colonnes
    df['Date de fin réelle'] = pd.to_datetime(df['Date de fin réelle'], errors='coerce')
    df[perf_date]           = pd.to_datetime(df[perf_date], errors='coerce')

    # 2) Calcul du délai en jours et indicateur on_time
    df['delay_days'] = (df['Date de fin réelle'] - df[perf_date]).dt.days
    df['on_time']    = df['delay_days'] <= 0
    logger.info(f"Variables 'delay_days' et 'on_time' calculées à partir de {perf_date}")
else:
    logger.warning("Aucun champ de date de fin prévue pour suivi des délais")

# 4) Export du journal de nettoyage / variables métier
journal_cols = ['Code'] + [c for c in [
    'margin_abs', 'margin_pct', 'flag_margin_low',
    'ecart_budget', 'delay_days', 'on_time'
] if c in df.columns]
journal_df = df[journal_cols]
journal_path = os.path.join(OUTPUT_DIR, "phase2_business_variables.csv")
journal_df.to_csv(journal_path, index=False)
logger.info(f"Journal de variables métier exporté vers {journal_path}")

# 4. KPI globaux (Étape 2.1)
total_ops       = len(df)
status_counts   = df['Statut commercial'].value_counts(dropna=False)
# 4d. Vérification cohérence statut commercial
total_status = status_counts.sum()
assert total_status == total_ops, f"Somme des statuts ({total_status}) != total_ops ({total_ops})"
logger.info(f"Vérification statuts : {total_status} = total_ops ({total_ops})")
won             = status_counts.get('Gagné', 0)
lost            = status_counts.get('Perdu', 0)
in_progress     = status_counts.get('En cours', 0)
conversion_rate = (won / total_ops) * 100 if total_ops else np.nan

total_revenue       = df['Total recette réalisé'].sum()
average_deal_value  = df.loc[df['Total recette réalisé'] > 0, 'Total recette réalisé'].mean()

# Dates pour durée projet
df["Date d'enregistrement"] = pd.to_datetime(df["Date d'enregistrement"], errors='coerce')
df['Date de fin réelle']    = pd.to_datetime(df['Date de fin réelle'],    errors='coerce')
# Dates pour durée projet
df["Date d'enregistrement"] = pd.to_datetime(df["Date d'enregistrement"], errors='coerce')
df['Date de fin réelle']    = pd.to_datetime(df['Date de fin réelle'],    errors='coerce')

# 4e. Contrôles renforcés sur les dates
date_cols = ["Date d'enregistrement", 'Date de fin réelle']
# 1) Logger les NaT résultants
for col in date_cols:
    n_nat = df[col].isna().sum()
    logger.info(f"{col}: {n_nat} valeurs non converties (NaT) après to_datetime")

# 2) Filtrage des dates hors bornes métiers (ex. 01/01/2010 – aujourd'hui)
min_date = pd.Timestamp('2010-01-01')
max_date = pd.Timestamp.today()
for col in date_cols:
    mask = df[col].notna() & ((df[col] < min_date) | (df[col] > max_date))
    if mask.any():
        n_bad = mask.sum()
        logger.warning(
            f"{col}: {n_bad} dates hors bornes métiers ({min_date.date()}–{max_date.date()}) "
            "passées en NaT"
        )
        df.loc[mask, col] = pd.NaT

# Calcul de la durée moyenne des projets
df['duration_days']         = (df['Date de fin réelle'] - df["Date d'enregistrement"]).dt.days
average_duration_days      = df['duration_days'].mean()

kpis = {
    'total_opportunities'   : total_ops,
    'won'                   : won,
    'lost'                  : lost,
    'in_progress'           : in_progress,
    'conversion_rate_%'     : conversion_rate,
    'total_revenue'         : total_revenue,
    'average_deal_value'    : average_deal_value,
    'average_duration_days' : average_duration_days
}
# Affichage et export
logging.info("=== KPI GLOBAUX Phase 2 ===")
for k, v in kpis.items():
    logger.info(f"{k} = {v}")
try:
    pd.DataFrame([kpis]).to_csv(os.path.join(OUTPUT_DIR, "phase2_global_kpis.csv"), index=False)
    logger.info("KPI globaux exportés vers phase2_global_kpis.csv")
except Exception as e:
    logger.error(f"Erreur lors de l'export des KPI globaux : {e}")
    raise

# 5. Analyse temporelle (Étape 2.2)
df['quarter'] = df["Date d'enregistrement"].dt.to_period('Q').astype(str)
ts = df.groupby('quarter').agg(
    n_ops  = ('Statut commercial', 'count'),
    revenue= ('Total recette réalisé', 'sum')
)
# 5b. Vérifications de cohérence temporelle
total_ts_ops = ts['n_ops'].sum()
assert total_ts_ops == total_ops, f"Somme n_ops trimestrielles ({total_ts_ops}) != total_ops ({total_ops})"
logger.info(f"Vérification temporelle : {total_ts_ops} = total_ops ({total_ops})")
total_ts_rev = ts['revenue'].sum()
assert np.isclose(total_ts_rev, total_revenue), f"Somme CA trimestriel ({total_ts_rev}) != total_revenue ({total_revenue})"
logger.info(f"Vérification CA : {total_ts_rev} = total_revenue ({total_revenue})")
plt.figure(figsize=(16, 6))
ax1 = ts['n_ops'].plot(marker='o', ms=6, lw=2, label="Nb opportunités")
ax2 = ts['revenue'].plot(marker='s', ms=6, lw=2, secondary_y=True, label="CA signé (€)")
ax1.set_xlabel('Trimestre')
ax1.set_ylabel("Nbr opportunités")
ax2.set_ylabel("CA signé (€)")
plt.title("Évolution trimestrielle des opportunités et du CA")
ax1.legend(loc='upper left'); ax2.legend(loc='upper right')
# légendes déportées et ticks pivotés
ax1.legend(loc='lower left', bbox_to_anchor=(0, 1.02), ncol=2)
ax2.legend(loc='lower right', bbox_to_anchor=(1, 1.02), ncol=2)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
try:
    plt.savefig(os.path.join(OUTPUT_DIR, "phase2_timeseries.png"))
    logger.info("Graphique temporel exporté vers phase2_timeseries.png")
except Exception as e:
    logger.error(f"Erreur lors de la sauvegarde du graphique temporel : {e}")
    raise
finally:
    plt.close()

# 5c. Évolution trimestrielle par Statut commercial (courbes)
if 'quarter' in df.columns:
    status_ts = (
        df.groupby(['quarter', 'Statut commercial'])
          .size()
          .unstack(fill_value=0)
    )
    fig, ax = plt.subplots(figsize=(12, 6))
    status_ts.plot(marker='o', ax=ax)
    ax.set_title("Évolution trimestrielle du nombre d’opportunités par statut")
    ax.set_xlabel("Trimestre")
    ax.set_ylabel("Nombre d’opportunités")
    ax.legend(title="Statut", loc='upper left', bbox_to_anchor=(1,1))
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "phase2_timeseries_by_status.png")
    fig.savefig(path)
    plt.close(fig)
    logger.info(f"Évolution nb opp par statut exportée vers {os.path.basename(path)}")

    # 5d. CA trimestriel par Entité opérationnelle (top 5)
    entity_ts = (
        df.groupby(['quarter', 'Entité opérationnelle'])['Total recette réalisé']
          .sum()
          .unstack(fill_value=0)
    )
    top5_ent = entity_ts.sum().sort_values(ascending=False).head(5).index
    fig, ax = plt.subplots(figsize=(12, 6))
    entity_ts[top5_ent].plot(marker='s', ax=ax)
    ax.set_title("Évolution trimestrielle du CA signé (Top 5 Entités)")
    ax.set_xlabel("Trimestre")
    ax.set_ylabel("CA réalisé (€)")
    ax.legend(title="Entité", loc='upper left', bbox_to_anchor=(1,1))
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "phase2_timeseries_by_entity.png")
    fig.savefig(path)
    plt.close(fig)
    logger.info(f"Évolution CA par entité exportée vers {os.path.basename(path)}")

    # 5e. CA trimestriel par Type d’opportunité (top 5)
    type_ts = (
        df.groupby(['quarter', "Type opportunité"])['Total recette réalisé']
          .sum()
          .unstack(fill_value=0)
    )
    top5_type = type_ts.sum().sort_values(ascending=False).head(5).index
    fig, ax = plt.subplots(figsize=(12, 6))
    type_ts[top5_type].plot(marker='^', ax=ax)
    ax.set_title("Évolution trimestrielle du CA signé (Top 5 Types d’opportunité)")
    ax.set_xlabel("Trimestre")
    ax.set_ylabel("CA réalisé (€)")
    ax.legend(title="Type d’opportunité", loc='upper left', bbox_to_anchor=(1,1))
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "phase2_timeseries_by_type.png")
    fig.savefig(path)
    plt.close(fig)
    logger.info(f"Évolution CA par type d’opportunité exportée vers {os.path.basename(path)}")
else:
    logger.warning("Colonne 'quarter' introuvable, évolutions temporelles détaillées ignorées")

# 6. Répartition par statut commercial (Étape 2.3)
counts   = status_counts
revenues = df.groupby('Statut commercial')['Total recette réalisé'].sum()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14,6))
counts.plot(kind='bar', ax=ax1); ax1.set_title("Nb opportunités par statut"); ax1.set_ylabel("Nombre")
revenues.plot(kind='bar', ax=ax2); ax2.set_title("CA réalisé par statut (€)"); ax2.set_ylabel("CA (€)")
# rotation uniforme des labels
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')
ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "phase2_status_pipeline.png"))
plt.close(fig)
logging.info("Répartition par statut exportée vers phase2_status_pipeline.png")

# 6b. Diagramme empilé du pipeline montants par statut par trimestre
if 'quarter' in df.columns:
    # Agrégation CA par statut et par trimestre
    pipeline_rev = (
        df.groupby(['quarter', 'Statut commercial'])['Total recette réalisé']
        .sum()
        .unstack(fill_value=0)
    )

    # Figure à deux subplots : normal scale (haut) et log scale (bas)
    fig, axes = plt.subplots(2, 1, figsize=(12, 12), sharex=True)

    # 1) Échelle normale
    pipeline_rev.plot(
        kind='bar',
        stacked=True,
        ax=axes[0],
        color=sns.color_palette("Dark2", len(pipeline_rev.columns))
    )
    axes[0].set_title("Pipeline CA par Statut commercial par trimestre (échelle normale)")
    axes[0].set_ylabel("CA réalisé (€)")
    # Ticks tous les 4 trimestres, pivotés
    quarters = pipeline_rev.index.tolist()
    axes[0].set_xticks(range(len(quarters))[::4])
    axes[0].set_xticklabels(quarters[::4], rotation=45, ha='right')
    axes[0].legend(title="Statut", loc='upper left', bbox_to_anchor=(1, 1))

    # 2) Échelle log
    pipeline_rev.plot(
        kind='bar',
        stacked=True,
        ax=axes[1],
        color=sns.color_palette("Dark2", len(pipeline_rev.columns))
    )
    axes[1].set_yscale('log')
    axes[1].set_title("Pipeline CA par Statut commercial par trimestre (échelle log)")
    axes[1].set_xlabel("Trimestre")
    axes[1].set_ylabel("CA réalisé (€) [log scale]")
    axes[1].set_xticks(range(len(quarters))[::4])
    axes[1].set_xticklabels(quarters[::4], rotation=45, ha='right')
    axes[1].legend(title="Statut", loc='upper left', bbox_to_anchor=(1, 1))

    plt.tight_layout()
    both_path = os.path.join(OUTPUT_DIR, "phase2_pipeline_trimestriel_normal_vs_log.png")
    fig.savefig(both_path)
    plt.close(fig)
    logger.info(f"Comparatif pipeline normal vs log exporté vers {os.path.basename(both_path)}")
else:
    logger.warning("Colonne 'quarter' introuvable, comparatif pipeline ignoré")

# 6c. Pipeline global empilé (Étape 2.3) – CA total par statut en une seule barre
# status_rev = df.groupby('Statut commercial')['Total recette réalisé'].sum()
# pipeline_df = status_rev.to_frame().T
# fig, ax = plt.subplots(figsize=(8, 6))
# pipeline_df.plot(kind='bar', stacked=True, ax=ax,
#                  color=sns.color_palette("Dark2", len(pipeline_df.columns)))
status_rev = df.groupby('Statut commercial')['Total recette réalisé'].sum()
pipeline_df = status_rev.to_frame().T
fig, ax = plt.subplots(figsize=(8, 6))
# Palette “deep” pour cohérence Phase 3 + contour noir
palette = sns.color_palette("deep", len(pipeline_df.columns))
pipeline_df.plot(kind='bar', stacked=True, ax=ax,
                 color=palette, edgecolor='black')

# calcul du total
total = status_rev.sum()
for bar in ax.patches:
    h = bar.get_height()
    pct = h / total * 100
    # centre du segment
    x = bar.get_x() + bar.get_width()/2
    y = bar.get_y() + h/2
    ax.text(x, y, f"{h:,.0f} €\n({pct:.1f} %)",
            ha='center', va='center', color='white', fontsize=10)

plt.title("Pipeline global – CA par statut commercial")
# plt.xlabel("")  # pas d’étiquette d’axe des X pour une seule barre
# plt.ylabel("CA réalisé (€)")
# → log–scale sur Y et annotation du tick unique
ax.set_yscale('log')
ax.set_ylabel("CA réalisé (€) [échelle log]")
ax.set_xticks([0])
ax.set_xticklabels(["Pipeline"], rotation=0, ha='center')
ax.legend(title="Statut", loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)
plt.tight_layout()
global_path = os.path.join(OUTPUT_DIR, "phase2_pipeline_global_revenue_stacked.png")
plt.savefig(global_path)
plt.close()
logger.info(f"Pipeline global empilé exporté vers {os.path.basename(global_path)}")

# 7. Top 10 clients (Étape 2.4)
# 7a. Calcul du Top 10 clients par nombre d’affaires
top_clients_count = df['Client'].value_counts().head(10)

# Par nombre d’affaires
fig, ax = plt.subplots(figsize=(12, 6))
bars = ax.barh(top_clients_count.index, top_clients_count.values, color=sns.color_palette("Blues_d", len(top_clients_count)))
ax.invert_yaxis()
ax.set_title("Top 10 clients par nombre d’affaires")
ax.set_xlabel("Nombre d’affaires")
# annotation des valeurs
for bar in bars:
    w = bar.get_width()
    ax.text(w + max(top_clients_count.values)*0.01, bar.get_y() + bar.get_height()/2,
            f"{int(w)}", va='center')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "phase2_top10_clients_count.png"))
plt.close()
logging.info("Top 10 clients (nombre) exportés vers phase2_top10_clients_count.png")

# Par CA réalisé
top_clients_revenue = df.groupby('Client')['Total recette réalisé'].sum().sort_values(ascending=False).head(10)
total_revenue = top_clients_revenue.sum()
fig, ax = plt.subplots(figsize=(12, 6))
bars = ax.barh(top_clients_revenue.index, top_clients_revenue.values, color=sns.color_palette("Blues_d", len(top_clients_revenue)))
ax.invert_yaxis()
ax.set_title("Top 10 clients par CA réalisé")
ax.set_xlabel("CA réalisé (€)")
for bar in bars:
    w = bar.get_width()
    ax.text(w + total_revenue*0.005, bar.get_y()+bar.get_height()/2,
            f"{w/1e6:.1f} M", va='center')
# --- Annotation des % de part relative ---
for bar in ax.patches:
    width = bar.get_width()
    pct = width / total_revenue * 100
    ax.text(
        width + total_revenue * 0.01,
        bar.get_y() + bar.get_height()/2,
        f"{pct:.1f}%",
        va="center"
    )
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "phase2_top10_clients_revenue.png"))
plt.title("Top 10 clients par CA réalisé"); plt.xlabel("CA réalisé (€)")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "phase2_top10_clients_revenue.png"))
plt.close()
logging.info("Top 10 clients (CA) exportés vers phase2_top10_clients_revenue.png")

# 7b. Pourcentage du CA total pour le Top 10 clients (Phase 2 enrichi) :contentReference[oaicite:0]{index=0}:contentReference[oaicite:1]{index=1}
# total_revenue = df['Total recette réalisé'].sum()
# top_clients_pct = (top_clients_revenue / total_revenue) * 100
# # Export CSV
# pct_path = os.path.join(OUTPUT_DIR, "phase2_top10_clients_revenue_pct.csv")
# top_clients_pct.to_csv(pct_path, header=["pct_total_revenue"], index=True)
# logger.info(f"Pourcentage du CA total pour le Top 10 clients exporté vers {os.path.basename(pct_path)}")
# # (Optionnel) Graphique des pourcentages
# fig, ax = plt.subplots(figsize=(12, 6))
# bars = ax.barh(top_clients_pct.index, top_clients_pct.values, color=sns.color_palette("Blues_d", len(top_clients_pct)))
# ax.invert_yaxis()
# ax.set_title("Top 10 clients : % du CA total")
# ax.set_xlabel("% du CA total")
# for bar in bars:
#     w = bar.get_width()
#     ax.text(w + max(top_clients_pct)*0.01, bar.get_y()+bar.get_height()/2,
#             f"{w:.1f} %", va='center')
# plt.tight_layout()
# plt.savefig(os.path.join(OUTPUT_DIR, "phase2_top10_clients_revenue_pct.png"))
# plt.title("Top 10 clients : % du CA total"); plt.xlabel("% du CA total")
# plt.tight_layout()
# plt.savefig(os.path.join(OUTPUT_DIR, "phase2_top10_clients_revenue_pct.png"))
# plt.close()
# logger.info("Graphique % du CA total pour le Top 10 clients exporté vers phase2_top10_clients_revenue_pct.png")

# 8. Répartition par Entité opérationnelle et Type d’opportunité (Étape 2.5)
# Entité opérationnelle
entity_rev = df.groupby('Entité opérationnelle')['Total recette réalisé'].sum().sort_values(ascending=False)
# plt.figure(figsize=(12, 6))
# entity_rev.plot(kind='bar'); plt.title("CA par Entité opérationnelle"); plt.xlabel("Entité")
# plt.xticks(rotation=45, ha='right'); plt.tight_layout()
# plt.savefig(os.path.join(OUTPUT_DIR, "phase2_revenue_by_entity.png"))
# plt.close()

fig, ax = plt.subplots(figsize=(12, 6))
# barplot avec edgecolor noir pour homogénéité
entity_rev.plot(kind='bar', edgecolor='black', ax=ax)
ax.set_title("CA par Entité opérationnelle")
ax.set_xlabel("Entité")
ax.set_ylabel("CA réalisé (€)")
ax.set_xticklabels(entity_rev.index, rotation=45, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "phase2_revenue_by_entity.png"))
plt.close()

total_ent = entity_rev.sum()
for bar in ax.patches:
    h = bar.get_height()
    pct = h / total_ent * 100
    x = bar.get_x() + bar.get_width()/2
    # petit offset pour ne pas coller le texte à la barre
    y = h + total_ent * 0.01
    ax.text(x, y, f"{h:,.0f} €\n({pct:.1f} %)",
            ha='center', va='bottom', fontsize=9)

logging.info("CA par entité exporté vers phase2_revenue_by_entity.png")

# Type d’opportunité
type_rev = df.groupby("Type opportunité")['Total recette réalisé'].sum().sort_values(ascending=False)
# plt.figure(figsize=(12, 6))
# type_rev.plot(kind='bar'); plt.title("CA par Type d’opportunité"); plt.xlabel("Type opportunité")
# plt.xticks(rotation=45, ha='right'); plt.tight_layout()
# plt.savefig(os.path.join(OUTPUT_DIR, "phase2_revenue_by_type.png"))
# plt.close()

fig, ax = plt.subplots(figsize=(12, 6))
# barplot avec contour noir
type_rev.plot(kind='bar', edgecolor='black', ax=ax)
ax.set_title("CA par Type d’opportunité")
ax.set_xlabel("Type d’opportunité")
ax.set_ylabel("CA réalisé (€)")
ax.set_xticklabels(type_rev.index, rotation=45, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "phase2_revenue_by_type.png"))
plt.close()

total_type = type_rev.sum()
for bar in ax.patches:
    h = bar.get_height()
    pct = h / total_type * 100
    x = bar.get_x() + bar.get_width()/2
    y = h + total_type * 0.01
    ax.text(x, y, f"{pct:.1f} %", ha='center', va='bottom', fontsize=9)

logging.info("CA par type d’opportunité exporté vers phase2_revenue_by_type.png")

# 8b. Répartition par Commercial responsable (terrain)
if 'Commercial' in df.columns:
    comm_rev = (
            df.groupby('Commercial')['Total recette réalisé']
              .sum()
    .sort_values(ascending=False)
    # .head(38)  # on limite aux non nuls
    .head(20) # pour réduire la taille de la figure (peu lisible sinon)
        )
    fig, ax = plt.subplots(figsize=(12, 6)) # on passe de 10 à 6 quand on passe de 38 à 20
    bars = ax.barh(comm_rev.index, comm_rev.values,
                   color = sns.color_palette("Blues_d", len(comm_rev)))
    ax.invert_yaxis()
    ax.set_title("CA par Commercial responsable (Top 40)")
    ax.set_xlabel("CA réalisé (€)")
    # annotation des valeurs en millions
    for bar in bars:
        w = bar.get_width()
        ax.text(w + total_revenue * 0.001, bar.get_y() + bar.get_height() / 2,
                f"{w / 1e6:.1f} M", va='center', size=10)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "phase2_revenue_by_commercial.png")
    plt.savefig(path)
    plt.close()
    logger.info(f"CA par Commercial responsable exporté vers {os.path.basename(path)}")
else:
    logger.warning("Colonne 'Commercial' introuvable, répartition par commercial ignorée")

# 9. Corrélations entre variables numériques (Étape 2.5)
# Sélection dynamique des variables numériques pour corrélation
all_num = df.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols = [c for c in all_num if df[c].nunique(dropna=True) > 1]
dropped_num = set(all_num) - set(numeric_cols)
if dropped_num:
    logger.info(f"Variables exclues de la corrélation (variance nulle) : {sorted(dropped_num)}")
logger.info(f"Variables retenues pour corrélation : {numeric_cols}")

# Calcul et export de la matrice de corrélation
corr_matrix = df[numeric_cols].corr()
corr_csv    = os.path.join(OUTPUT_DIR, "phase2_correlation_matrix.csv")
corr_matrix.to_csv(corr_csv, index=True)
logger.info(f"Matrice de corrélation exportée vers {corr_csv}")

# Visualisation : heatmap annotée
plt.figure(figsize=(12, 10))
sns.heatmap(
    corr_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    vmin=-1,
    vmax=1,
    linewidths=0.5,
    cbar_kws={'shrink':0.7, 'label':'Coefficient de corrélation'}
)
plt.title("Matrice de corrélation des variables numériques")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
heatmap_png = os.path.join(OUTPUT_DIR, "phase2_correlation_heatmap.png")
plt.savefig(heatmap_png)
plt.close()
logger.info(f"Heatmap de corrélation sauvegardée vers {heatmap_png}")

# ─── Taux de conversion par type d’opportunité ─────────────────────────
sub = df[df["Statut commercial"].isin(["Gagné", "Perdu"])]
conv = (
    sub.groupby("Type")["Statut commercial"]
       .apply(lambda x: (x=="Gagné").sum() / len(x) * 100)
       .sort_values()
)
fig, ax = plt.subplots(figsize=(10, 6), dpi=200)
sns.barplot(x=conv.values, y=conv.index, edgecolor="black", ax=ax)
ax.set_xlabel("Taux de conversion (%)")
ax.set_title("Taux de conversion par type d’opportunité")
# Annotation % sur chaque barre
for bar in ax.patches:
    pct = bar.get_width()
    ax.text(
        pct + 0.5,
        bar.get_y() + bar.get_height()/2,
        f"{pct:.1f}%",
        va="center"
    )
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "phase2_conversion_rate_by_type.png"), dpi=300, bbox_inches="tight")
plt.close(fig)


# → FIN Phase 2 (Étape 2.5 uniquement) ←
logger.info("Phase 2 – EDA exploratoire globale terminée (points 2.1 à 2.5).")

# 10. Génération du PDF regroupant toutes les figures (Phase 1 style)
try:
    pdf_path = os.path.join(OUTPUT_DIR, "phase2_figures.pdf")
    png_files = sorted([f for f in os.listdir(OUTPUT_DIR) if f.lower().endswith(".png")])
    with PdfPages(pdf_path) as pdf:
        for png in png_files:
            fig = plt.figure(dpi=400)
            img = plt.imread(os.path.join(OUTPUT_DIR, png))
            plt.imshow(img)
            plt.axis("off")
            pdf.savefig(fig)
            plt.close(fig)
    logger.info(f"PDF des figures exporté vers {pdf_path}")
except Exception as e:
    logger.error(f"Erreur lors de la génération du PDF des figures : {e}")

# 11. Génération de la liste des fichiers
try:
    list_path = os.path.join(OUTPUT_DIR, "liste_fichiers.txt")
    files = sorted(os.listdir(OUTPUT_DIR))
    with open(list_path, "w", encoding="utf-8") as f:
        for fname in files:
            f.write(f"{fname}\n")
    logger.info(f"Liste des fichiers exportée vers {list_path}")
except Exception as e:
    logger.error(f"Erreur lors de la génération de la liste des fichiers : {e}")

### COMPLEMENT A POSTERIORI ###

# ---------------------------------------------------------------
# Focus Budget ↔ CA réalisé
# ---------------------------------------------------------------
print('#########################################################################\n')
# 0) Filtrage – on exclut les budgets nuls / manquants
mask      = (df['Budget client estimé'] > 0) & df['Total recette réalisé'].notna()
df_budget = df.loc[mask].copy()

# ------------------------------------------------------------------
# 1) Scatter log–log + droite OLS
# ------------------------------------------------------------------
plt.figure(figsize=(12, 6), dpi=200)

# 1) Nuage de points identique (alpha 0.3)
sns.scatterplot(
    x='Budget client estimé',
    y='Total recette réalisé',
    data=df_budget,
    alpha=0.3,
    edgecolor=None
)

# 2) Droite OLS en log10
x = df_budget['Budget client estimé']
y = df_budget['Total recette réalisé']
log_x = np.log10(x)
log_y = np.log10(y)
mask  = (~np.isfinite(log_x)) | (~np.isfinite(log_y))
slope, intercept = np.polyfit(log_x[~mask], log_y[~mask], 1)

# 3) Générer la droite en coordonnées réelles
x_line = np.logspace(np.log10(x.min()), np.log10(x.max()), 100)
y_line = 10**(slope * np.log10(x_line) + intercept)
plt.plot(x_line, y_line, color="red", linewidth=2, label=f"OLS log–log\nslope={slope:.2f}")

# 4) Mise en forme comme avant
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Budget client estimé (€)')
plt.ylabel('CA réalisé (€)')
plt.title('Budget estimé vs CA réalisé – OLS log‑log')
plt.legend()
plt.tight_layout()
scatter_path = os.path.join(OUTPUT_DIR, 'phase2_scatter_budget_vs_real.png')
plt.savefig(scatter_path, dpi=300)
plt.close()

# ------------------------------------------------------------------
# 2) Histogramme du ratio (log-bins, sans les 1 % extrêmes)
# ------------------------------------------------------------------
df_budget['ratio_real_budget'] = df_budget['Total recette réalisé'] / df_budget['Budget client estimé']
ratio         = df_budget['ratio_real_budget'].replace([np.inf, -np.inf], np.nan).dropna()

# On coupe les 1 % valeurs les plus hautes pour éviter la barre à 120 000.
ratio_clip    = ratio[ratio <= ratio.quantile(0.99)]

# Bins logarithmiques de 0,1 à 1 000 (adaptable)
log_bins      = np.logspace(np.log10(0.1), np.log10(1_000), num=40)

plt.figure(figsize=(12, 6), dpi=200)
plt.hist(ratio_clip, bins=log_bins, edgecolor='black')
plt.xscale('log')
plt.axvline(ratio_clip.median(), color='red', linestyle='--', label=f'Médiane = {ratio_clip.median():.2f}')
plt.xlabel('Ratio CA réalisé / Budget initial (échelle log)')
plt.ylabel('Fréquence')
plt.title('Distribution du ratio Réalisé vs Budget (hors 1 % extrême)')
plt.legend()
plt.tight_layout()
hist_path = os.path.join(OUTPUT_DIR, 'phase2_hist_ratio_budget.png')
plt.savefig(hist_path, dpi=300)
plt.close()

# 3) Stats descriptives du ratio filtré
stats = ratio_clip.describe(percentiles=[.1, .25, .5, .75, .9, .95]).round(3)
stats_path = os.path.join(OUTPUT_DIR, 'phase2_budget_stats.csv')
stats.to_csv(stats_path, header=['value'])
print(f"[Focus Budget] Scatter : {scatter_path}")
print(f"[Focus Budget] Histogramme : {hist_path}")
print(f"[Focus Budget] Stats CSV : {stats_path}")

# ---------------------------------------------------------------
# Focus Durées de cycle – Box-plot & Violin-plot
# ---------------------------------------------------------------

# Nettoyage minimal : on garde les lignes où duration_days > 0
df_cycle = df.copy()
df_cycle['duration_days'] = df_cycle['duration_days'].replace([np.inf, -np.inf], np.nan)
df_cycle = df_cycle.loc[df_cycle['duration_days'] > 0]

# -------- 1) Box-plot des durées par statut commercial ---------
plt.figure(figsize=(12, 6), dpi=200)
sns.boxplot(
    data=df_cycle,
    x='Statut commercial',
    y='duration_days',
    showfliers=False,
    order=['Gagné', 'Perdu', 'Abandonné', 'En cours', 'Autres']
)
plt.yscale('log')
plt.xlabel('Statut commercial')
plt.ylabel('Durée totale du projet (jours, échelle log)')
plt.title('Distribution des durées (Date ouverture → Date fin) par statut')
plt.tight_layout()
box_path = os.path.join(OUTPUT_DIR, 'phase2_box_duration_by_status.png')
plt.savefig(box_path, dpi=300)
plt.close()

# -------- 2) Violin-plot du cycle de vente par quartile de CA ----
# On découpe le CA réalisé en quartiles
df_cycle['CA_quartile'] = pd.qcut(
    df_cycle['Total recette réalisé'],
    q=4,
    labels=['Q1 (<=25 %)', 'Q2', 'Q3', 'Q4 (>=75 %)']
)

plt.figure(figsize=(12, 6), dpi=200)
sns.violinplot(
    data=df_cycle,
    x='CA_quartile',
    y='cycle_vente_jours',
    inner='quartile',
    scale='width',
    cut=0
)
plt.yscale('log')
plt.xlabel('Quartile de CA réalisé')
plt.ylabel('Cycle de vente (jours, échelle log)')
plt.title('Cycle de vente selon la taille du deal')
plt.tight_layout()
violin_path = os.path.join(OUTPUT_DIR, 'phase2_violin_cycle_by_CA.png')
plt.savefig(violin_path, dpi=300)
plt.close()

print(f"[Durées] Box-plot : {box_path}")
print(f"[Durées] Violin-plot : {violin_path}")



###################################################################################

# INFO:root:Logger configuré : console + fichier (D:\DATAPREDICT\DATAPREDICT 2024\Missions\Digora\phase2_output\phase2.log)
# 2025-05-18 10:50:24,566 - INFO - Logger configuré : console + fichier (D:\DATAPREDICT\DATAPREDICT 2024\Missions\Digora\phase2_output\phase2.log)
# INFO:root:Jeu nettoyé chargé : 16100 lignes × 51 colonnes
# 2025-05-18 10:50:24,976 - INFO - Jeu nettoyé chargé : 16100 lignes × 51 colonnes
# INFO:root:Vérification des colonnes critiques terminée.
# 2025-05-18 10:50:24,976 - INFO - Vérification des colonnes critiques terminée.
# WARNING:root:Colonnes introuvables pour bucketing, ignorées : ['Pilier']
# 2025-05-18 10:50:24,976 - WARNING - Colonnes introuvables pour bucketing, ignorées : ['Pilier']
# INFO:root:Bucketing Statut commercial : 1 modalités rares (0.94% des enregistrements) regroupées en 'Autres'
# 2025-05-18 10:50:24,981 - INFO - Bucketing Statut commercial : 1 modalités rares (0.94% des enregistrements) regroupées en 'Autres'
# INFO:root:Bucketing Statut production : 1 modalités rares (0.29% des enregistrements) regroupées en 'Autres'
# 2025-05-18 10:50:24,985 - INFO - Bucketing Statut production : 1 modalités rares (0.29% des enregistrements) regroupées en 'Autres'
# INFO:root:Bucketing Sous-catégorie : 1 modalités rares (0.06% des enregistrements) regroupées en 'Autres'
# 2025-05-18 10:50:24,988 - INFO - Bucketing Sous-catégorie : 1 modalités rares (0.06% des enregistrements) regroupées en 'Autres'
# WARNING:root:Colonnes introuvables pour export des manquants, ignorées : ['Motif non conformité']
# 2025-05-18 10:50:24,989 - WARNING - Colonnes introuvables pour export des manquants, ignorées : ['Motif non conformité']
# INFO:root:Dictionnaire de données exporté vers D:\DATAPREDICT\DATAPREDICT 2024\Missions\Digora\phase2_output\phase2_data_dictionary.xlsx
# 2025-05-18 10:50:25,305 - INFO - Dictionnaire de données exporté vers D:\DATAPREDICT\DATAPREDICT 2024\Missions\Digora\phase2_output\phase2_data_dictionary.xlsx
# INFO:root:Marge calculée (Charge prévisionnelle projet) : abs, %, flag faible/negatif
# 2025-05-18 10:50:25,307 - INFO - Marge calculée (Charge prévisionnelle projet) : abs, %, flag faible/negatif
# INFO:root:Variable 'ecart_budget' calculée
# 2025-05-18 10:50:25,308 - INFO - Variable 'ecart_budget' calculée
# INFO:root:Date de fin initiale : 15036 valeurs non converties (NaT) après to_datetime
# 2025-05-18 10:50:25,312 - INFO - Date de fin initiale : 15036 valeurs non converties (NaT) après to_datetime
# INFO:root:Date d'enregistrement: conversion forcée en datetime, 1 NaT après coercion
# 2025-05-18 10:50:25,326 - INFO - Date d'enregistrement: conversion forcée en datetime, 1 NaT après coercion
# INFO:root:Date de fin réelle: conversion forcée en datetime, 8796 NaT après coercion
# 2025-05-18 10:50:25,331 - INFO - Date de fin réelle: conversion forcée en datetime, 8796 NaT après coercion
# INFO:root:78 enregistrements hors bornes pour Date d'enregistrement exportés dans D:\DATAPREDICT\DATAPREDICT 2024\Missions\Digora\phase2_output\out_of_bounds_Date_d'enregistrement.csv
# 2025-05-18 10:50:25,335 - INFO - 78 enregistrements hors bornes pour Date d'enregistrement exportés dans D:\DATAPREDICT\DATAPREDICT 2024\Missions\Digora\phase2_output\out_of_bounds_Date_d'enregistrement.csv
# INFO:root:Exemple de dates hors bornes pour Date d'enregistrement :
# [Timestamp('2009-09-15 00:00:00'), Timestamp('2009-09-21 00:00:00'), Timestamp('2009-09-28 00:00:00'), Timestamp('2009-10-01 00:00:00'), Timestamp('2009-10-01 00:00:00')]
# 2025-05-18 10:50:25,336 - INFO - Exemple de dates hors bornes pour Date d'enregistrement :
# [Timestamp('2009-09-15 00:00:00'), Timestamp('2009-09-21 00:00:00'), Timestamp('2009-09-28 00:00:00'), Timestamp('2009-10-01 00:00:00'), Timestamp('2009-10-01 00:00:00')]
# INFO:root:Dates hors bornes pour Date d'enregistrement – min : 2009-09-15 00:00:00, median : 2020-06-12 09:04:00, max : 2025-04-29 17:03:00
# 2025-05-18 10:50:25,337 - INFO - Dates hors bornes pour Date d'enregistrement – min : 2009-09-15 00:00:00, median : 2020-06-12 09:04:00, max : 2025-04-29 17:03:00
# INFO:root:1 enregistrements hors bornes pour Date de fin réelle exportés dans D:\DATAPREDICT\DATAPREDICT 2024\Missions\Digora\phase2_output\out_of_bounds_Date_de_fin_réelle.csv
# 2025-05-18 10:50:25,341 - INFO - 1 enregistrements hors bornes pour Date de fin réelle exportés dans D:\DATAPREDICT\DATAPREDICT 2024\Missions\Digora\phase2_output\out_of_bounds_Date_de_fin_réelle.csv
# INFO:root:Exemple de dates hors bornes pour Date de fin réelle :
# [Timestamp('2009-12-16 00:00:00')]
# 2025-05-18 10:50:25,341 - INFO - Exemple de dates hors bornes pour Date de fin réelle :
# [Timestamp('2009-12-16 00:00:00')]
# INFO:root:Dates hors bornes pour Date de fin réelle – min : 2009-12-16 00:00:00, median : 2020-10-13 00:00:00, max : 2025-04-30 00:00:00
# 2025-05-18 10:50:25,343 - INFO - Dates hors bornes pour Date de fin réelle – min : 2009-12-16 00:00:00, median : 2020-10-13 00:00:00, max : 2025-04-30 00:00:00
# INFO:root:Variables 'delay_days' et 'on_time' calculées à partir de Date de fin initiale
# 2025-05-18 10:50:25,350 - INFO - Variables 'delay_days' et 'on_time' calculées à partir de Date de fin initiale
# INFO:root:Journal de variables métier exporté vers D:\DATAPREDICT\DATAPREDICT 2024\Missions\Digora\phase2_output\phase2_business_variables.csv
# 2025-05-18 10:50:25,405 - INFO - Journal de variables métier exporté vers D:\DATAPREDICT\DATAPREDICT 2024\Missions\Digora\phase2_output\phase2_business_variables.csv
# INFO:root:Vérification statuts : 16100 = total_ops (16100)
# 2025-05-18 10:50:25,407 - INFO - Vérification statuts : 16100 = total_ops (16100)
# INFO:root:Date d'enregistrement: 1 valeurs non converties (NaT) après to_datetime
# 2025-05-18 10:50:25,430 - INFO - Date d'enregistrement: 1 valeurs non converties (NaT) après to_datetime
# INFO:root:Date de fin réelle: 8796 valeurs non converties (NaT) après to_datetime
# 2025-05-18 10:50:25,431 - INFO - Date de fin réelle: 8796 valeurs non converties (NaT) après to_datetime
# WARNING:root:Date d'enregistrement: 78 dates hors bornes métiers (2010-01-01–2025-05-18) passées en NaT
# 2025-05-18 10:50:25,432 - WARNING - Date d'enregistrement: 78 dates hors bornes métiers (2010-01-01–2025-05-18) passées en NaT
# WARNING:root:Date de fin réelle: 1 dates hors bornes métiers (2010-01-01–2025-05-18) passées en NaT
# 2025-05-18 10:50:25,434 - WARNING - Date de fin réelle: 1 dates hors bornes métiers (2010-01-01–2025-05-18) passées en NaT
# INFO:root:=== KPI GLOBAUX Phase 2 ===
# 2025-05-18 10:50:25,435 - INFO - === KPI GLOBAUX Phase 2 ===
# INFO:root:total_opportunities = 16100
# 2025-05-18 10:50:25,436 - INFO - total_opportunities = 16100
# INFO:root:won = 9348
# 2025-05-18 10:50:25,436 - INFO - won = 9348
# INFO:root:lost = 2103
# 2025-05-18 10:50:25,437 - INFO - lost = 2103
# INFO:root:in_progress = 812
# 2025-05-18 10:50:25,437 - INFO - in_progress = 812
# INFO:root:conversion_rate_% = 58.06211180124223
# 2025-05-18 10:50:25,437 - INFO - conversion_rate_% = 58.06211180124223
# INFO:root:total_revenue = 248632732.80999997
# 2025-05-18 10:50:25,438 - INFO - total_revenue = 248632732.80999997
# INFO:root:average_deal_value = 29104.206006086857
# 2025-05-18 10:50:25,438 - INFO - average_deal_value = 29104.206006086857
# INFO:root:average_duration_days = 353.5581491712707
# 2025-05-18 10:50:25,438 - INFO - average_duration_days = 353.5581491712707
# INFO:root:KPI globaux exportés vers phase2_global_kpis.csv
# 2025-05-18 10:50:25,440 - INFO - KPI globaux exportés vers phase2_global_kpis.csv
# INFO:root:Vérification temporelle : 16100 = total_ops (16100)
# 2025-05-18 10:50:25,525 - INFO - Vérification temporelle : 16100 = total_ops (16100)
# INFO:root:Vérification CA : 248632732.81 = total_revenue (248632732.80999997)
# 2025-05-18 10:50:25,525 - INFO - Vérification CA : 248632732.81 = total_revenue (248632732.80999997)
# INFO:root:Graphique temporel exporté vers phase2_timeseries.png
# 2025-05-18 10:50:26,242 - INFO - Graphique temporel exporté vers phase2_timeseries.png
# INFO:root:Évolution nb opp par statut exportée vers phase2_timeseries_by_status.png
# 2025-05-18 10:50:26,628 - INFO - Évolution nb opp par statut exportée vers phase2_timeseries_by_status.png
# INFO:root:Évolution CA par entité exportée vers phase2_timeseries_by_entity.png
# 2025-05-18 10:50:27,078 - INFO - Évolution CA par entité exportée vers phase2_timeseries_by_entity.png
# INFO:root:Évolution CA par type d’opportunité exportée vers phase2_timeseries_by_type.png
# 2025-05-18 10:50:27,461 - INFO - Évolution CA par type d’opportunité exportée vers phase2_timeseries_by_type.png
# INFO:root:Répartition par statut exportée vers phase2_status_pipeline.png
# 2025-05-18 10:50:27,915 - INFO - Répartition par statut exportée vers phase2_status_pipeline.png
# INFO:root:Comparatif pipeline normal vs log exporté vers phase2_pipeline_trimestriel_normal_vs_log.png
# 2025-05-18 10:50:29,734 - INFO - Comparatif pipeline normal vs log exporté vers phase2_pipeline_trimestriel_normal_vs_log.png
# INFO:root:Pipeline global empilé exporté vers phase2_pipeline_global_revenue_stacked.png
# 2025-05-18 10:50:30,117 - INFO - Pipeline global empilé exporté vers phase2_pipeline_global_revenue_stacked.png
# INFO:root:Top 10 clients (nombre) exportés vers phase2_top10_clients_count.png
# 2025-05-18 10:50:30,488 - INFO - Top 10 clients (nombre) exportés vers phase2_top10_clients_count.png
# INFO:root:Top 10 clients (CA) exportés vers phase2_top10_clients_revenue.png
# 2025-05-18 10:50:31,171 - INFO - Top 10 clients (CA) exportés vers phase2_top10_clients_revenue.png
# INFO:root:CA par entité exporté vers phase2_revenue_by_entity.png
# 2025-05-18 10:50:31,521 - INFO - CA par entité exporté vers phase2_revenue_by_entity.png
# INFO:root:CA par type d’opportunité exporté vers phase2_revenue_by_type.png
# 2025-05-18 10:50:31,829 - INFO - CA par type d’opportunité exporté vers phase2_revenue_by_type.png
# INFO:root:CA par Commercial responsable exporté vers phase2_revenue_by_commercial.png
# 2025-05-18 10:50:32,275 - INFO - CA par Commercial responsable exporté vers phase2_revenue_by_commercial.png
# INFO:root:Variables exclues de la corrélation (variance nulle) : ['Charge prévisionnelle projet', 'margin_pct']
# 2025-05-18 10:50:32,283 - INFO - Variables exclues de la corrélation (variance nulle) : ['Charge prévisionnelle projet', 'margin_pct']
# INFO:root:Variables retenues pour corrélation : ['Avant_vente', 'Total recette actualisé', 'Total recette réalisé', 'Total recette produit', 'Budget client estimé', 'cycle_vente_jours', 'taux_realisation', 'marge_estimee', 'ecart_budget', 'margin_abs', 'delay_days', 'duration_days']
# 2025-05-18 10:50:32,284 - INFO - Variables retenues pour corrélation : ['Avant_vente', 'Total recette actualisé', 'Total recette réalisé', 'Total recette produit', 'Budget client estimé', 'cycle_vente_jours', 'taux_realisation', 'marge_estimee', 'ecart_budget', 'margin_abs', 'delay_days', 'duration_days']
# INFO:root:Matrice de corrélation exportée vers D:\DATAPREDICT\DATAPREDICT 2024\Missions\Digora\phase2_output\phase2_correlation_matrix.csv
# 2025-05-18 10:50:32,291 - INFO - Matrice de corrélation exportée vers D:\DATAPREDICT\DATAPREDICT 2024\Missions\Digora\phase2_output\phase2_correlation_matrix.csv
# INFO:root:Heatmap de corrélation sauvegardée vers D:\DATAPREDICT\DATAPREDICT 2024\Missions\Digora\phase2_output\phase2_correlation_heatmap.png
# 2025-05-18 10:50:33,158 - INFO - Heatmap de corrélation sauvegardée vers D:\DATAPREDICT\DATAPREDICT 2024\Missions\Digora\phase2_output\phase2_correlation_heatmap.png
# INFO:root:Phase 2 – EDA exploratoire globale terminée (points 2.1 à 2.5).
# 2025-05-18 10:50:33,653 - INFO - Phase 2 – EDA exploratoire globale terminée (points 2.1 à 2.5).
# INFO:root:PDF des figures exporté vers D:\DATAPREDICT\DATAPREDICT 2024\Missions\Digora\phase2_output\phase2_figures.pdf
# 2025-05-18 10:50:41,327 - INFO - PDF des figures exporté vers D:\DATAPREDICT\DATAPREDICT 2024\Missions\Digora\phase2_output\phase2_figures.pdf
# INFO:root:Liste des fichiers exportée vers D:\DATAPREDICT\DATAPREDICT 2024\Missions\Digora\phase2_output\liste_fichiers.txt
# 2025-05-18 10:50:41,327 - INFO - Liste des fichiers exportée vers D:\DATAPREDICT\DATAPREDICT 2024\Missions\Digora\phase2_output\liste_fichiers.txt
#########################################################################

# C:\Users\johan\PycharmProjects\Fissure-master\.venv\lib\site-packages\pandas\core\arraylike.py:396: RuntimeWarning: divide by zero encountered in log10
#   result = getattr(ufunc, method)(*inputs, **kwargs)
# .\tempo.py:805: UserWarning: Glyph 8209 (\N{NON-BREAKING HYPHEN}) missing from current font.
#   plt.tight_layout()
# .\tempo.py:807: UserWarning: Glyph 8209 (\N{NON-BREAKING HYPHEN}) missing from current font.
#   plt.savefig(scatter_path, dpi=300)
# [Focus Budget] Scatter : D:\DATAPREDICT\DATAPREDICT 2024\Missions\Digora\phase2_output\phase2_scatter_budget_vs_real.png
# [Focus Budget] Histogramme : D:\DATAPREDICT\DATAPREDICT 2024\Missions\Digora\phase2_output\phase2_hist_ratio_budget.png
# [Focus Budget] Stats CSV : D:\DATAPREDICT\DATAPREDICT 2024\Missions\Digora\phase2_output\phase2_budget_stats.csv
# .\tempo.py:879: FutureWarning:
#
# The `scale` parameter has been renamed and will be removed in v0.15.0. Pass `density_norm='width'` for the same effect.
#   sns.violinplot(
# [Durées] Box-plot : D:\DATAPREDICT\DATAPREDICT 2024\Missions\Digora\phase2_output\phase2_box_duration_by_status.png
# [Durées] Violin-plot : D:\DATAPREDICT\DATAPREDICT 2024\Missions\Digora\phase2_output\phase2_violin_cycle_by_CA.png

# out_of_bounds_Date_d'enregistrement.csv
# out_of_bounds_Date_de_fin_réelle.csv
# phase2.log
# phase2_business_variables.csv
# phase2_conversion_rate_by_type.png
# phase2_correlation_heatmap.png
# phase2_correlation_matrix.csv
# phase2_data_dictionary.xlsx
# phase2_figures.pdf
# phase2_global_kpis.csv
# phase2_pipeline_global_revenue_stacked.png
# phase2_pipeline_trimestriel_normal_vs_log.png
# phase2_revenue_by_commercial.png
# phase2_revenue_by_entity.png
# phase2_revenue_by_type.png
# phase2_status_pipeline.png
# phase2_timeseries.png
# phase2_timeseries_by_entity.png
# phase2_timeseries_by_status.png
# phase2_timeseries_by_type.png
# phase2_top10_clients_count.png
# phase2_top10_clients_revenue.png

