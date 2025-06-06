import os
import pandas as pd
import yaml

# 1. Trouver le fichier d’entrée de la phase 3 (jeu de données final)
input_file = None
phase3_list_path = r"\\Lenovo\d\DATAPREDICT\DATAPREDICT 2024\Missions\Digora\phase3_output\phase3_output_files.txt"
if os.path.exists(phase3_list_path):
    with open(phase3_list_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    # Recherche d’un fichier contenant "cleaned_all" (toutes variables)
    for line in lines:
        if "phase3_cleaned_all.csv" in line:
            input_file = line
            break
    # À défaut, on prend le premier fichier listé comme candidat
    if input_file is None and lines:
        input_file = lines[0]

# Si le fichier n’a pas été trouvé via la liste, on essaie un nom par défaut
if input_file is None:
    input_file = r"\\Lenovo\d\DATAPREDICT\DATAPREDICT 2024\Missions\Digora\phase3_output\phase3_cleaned_all.csv"

# 2. Identifier la colonne cible dans le fichier de données
target_col = None
if os.path.exists(input_file):
    # Lire uniquement les premières lignes pour analyser les colonnes sans tout charger
    df_sample = pd.read_csv(input_file, nrows=500)
    # Liste des colonnes candidates (ayant peu de valeurs uniques, possiblement binaires)
    binary_candidates = []
    for col in df_sample.columns:
        # On collecte les valeurs uniques non nulles de la colonne échantillon
        values = df_sample[col].dropna().unique()
        if len(values) <= 2:
            binary_candidates.append(col)
            # Vérification par mot-clé sur le nom de colonne
            col_lower = col.lower()
            if any(kw in col_lower for kw in ["Gagné", "convert", "opportun", "cible", "target"]):
                target_col = col
                break
    # Si aucun mot-clé n’a matché mais qu’on a des candidats binaires
    if target_col is None and binary_candidates:
        # Si une seule colonne binaire existe, on la prend
        if len(binary_candidates) == 1:
            target_col = binary_candidates[0]
        else:
            # Sinon, on peut appliquer un heuristique : par exemple la colonne binaire la plus déséquilibrée
            # (car le taux de conversion est souvent faible)
            imbalance_scores = {}
            for col in binary_candidates:
                # Calcul du ratio de 1 (ou True) si la colonne est numérique ou booléenne
                try:
                    ratio = df_sample[col].mean()  # proportion de 1 si 0/1
                except Exception:
                    # Si ce n’est pas numérique (ex: "Oui"/"Non"), on convertit temporairement
                    ratio = (df_sample[col].astype(str).str.lower().isin(["1", "true", "oui", "yes", "Gagné"])).mean()
                imbalance_scores[col] = abs(ratio - 0.5)
            # Choisir la colonne dont le ratio s’éloigne le plus de 0.5 (donc très déséquilibré)
            target_col = max(imbalance_scores, key=imbalance_scores.get)

# 3. Définir le dossier de sortie pour les résultats du lead scoring
output_dir = "lead_scoring_output"
# (On peut ajuster ce chemin en fonction de la structure du projet,
# par exemple 'results/lead_scoring' ou un dossier phase4 dédié.)

# 4. Construire le bloc de configuration lead_scoring
lead_scoring_block = {
    "lead_scoring": {
        "input_file": input_file,
        "target_col": target_col if target_col else "<nom_colonne_cible>",
        "output_dir": output_dir
    }
}

# 5. Afficher le bloc YAML formaté
print(yaml.dump(lead_scoring_block, sort_keys=False, allow_unicode=True))
