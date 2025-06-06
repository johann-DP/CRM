import pandas as pd
from pathlib import Path
from phase4bis.export_famd_inertias import export_famd_inertias


def test_export_famd_inertias(tmp_path: Path):
    df = pd.DataFrame({
        "Code": [1, 2, 3, 4],
        "Date de début actualisée": ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"],
        "Date de fin réelle": ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"],
        "Total recette réalisé": [10, 20, 30, 40],
        "Budget client estimé": [12, 22, 32, 42],
        "Categorie": ["A", "B", "A", "B"],
        "Statut commercial": ["Gagné", "Gagné", "Gagné", "Gagné"],
    })
    csv_path = tmp_path / "raw.csv"
    df.to_csv(csv_path, index=False)

    cfg = {
        "input_file": str(csv_path),
        "dataset": "raw",
        "min_modalite_freq": 1,
    }
    out = tmp_path / "inerties.csv"

    table = export_famd_inertias(cfg, output=out)
    assert out.exists()
    loaded = pd.read_csv(out)
    assert set(["dimension", "valeur_propre", "variance_expliquee_pct", "variance_cumulee_pct"]).issubset(loaded.columns)
    assert loaded["variance_cumulee_pct"].iloc[-1] >= 95
    assert len(loaded) == len(table)
