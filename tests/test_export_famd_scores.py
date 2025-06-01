import pandas as pd
from pathlib import Path
from export_famd_scores import export_famd_scores


def test_export_famd_scores(tmp_path: Path):
    df = pd.DataFrame({
        "Code": [1, 2, 3],
        "Date de début actualisée": ["2024-01-01", "2024-01-02", "2024-01-03"],
        "Date de fin réelle": ["2024-01-02", "2024-01-03", "2024-01-04"],
        "Total recette réalisé": [10, 20, 30],
        "Budget client estimé": [12, 22, 32],
        "Categorie": ["A", "B", "A"],
        "Statut commercial": ["Gagné", "Gagné", "Gagné"],
    })
    csv_path = tmp_path / "raw.csv"
    df.to_csv(csv_path, index=False)

    cfg = {
        "input_file": str(csv_path),
        "dataset": "raw",
        "min_modalite_freq": 1,
        "famd": {"n_components": 3},
    }
    out = tmp_path / "coords.csv"

    coords = export_famd_scores(cfg, output=out)
    assert out.exists()
    loaded = pd.read_csv(out)
    assert list(loaded.columns)[0] == "ID"
    assert "Dim1_FAMD" in loaded.columns
    assert len(loaded) == len(coords) == len(df)
