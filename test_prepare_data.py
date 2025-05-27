import importlib
from pathlib import Path

import numpy as np
import pandas as pd


def test_prepare_data_basic():
    mod = importlib.import_module("data_preparation")

    df = pd.DataFrame({
        "Code": [1, 2, 2, 3],
        "Date de début actualisée": ["2024-01-01", "2024-01-05", "2024-01-05", "1980-01-01"],
        "Date de fin réelle": ["2024-01-10", "2024-01-15", "2024-01-15", "2051-01-01"],
        "Total recette réalisé": ["1000", "-50", "1500", "2000"],
        "Budget client estimé": ["1000", "0", "1600", "2000"],
        "Charge prévisionnelle projet": [800, 800, 800, 800],
        "Statut commercial": ["Gagné", "Perdu", "Annulé", "Gagné"],
        "flag_multivariate": [False, True, False, False],
    })

    original = df.copy(deep=True)
    cleaned = mod.prepare_data(df, exclude_lost=True)

    # original DataFrame should remain unchanged
    pd.testing.assert_frame_equal(df, original)

    # lost / flagged rows removed -> keep codes 1 and 3
    assert list(cleaned["Code"]) == [1, 3]

    # numeric columns are scaled to mean ~0
    num_cols = [c for c in cleaned.select_dtypes(include=np.number).columns if c != "Code"]
    means = cleaned[num_cols].mean()
    assert np.allclose(means, 0, atol=1e-6)
    stds = cleaned[num_cols].std(ddof=0)
    non_constant = cleaned[num_cols].nunique() > 1
    for col in num_cols:
        if non_constant[col]:
            assert np.isclose(stds[col], 1, atol=1e-6)
        else:
            assert np.isclose(stds[col], 0, atol=1e-6)

    # derived columns exist
    assert {"duree_projet_jours", "taux_realisation", "marge_estimee"} <= set(cleaned.columns)


def test_prepare_data_flag_file(tmp_path: Path):
    mod = importlib.import_module("data_preparation")

    df = pd.DataFrame(
        {
            "Code": [1, 2, 3],
            "Date de début actualisée": ["2024-01-01"] * 3,
            "Date de fin réelle": ["2024-01-02"] * 3,
            "Total recette réalisé": [10, 20, 30],
            "Budget client estimé": [10, 20, 30],
        }
    )

    flagged = pd.DataFrame({"Code": [2]})
    flagged_path = tmp_path / "dataset_phase3_flagged.csv"
    flagged.to_csv(flagged_path, index=False)

    cleaned = mod.prepare_data(df, exclude_lost=False, flagged_ids_path=flagged_path)

    assert list(cleaned["Code"]) == [1, 3]

def test_prepare_data_imputes_missing_values():
    mod = importlib.import_module("data_preparation")

    df = pd.DataFrame(
        {
            "Code": [1, 2, 3],
            "Date de début actualisée": ["2024-01-01"] * 3,
            "Date de fin réelle": ["2024-01-02"] * 3,
            "Total recette réalisé": [np.nan, "-5", "100"],
            "Budget client estimé": [np.nan, 40, 60],
            "Statut commercial": ["Gagné", "Gagné", "Gagné"],
        }
    )

    cleaned = mod.prepare_data(df, exclude_lost=False)

    num_cols = cleaned.select_dtypes(include=np.number).columns
    assert not cleaned[num_cols].isna().any().any()

    means = cleaned[num_cols.difference(["Code"])].mean()
    assert np.allclose(means, 0, atol=1e-6)


def test_prepare_data_keep_lost():
    mod = importlib.import_module("data_preparation")

    df = pd.DataFrame(
        {
            "Code": [1, 2, 3],
            "Date de début actualisée": ["2024-01-01"] * 3,
            "Date de fin réelle": ["2024-01-02"] * 3,
            "Total recette réalisé": [10, 20, 30],
            "Budget client estimé": [10, 20, 30],
            "Statut commercial": ["Gagné", "Perdu", "Annulé"],
        }
    )

    cleaned = mod.prepare_data(df, exclude_lost=False)

    assert list(cleaned["Code"]) == [1, 2, 3]

