import pandas as pd
import numpy as np

from variable_selection import select_variables


def test_select_variables_basic():
    df = pd.DataFrame({
        "Code": [1, 2, 3],
        "Total recette actualisé": [1000, 2000, 1500],
        "Budget client estimé": [1200, 2100, 1400],
        "Statut commercial": ["Gagné", "Perdu", "Gagné"],
        "Type opportunité": ["T1", "T1", "T1"],
        "Catégorie": ["A", "B", "A"],
        "texte": ["x", "y", "z"],
    })
    for col in ["Statut commercial", "Type opportunité", "Catégorie"]:
        df[col] = df[col].astype("category")

    df_active, quant_vars, qual_vars = select_variables(df, min_modalite_freq=1)

    assert quant_vars == ["Total recette actualisé", "Budget client estimé"]
    assert "Statut commercial" in qual_vars
    assert "Type opportunité" not in qual_vars  # unique modality removed
    assert "Code" not in df_active.columns
    assert np.isclose(df_active[quant_vars].mean().abs().sum(), 0.0)
    assert df_active["Statut commercial"].dtype.name == "category"


def test_select_variables_rare_modalities():
    df = pd.DataFrame({
        "Total recette actualisé": [1, 2, 3, 4, 5],
        "Statut commercial": ["A", "B", "rare", "rare", "B"],
    })
    df["Statut commercial"] = df["Statut commercial"].astype("category")

    df_active, _, qual_vars = select_variables(df, min_modalite_freq=2)

    assert "Statut commercial" in qual_vars
    assert "Autre" in df_active["Statut commercial"].cat.categories
    assert (df_active["Statut commercial"] == "Autre").sum() == 1


def test_select_variables_constant_numeric_and_text():
    df = pd.DataFrame({
        "ID": [1, 2, 3, 4],
        "amount": [10, 20, 30, 40],
        "const": [1, 1, 1, 1],
        "cat": ["a", "b", "a", "b"],
        "notes": ["foo", "bar", "baz", "qux"],
    })
    df["cat"] = df["cat"].astype("category")

    df_active, quant_vars, qual_vars = select_variables(df, min_modalite_freq=1)

    assert quant_vars == ["amount"]
    assert qual_vars == ["cat"]
    assert "const" not in df_active.columns
    assert "notes" not in df_active.columns
    assert df_active["amount"].dtype.kind == "f"
    assert df_active["cat"].dtype.name == "category"
