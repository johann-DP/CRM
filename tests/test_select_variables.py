import pandas as pd
import numpy as np

import phase4_functions as pf


def test_select_variables_basic():
    df = pd.DataFrame({
        "ID": [1, 2, 3],
        "Total": [1000, 2000, 1500],
        "Budget": [1200, 2100, 1400],
        "Industry": ["Tech", "Finance", "Tech"],
        "Type opportunité": ["T1", "T1", "T1"],
        "texte": ["foo", "bar", "baz"],
    })
    df["Industry"] = df["Industry"].astype("category")
    df["Type opportunité"] = df["Type opportunité"].astype("category")

    df_active, quant_vars, qual_vars = pf.select_variables(df, min_modalite_freq=1)

    assert quant_vars == ["Total", "Budget"]
    assert "Industry" in qual_vars
    assert "Type opportunité" not in qual_vars  # unique modality removed
    assert "ID" not in df_active.columns
    assert np.isclose(df_active[quant_vars].mean().abs().sum(), 0.0)
    assert df_active["Industry"].dtype.name == "category"


def test_select_variables_rare_modalities():
    df = pd.DataFrame({
        "amount": [1, 2, 3, 4, 5],
        "Industry": ["A", "B", "rare", "rare", "B"],
    })
    df["Industry"] = df["Industry"].astype("category")

    df_active, _, qual_vars = pf.select_variables(df, min_modalite_freq=2)

    assert "Industry" in qual_vars
    assert "Autre" in df_active["Industry"].cat.categories
    assert (df_active["Industry"] == "Autre").sum() == 1


def test_select_variables_constant_numeric_and_text():
    df = pd.DataFrame({
        "ID": [1, 2, 3, 4],
        "amount": [10, 20, 30, 40],
        "const": [1, 1, 1, 1],
        "cat": ["a", "b", "a", "b"],
        "notes": ["foo", "bar", "baz", "qux"],
    })
    df["cat"] = df["cat"].astype("category")

    df_active, quant_vars, qual_vars = pf.select_variables(df, min_modalite_freq=1)

    assert quant_vars == ["amount"]
    assert qual_vars == ["cat"]
    assert "const" not in df_active.columns
    assert "notes" not in df_active.columns
    assert df_active["amount"].dtype.kind == "f"
    assert df_active["cat"].dtype.name == "category"
