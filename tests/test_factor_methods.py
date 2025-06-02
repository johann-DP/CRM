import pandas as pd
import numpy as np

import phase4.functions as pf


def sample_df():
    return pd.DataFrame({
        "num1": [1, 2, 3, 4, 5],
        "num2": [5, 4, 3, 2, 1],
        "cat1": ["a", "b", "a", "b", "a"],
        "cat2": ["x", "y", "x", "y", "x"],
    })


def test_run_pca_basic():
    df = sample_df()
    res = pf.run_pca(df, ["num1", "num2"], optimize=True)
    assert res["embeddings"].shape[0] == len(df)
    assert len(res["inertia"]) == res["embeddings"].shape[1]
    assert np.isclose(res["inertia"].sum(), 1.0)


def test_run_mca_basic():
    df = sample_df()
    res = pf.run_mca(df, ["cat1", "cat2"], optimize=True)
    assert res["embeddings"].shape[0] == len(df)
    assert len(res["inertia"]) == res["embeddings"].shape[1]
    assert np.isclose(res["inertia"].sum(), 1.0)


def test_run_famd_basic():
    df = sample_df()
    res = pf.run_famd(df, ["num1", "num2"], ["cat1", "cat2"], optimize=True)
    assert res["embeddings"].shape[0] == len(df)
    assert len(res["inertia"]) == res["embeddings"].shape[1]
    assert np.isclose(res["inertia"].sum(), 1.0)


def test_run_mfa_basic():
    df = sample_df()
    groups = [["num1", "num2"], ["cat1", "cat2"]]
    res = pf.run_mfa(df, groups, optimize=True)
    assert res["embeddings"].shape[0] == len(df)
    assert len(res["inertia"]) == res["embeddings"].shape[1]
    assert np.isclose(res["inertia"].sum(), 1.0)


def test_run_mfa_with_weights():
    df = sample_df()
    groups = {"Num": ["num1", "num2"], "Cat": ["cat1", "cat2"]}
    res = pf.run_mfa(df, groups, optimize=True, weights={"Num": 2.0, "Cat": 1.0})
    assert res["embeddings"].shape[0] == len(df)


def test_pca_contributions():
    df = sample_df()
    res = pf.run_pca(df, ["num1", "num2"], n_components=2)
    var_contrib = pf.pca_variable_contributions(res["loadings"])
    assert np.allclose(var_contrib[["F1", "F2"]].sum().values, [100.0, 100.0])
    ind_contrib = pf.pca_individual_contributions(res["embeddings"])
    total = ind_contrib.sum(axis=1, skipna=False).dropna()
    assert np.allclose(total.values, 100.0)

def test_mfa_group_contributions():
    df = sample_df()
    groups = {"Num": ["num1", "num2"], "Cat": ["cat1", "cat2"]}
    res = pf.run_mfa(df, groups, n_components=2)
    contrib = pf.mfa_group_contributions(res["model"])
    assert set(groups).issubset(contrib.columns)
    sums = contrib[list(groups.keys())].sum(axis=1)
    assert np.allclose(sums.values, 100.0)
    assert list(contrib.index)[0] == "F1"
