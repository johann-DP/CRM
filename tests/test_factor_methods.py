import pandas as pd
import numpy as np

import phase4_functions as pf


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


def test_run_mfa_basic():
    df = sample_df()
    groups = [["num1", "num2"], ["cat1", "cat2"]]
    res = pf.run_mfa(df, groups, optimize=True)
    assert res["embeddings"].shape[0] == len(df)
    assert len(res["inertia"]) == res["embeddings"].shape[1]

