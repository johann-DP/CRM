import pandas as pd
import numpy as np
import pytest

from factor_methods import run_pca, run_mca, run_famd, run_mfa


def sample_df():
    return pd.DataFrame(
        {
            "num1": [1, 2, 3, 4, 5],
            "num2": [5, 4, 3, 2, 1],
            "cat1": ["a", "b", "a", "b", "a"],
            "cat2": ["x", "y", "x", "y", "x"],
        }
    )


def test_run_pca_basic():
    df = sample_df()
    res = run_pca(df, ["num1", "num2"], optimize=True, variance_threshold=0.8)
    assert set(res.keys()) >= {
        "model",
        "inertia",
        "embeddings",
        "loadings",
        "runtime_s",
    }
    assert isinstance(res["embeddings"], pd.DataFrame)
    assert res["embeddings"].shape[0] == len(df)
    assert len(res["inertia"]) == res["embeddings"].shape[1]


def test_run_pca_autocomponents():
    df = pd.DataFrame(
        {
            "x": [1, 2, 3, 4, 5],
            "y": [10, 11, 9, 12, 8],
            "z": [0.01, 0.02, 0.03, 0.01, 0.02],
        }
    )
    res = run_pca(df, ["x", "y", "z"], optimize=True, variance_threshold=0.8)
    # expect two components kept
    assert res["model"].n_components_ <= 3
    assert len(res["inertia"]) == res["model"].n_components_
    assert res["model"].n_components_ == 2
    assert "coords" in res and "explained_variance_ratio" in res and "runtime" in res


def test_run_mca_basic():
    df = sample_df()
    res = run_mca(df, ["cat1", "cat2"], optimize=True)
    assert res["embeddings"].shape[0] == len(df)
    assert len(res["inertia"]) == res["embeddings"].shape[1]


def test_run_mca_inertia_sum():
    df = pd.DataFrame(
        {
            "a": list("abcabcabca"),
            "b": list("xyzxyzxyzx"),
        }
    )
    res = run_mca(df, ["a", "b"], optimize=True)
    assert pytest.approx(res["inertia"].sum(), rel=1e-6) == 1.0
    assert res["embeddings"].shape[0] == len(df)


def test_run_famd_basic():
    df = sample_df()
    res = run_famd(df, ["num1", "num2"], ["cat1", "cat2"], optimize=True)
    assert res["embeddings"].shape[0] == len(df)
    assert len(res["inertia"]) == res["embeddings"].shape[1]


def test_run_famd_axes():
    df = pd.DataFrame(
        {
            "num1": [1.0, 2.0, 3.0, 4.0, 5.0],
            "num2": [2.0, 1.5, 3.5, 2.2, 5.1],
            "cat": ["a", "b", "a", "b", "a"],
        }
    )
    res = run_famd(df, ["num1", "num2"], ["cat"], optimize=True)
    assert res["model"].n_components <= 3
    inertias = res["inertia"]
    assert np.all(np.diff(inertias.cumsum()) >= -1e-6)


def test_run_mfa_basic():
    df = sample_df()
    groups = [["num1", "num2"], ["cat1", "cat2"]]
    res = run_mfa(df, groups, optimize=True)
    assert res["embeddings"].shape[0] == len(df)
    assert len(res["inertia"]) == res["embeddings"].shape[1]


def test_run_mfa_groups():
    df = sample_df()
    groups = [["num1"], ["num2", "cat1"], ["cat2"]]
    res = run_mfa(df, groups, optimize=True)
    assert set(["runtime", "coords"]).issubset(res.keys())
    assert res["embeddings"].shape[0] == len(df)


def test_run_famd_extra_kwargs():
    df = sample_df()
    res = run_famd(
        df,
        ["num1", "num2"],
        ["cat1", "cat2"],
        optimize=True,
        weighting="balanced",
        n_components_rule="elbow",
    )
    assert res["embeddings"].shape[0] == len(df)
