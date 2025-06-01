import builtins
import numpy as np
import pandas as pd
import phase4_functions as pf


def sample_df():
    df = pd.DataFrame({
        "num1": [1, 2, 3, 4, 5],
        "num2": [5, 4, 3, 2, 1],
        "cat1": ["a", "b", "a", "b", "a"],
        "cat2": ["x", "y", "x", "y", "x"],
    })
    df["cat1"] = df["cat1"].astype("category")
    df["cat2"] = df["cat2"].astype("category")
    return df


def assert_same(a, b):
    assert np.allclose(a.values, b.values)


def test_run_umap_basic():
    df = sample_df()
    res = pf.run_umap(df)
    assert res["embeddings"].shape == (len(df), 2)


def test_run_phate_basic():
    df = sample_df()
    res = pf.run_phate(df)
    assert res["embeddings"].shape[0] == len(df)
    if res["model"] is not None:
        assert res["embeddings"].shape[1] == 2
    else:
        assert res["embeddings"].empty


def test_run_phate_knn_alias():
    df = sample_df()
    res_knn = pf.run_phate(df, knn=3)
    assert res_knn["params"]["k"] == 3


def test_run_phate_decay_alias():
    df = sample_df()
    res_decay = pf.run_phate(df, decay=5)
    assert res_decay["params"]["a"] == 5


def test_run_phate_knn_auto():
    df = sample_df()
    res = pf.run_phate(df, knn="auto")
    # invalid string should fall back to default (15 from BEST_PARAMS)
    assert res["params"]["k"] == 15


def test_run_pacmap_basic():
    df = sample_df()
    res = pf.run_pacmap(df)
    assert res["embeddings"].shape[0] == len(df)
    if res["model"] is not None:
        assert res["embeddings"].shape[1] == 2
    else:
        assert res["embeddings"].empty


def test_run_pacmap_missing(monkeypatch):
    df = sample_df()
    monkeypatch.setattr(pf, "pacmap", None)

    orig_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "pacmap":
            raise ImportError("mock")
        return orig_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    res = pf.run_pacmap(df)
    assert res["model"] is None
    assert res["embeddings"].empty


def test_run_tsne_basic():
    df = sample_df()
    res = pf.run_tsne(df)
    assert res["embeddings"].shape == (len(df), 2)
