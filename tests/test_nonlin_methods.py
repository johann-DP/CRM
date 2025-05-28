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


def test_run_umap_reproducible():
    df = sample_df()
    res1 = pf.run_umap(df, random_state=42)
    res2 = pf.run_umap(df, random_state=42)
    assert res1["embeddings"].shape == (len(df), 2)
    assert_same(res1["embeddings"], res2["embeddings"])


def test_run_phate_reproducible():
    df = sample_df()
    res1 = pf.run_phate(df, random_state=42)
    res2 = pf.run_phate(df, random_state=42)
    assert res1["embeddings"].shape[0] == len(df)
    if res1["model"] is not None:
        assert res1["embeddings"].shape[1] == 2
        assert_same(res1["embeddings"], res2["embeddings"])
    else:
        assert res1["embeddings"].empty


def test_run_pacmap_reproducible():
    df = sample_df()
    res1 = pf.run_pacmap(df, random_state=42)
    res2 = pf.run_pacmap(df, random_state=42)
    assert res1["embeddings"].shape[0] == len(df)
    if res1["model"] is not None:
        assert res1["embeddings"].shape[1] == 2
        assert_same(res1["embeddings"], res2["embeddings"])
    else:
        assert res1["embeddings"].empty


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

