import pandas as pd
import numpy as np

from nonlinear_methods import run_umap, run_phate, run_pacmap, run_all_nonlinear


def sample_df() -> pd.DataFrame:
    return pd.DataFrame({
        "num1": [1, 2, 3, 4, 5],
        "num2": [5, 4, 3, 2, 1],
        "cat": ["a", "b", "a", "b", "a"],
    })


def test_run_umap_basic():
    df = sample_df()
    res = run_umap(df, n_components=2, n_neighbors=3, min_dist=0.1, random_state=0)
    assert res["embeddings"].shape == (len(df), 2)
    assert res["params"]["n_neighbors"] == 3
    assert res["runtime_s"] > 0
    assert np.var(res["embeddings"].values) > 0


def test_run_umap_missing_lib(monkeypatch):
    import nonlinear_methods as nl
    monkeypatch.setattr(nl, "umap", None)
    res = nl.run_umap(sample_df())
    assert res["model"] is None


def test_run_phate_basic():
    df = sample_df()
    res = run_phate(df, n_components=2, k=3, a=10, random_state=0)
    if res["model"] is None:
        assert res["embeddings"].empty
    else:
        assert res["embeddings"].shape == (len(df), 2)
        assert res["runtime_s"] > 0


def test_run_pacmap_basic():
    df = sample_df()
    import types
    import nonlinear_methods as nl
    if nl.pacmap is None:
        res = nl.run_pacmap(df)
        assert res["model"] is None
        return
    # Avoid heavy numba compilation by mocking the pacmap dependency
    original = nl.pacmap
    try:
        dummy = types.SimpleNamespace(
            PaCMAP=lambda **kwargs: types.SimpleNamespace(
                fit_transform=lambda X: np.zeros((len(X), 2))
            )
        )
        nl.pacmap = dummy
        res = nl.run_pacmap(df, n_components=2, n_neighbors=3, random_state=0)
        assert res["embeddings"].shape == (len(df), 2)
        assert res["runtime_s"] >= 0
    finally:
        nl.pacmap = original


def test_run_all_nonlinear():
    df = sample_df()
    results = run_all_nonlinear(df)
    assert set(results.keys()) >= {"umap", "phate", "pacmap"}
