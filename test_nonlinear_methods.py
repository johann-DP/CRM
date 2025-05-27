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
    res = run_umap(df, n_components=2, n_neighbors=3, min_dist=0.1)
    assert res["embeddings"].shape == (len(df), 2)
    assert res["params"]["n_neighbors"] == 3


def test_run_phate_basic():
    df = sample_df()
    res = run_phate(df, n_components=2, k=3, a=10)
    assert res["embeddings"].shape[0] == len(df)
    assert res["embeddings"].shape[1] == 2


def test_run_pacmap_basic():
    df = sample_df()
    # Avoid heavy numba compilation by mocking the pacmap dependency
    import types
    import nonlinear_methods as nl
    original = nl.pacmap
    try:
        dummy = types.SimpleNamespace(
            PaCMAP=lambda **kwargs: types.SimpleNamespace(
                fit_transform=lambda X: np.zeros((len(X), 2))
            )
        )
        nl.pacmap = dummy
        res = nl.run_pacmap(df, n_components=2, n_neighbors=3)
        assert res["embeddings"].shape == (len(df), 2)
    finally:
        nl.pacmap = original


def test_run_all_nonlinear():
    df = sample_df()
    results = run_all_nonlinear(df)
    assert set(results.keys()) >= {"umap", "phate", "pacmap"}
