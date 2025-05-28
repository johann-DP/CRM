import sys
import types

import numpy as np
import pandas as pd
import pytest

from unsupervised_cv import unsupervised_cv_and_temporal_tests
import math


def sample_df() -> pd.DataFrame:
    return pd.DataFrame({
        "num1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "num2": [10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
        "cat": ["a", "b"] * 5,
        "date": pd.date_range("2020-01-01", periods=10, freq="M"),
    })


def dummy_umap_module():
    def factory(**kwargs):
        n = kwargs.get("n_components", 2)
        return types.SimpleNamespace(
            fit=lambda X: None,
            fit_transform=lambda X: np.vstack([np.arange(len(X)), np.zeros(len(X))]).T[:, :n],
            transform=lambda X: np.vstack([np.arange(len(X)), np.zeros(len(X))]).T[:, :n],
        )
    return types.SimpleNamespace(UMAP=factory)


@pytest.fixture(autouse=True)
def patch_umap(monkeypatch):
    mod = dummy_umap_module()
    monkeypatch.setitem(sys.modules, "umap", mod)
    yield
    monkeypatch.setitem(sys.modules, "umap", mod)


def test_unsupervised_cv_basic():
    df = sample_df()
    res = unsupervised_cv_and_temporal_tests(df, ["num1", "num2"], ["cat"], n_splits=3)
    assert set(res) == {"cv_stability", "temporal_shift"}
    assert "pca_axis_corr_mean" in res["cv_stability"]
    assert isinstance(res["temporal_shift"], dict)


def test_unsupervised_cv_skip_cv():
    df = sample_df()
    res = unsupervised_cv_and_temporal_tests(df, ["num1", "num2"], ["cat"], n_splits=1)
    assert math.isnan(res["cv_stability"]["pca_axis_corr_mean"])
    assert res["temporal_shift"] is not None


def test_unsupervised_cv_no_date():
    df = sample_df().drop(columns=["date"])
    res = unsupervised_cv_and_temporal_tests(df, ["num1", "num2"], ["cat"], n_splits=2)
    assert res["temporal_shift"] is None
