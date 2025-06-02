import pandas as pd
import numpy as np
import phase4.functions as pf


def generate_data(shift: float = 0.0, n: int = 200, seed: int = 0):
    rng = np.random.default_rng(seed)
    x1 = rng.normal(-1, 0.5, n // 2)
    x2 = rng.normal(1 + shift, 0.5, n // 2)
    x = np.concatenate([x1, x2])
    y = rng.normal(0, 0.2, n)
    cat = ["A"] * (n // 2) + ["B"] * (n // 2)
    dates = pd.date_range("2020-01-01", periods=n, freq="M")
    return pd.DataFrame({"num1": x, "num2": y, "cat": cat, "date": dates})


def test_unsupervised_cv_temporal_keys_and_ranges():
    df = generate_data()
    res = pf.unsupervised_cv_and_temporal_tests(
        df, ["num1", "num2"], ["cat"], n_splits=5
    )

    cv = res["cv_stability"]
    expected_cv_keys = {
        "pca_axis_corr_mean",
        "pca_axis_corr_std",
        "pca_var_first_axis_mean",
        "pca_var_first_axis_std",
        "pca_distance_mse_mean",
        "pca_distance_mse_std",
        "umap_distance_mse_mean",
        "umap_distance_mse_std",
    }
    assert set(cv) == expected_cv_keys
    for val in cv.values():
        assert isinstance(val, float)
        assert not np.isnan(val)

    ts = res["temporal_shift"]
    assert ts is not None
    expected_ts_keys = {
        "pca_axis_corr",
        "pca_distance_mse",
        "pca_mean_shift",
        "umap_distance_mse",
    }
    assert set(ts) == expected_ts_keys
    assert 0 <= ts["pca_axis_corr"] <= 1


def test_unsupervised_cv_temporal_detects_shift():
    df_ref = generate_data(shift=0.0)
    df_shift = generate_data(shift=3.0)

    res_ref = pf.unsupervised_cv_and_temporal_tests(
        df_ref, ["num1", "num2"], ["cat"], n_splits=5
    )
    res_shift = pf.unsupervised_cv_and_temporal_tests(
        df_shift, ["num1", "num2"], ["cat"], n_splits=5
    )

    assert res_shift["temporal_shift"]["pca_mean_shift"] > res_ref["temporal_shift"]["pca_mean_shift"] + 2
