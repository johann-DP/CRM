import pandas as pd
from pred_aggregated_amount.external_data import align_exogenous, merge_target_exog
from pred_aggregated_amount.features_utils import make_lag_features


def test_align_exogenous_basic():
    target = pd.Series(
        [1, 2, 3], index=pd.date_range("2020-01-01", periods=3, freq="M")
    )
    ex1 = pd.Series(
        [10, 20, 30], index=pd.date_range("2020-01-01", periods=3, freq="M")
    )
    df = align_exogenous(target, {"ex": ex1})
    assert list(df.columns) == ["ex"]
    pd.testing.assert_index_equal(df.index, target.index)


def test_merge_target_exog_and_supervised():
    target = pd.Series(
        [1, 2, 3, 4, 5], index=pd.date_range("2020-01-01", periods=5, freq="M")
    )
    ex = pd.Series(
        [0, 1, 0, 1, 0], index=pd.date_range("2020-01-01", periods=5, freq="M")
    )
    df = merge_target_exog(target, {"flag": ex})
    df_sup = make_lag_features(df["y"], 2, "M", False).join(df[["flag"]])
    X = df_sup.drop(columns=["y"])
    y = df_sup["y"]
    assert "flag" in X.columns
    assert len(X) == len(y)
