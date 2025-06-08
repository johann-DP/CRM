import pandas as pd
import numpy as np
import pytest

from pred_aggregated_amount.preprocess_timeseries import (
    preprocess_series,
    preprocess_all,
)



def test_preprocess_series_interpolate_and_clip():
    index = pd.to_datetime([
        "2020-01-31",
        "2020-03-31",
        "2020-04-30",
        "2020-06-30",
    ])
    s = pd.Series([np.nan, 5, 1, 100], index=index)

    cleaned = preprocess_series(s, freq="ME")

    # rebuild expected steps to compute clipping threshold
    tmp = s.asfreq("ME")
    tmp = tmp.interpolate(method="linear").fillna(0)
    q1 = tmp.quantile(0.25)
    q3 = tmp.quantile(0.75)
    iqr = q3 - q1
    upper = q3 + 1.5 * iqr

    assert cleaned.index.freqstr == "ME"
    assert not cleaned.isna().any()
    assert cleaned.iloc[0] == 0
    assert cleaned.iloc[-1] == pytest.approx(upper)


def test_preprocess_all_frequencies():
    monthly = pd.Series([1, 2], index=pd.date_range("2020-01-31", periods=2, freq="M"))
    quarterly = pd.Series([1, 2], index=pd.date_range("2020-03-31", periods=2, freq="Q"))
    yearly = pd.Series([1, 2], index=pd.date_range("2020-12-31", periods=2, freq="A"))

    m, q, y = preprocess_all(monthly, quarterly, yearly)

    assert m.index.freqstr == "ME"
    assert q.index.freqstr.startswith("QE")
    assert y.index.freqstr.startswith("YE")
