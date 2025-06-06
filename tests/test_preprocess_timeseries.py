import pandas as pd
import numpy as np
import pytest

from pred_aggregated_amount.preprocess_timeseries import (
    load_and_aggregate,
    preprocess_series,
    preprocess_all,
)


def test_load_and_aggregate_basic(tmp_path):
    df = pd.DataFrame(
        {
            "Date": [
                "10/01/2020",
                "20/01/2020",
                "10/02/2020",
                "15/03/2020",
            ],
            "Status": ["won", "lost", "won", "won"],
            "Amount": [100.0, 200.0, 300.0, 400.0],
        }
    )
    csv_path = tmp_path / "data.csv"
    df.to_csv(csv_path, index=False)

    cfg = {
        "csv_path": str(csv_path),
        "date_col": "Date",
        "status_col": "Status",
        "won_value": "won",
        "amount_col": "Amount",
    }

    monthly, quarterly, yearly = load_and_aggregate(cfg)

    assert isinstance(monthly, pd.Series)
    assert isinstance(quarterly, pd.Series)
    assert isinstance(yearly, pd.Series)
    assert monthly.sum() == pytest.approx(df[df["Status"] == "won"]["Amount"].sum())
    assert monthly.index.freqstr in {"M", "ME"}
    assert quarterly.index.freqstr.startswith("Q")
    assert yearly.index.freqstr.startswith("A") or yearly.index.freqstr.startswith("YE")


def test_load_and_aggregate_file_not_found():
    cfg = {
        "csv_path": "does_not_exist.csv",
        "date_col": "Date",
        "status_col": "Status",
        "won_value": "won",
        "amount_col": "Amount",
    }
    with pytest.raises(FileNotFoundError):
        load_and_aggregate(cfg)


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
