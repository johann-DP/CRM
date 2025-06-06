import pandas as pd
import importlib

ar = importlib.import_module("pred_aggregated_amount.aggregate_revenue")


def test_load_won_opportunities_filters_and_parses(tmp_path):
    df = pd.DataFrame({
        "Date de fin actualisée": ["2024-01-10", "2024-01-20", "2024-02-10", "2024-03-05"],
        "Statut commercial": ["Gagné", "Perdu", "Won", "Annulé"],
        "Total recette réalisé": [100, 200, 300, 400],
    })
    csv = tmp_path / "data.csv"
    df.to_csv(csv, index=False)

    out = ar.load_won_opportunities(csv)

    assert list(out.columns) == ["Total recette réalisé"]
    assert isinstance(out.index, pd.DatetimeIndex)
    assert list(out.index) == [pd.Timestamp("2024-01-10"), pd.Timestamp("2024-02-10")]
    assert list(out["Total recette réalisé"]) == [100, 300]


def test_aggregate_revenue_basic():
    idx = pd.to_datetime(["2024-01-10", "2024-01-20", "2024-03-15"])
    df = pd.DataFrame({"Total": [100, 200, 300]}, index=idx)

    monthly, quarterly, yearly = ar.aggregate_revenue(df, "Total")

    exp_month_idx = pd.to_datetime(["2024-01-31", "2024-02-29", "2024-03-31"])
    exp_month = pd.Series([300, 0, 300], index=exp_month_idx, name="Total")
    pd.testing.assert_series_equal(monthly, exp_month, check_freq=False)

    exp_quarter_idx = pd.to_datetime(["2024-03-31"])
    exp_quarter = pd.Series([600], index=exp_quarter_idx, name="Total")
    pd.testing.assert_series_equal(quarterly, exp_quarter, check_freq=False)

    exp_year_idx = pd.to_datetime(["2024-12-31"])
    exp_year = pd.Series([600], index=exp_year_idx, name="Total")
    pd.testing.assert_series_equal(yearly, exp_year, check_freq=False)


def test_build_timeseries_calls(monkeypatch):
    data = pd.DataFrame({"Total": [1]}, index=pd.to_datetime(["2024-01-10"]))
    called = {}

    def fake_load(path, **kwargs):
        called["load"] = True
        return data

    def fake_agg(df, amount_col):
        called["agg"] = df is data
        s = pd.Series([1], index=pd.to_datetime(["2024-01-31"]), name=amount_col)
        return s, s, s

    monkeypatch.setattr(ar, "load_won_opportunities", fake_load)
    monkeypatch.setattr(ar, "aggregate_revenue", fake_agg)

    res = ar.build_timeseries("dummy.csv", amount_col="Total")

    assert called.get("load")
    assert called.get("agg")
    for series in res:
        pd.testing.assert_series_equal(
            series,
            pd.Series([1], index=pd.to_datetime(["2024-01-31"]), name="Total"),
            check_freq=False,
        )

