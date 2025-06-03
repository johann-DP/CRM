from pred_aggregated_amount.compare_granularities import build_performance_table


def test_build_performance_table_basic():
    results = {
        "ARIMA": {
            "monthly": {"MAE": 1.0, "RMSE": 2.0, "MAPE": 3.0},
            "quarterly": {"MAE": 0.5, "RMSE": 1.5, "MAPE": 2.5},
            "yearly": {"MAE": 2.0, "RMSE": 3.0, "MAPE": 4.0},
        },
        "Prophet": {
            "monthly": {"MAE": 1.1, "RMSE": 2.1, "MAPE": 3.1},
            "quarterly": {"MAE": 0.6, "RMSE": 1.6, "MAPE": 2.6},
            "yearly": {"MAE": 2.1, "RMSE": 3.1, "MAPE": 4.1},
        },
    }

    table = build_performance_table(results)
    assert list(table.index) == ["ARIMA", "Prophet"]
    expected_cols = {
        "MAE_monthly",
        "RMSE_monthly",
        "MAPE_monthly",
        "MAE_quarterly",
        "RMSE_quarterly",
        "MAPE_quarterly",
        "MAE_yearly",
        "RMSE_yearly",
        "MAPE_yearly",
    }
    assert set(table.columns) == expected_cols
    assert table.loc["ARIMA", "MAE_monthly"] == 1.0
    assert table.loc["Prophet", "MAPE_yearly"] == 4.1
