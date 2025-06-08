import pandas as pd
import numpy as np
from pathlib import Path
import pytest

from pred_aggregated_amount import evaluate_models as em
from pred_aggregated_amount import future_forecast as ff
from pred_aggregated_amount import catboost_forecast as cb
from pred_aggregated_amount import make_plots


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def test_safe_mape_ignores_zero():
    y_true = np.array([10, 0, 20])
    y_pred = np.array([12, 0, 18])
    mape = em.safe_mape(y_true, y_pred)
    assert mape == pytest.approx(15.0)


def test_compute_metrics_basic():
    true = [1, 2, 3]
    pred = [1, 2, 4]
    metrics = em._compute_metrics(true, pred)
    assert set(metrics) == {"MAE", "RMSE", "MAPE"}
    assert metrics["MAE"] == pytest.approx(1 / 3)
    assert metrics["RMSE"] == pytest.approx((1 / 3) ** 0.5)
    assert metrics["MAPE"] == pytest.approx(11.111111, rel=1e-5)


# ---------------------------------------------------------------------------
# Forecasting helpers
# ---------------------------------------------------------------------------


class DummyArima:
    def predict(self, h=1, level=None):
        return pd.DataFrame({
            "mean": [1] * h,
            "lo-95": [0] * h,
            "hi-95": [2] * h,
        })


def sample_series(freq="M", periods: int = 3):
    idx = pd.date_range("2020-01-31", periods=periods, freq=freq)
    return pd.Series(range(1, periods + 1), index=idx, dtype=float)


def test_forecast_arima_basic(monkeypatch):
    monkeypatch.setattr(ff, "AutoARIMA", object)
    s = sample_series()
    res = ff.forecast_arima(DummyArima(), s, 2)
    assert list(res.columns) == ["forecast", "lower_ci", "upper_ci"]
    assert len(res) == 2
    assert res.index[0] > s.index[-1]


def test_forecast_arima_requires_freq(monkeypatch):
    monkeypatch.setattr(ff, "AutoARIMA", object)
    s = pd.Series(
        [1, 2, 3],
        index=pd.to_datetime(["2020-01-01", "2020-01-05", "2020-02-01"]),
    )
    with pytest.raises(ValueError):
        ff.forecast_arima(DummyArima(), s, 1)


class DummyXGB:
    def predict(self, X):
        return np.array([5.0])


def test_forecast_xgb_iterative(monkeypatch):
    monkeypatch.setattr(ff, "XGBRegressor", object)

    call_lengths = []

    def fake_to_supervised(series, n_lags, add_time_features=True, exog=None):
        call_lengths.append(len(series))
        cols = {f"lag{i}": [0] for i in range(1, n_lags + 1)}
        return pd.DataFrame(cols), pd.Series([0])

    monkeypatch.setattr(ff, "_to_supervised", fake_to_supervised)

    s = sample_series()
    res = ff.forecast_xgb(DummyXGB(), s, 3, n_lags=2, rmse=1.0)

    assert list(res.columns) == ["forecast", "lower_ci", "upper_ci"]
    assert len(res) == 3
    assert call_lengths == [3, 4, 5]


class DummyModel:
    def predict(self, X, verbose=0):
        return np.array([[0.0]])


class DummyScaler:
    def transform(self, x):
        return np.asarray(x)

    def inverse_transform(self, x):
        return np.asarray(x)


def test_forecast_lstm():
    s = sample_series()
    res = ff.forecast_lstm(
        DummyModel(),
        DummyScaler(),
        s,
        periods=2,
        window_size=2,
        rmse=1.0,
    )
    assert list(res.columns) == ["forecast", "lower_ci", "upper_ci"]
    assert len(res) == 2

    s_nofreq = pd.Series(
        [1, 2, 3],
        index=pd.to_datetime(["2020-01-01", "2020-01-05", "2020-01-20"]),
    )
    with pytest.raises(ValueError):
        ff.forecast_lstm(
            DummyModel(),
            DummyScaler(),
            s_nofreq,
            periods=1,
            window_size=1,
            rmse=1.0,
        )


class DummyCat:
    def __init__(self, *a, **k):
        pass

    def fit(self, X, y, cat_features=None):
        self.fitted = True

    def predict(self, X):
        return np.array([1.0] * len(X))


def test_forecast_future_catboost(monkeypatch):
    monkeypatch.setattr(cb, "CatBoostRegressor", DummyCat, raising=False)
    series = sample_series(periods=12)
    res = cb.forecast_future_catboost(series, "M", horizon=2)
    assert "yhat_catboost" in res.columns
    assert len(res) == 2


def test_forecast_future_catboost_constant(monkeypatch):
    monkeypatch.setattr(cb, "CatBoostRegressor", DummyCat, raising=False)
    series = pd.Series([5.0] * 12, index=pd.date_range("2020-01-31", periods=12, freq="M"))
    res = cb.forecast_future_catboost(series, "M", horizon=3)
    assert (res["yhat_catboost"] == 5.0).all()


def test_evaluate_catboost_constant():
    series = pd.Series([5.0] * 24, index=pd.date_range("2020-01-31", periods=24, freq="M"))
    metrics = em._evaluate_catboost(series, "M", test_size=6)
    assert metrics == {"MAE": 0.0, "RMSE": 0.0, "MAPE": 0.0}


def test_rolling_forecast_catboost_constant(monkeypatch):
    monkeypatch.setattr(cb, "CatBoostRegressor", DummyCat, raising=False)
    df = cb.prepare_supervised(pd.Series([5.0] * 18, index=pd.date_range("2020-01-31", periods=18, freq="M")), "M")
    preds, actuals = cb.rolling_forecast_catboost(df, "M", test_size=3)
    assert preds == [5.0, 5.0, 5.0]
    assert preds == actuals


# ---------------------------------------------------------------------------
# Plotting and pipeline
# ---------------------------------------------------------------------------



def test_plot_with_forecasts(monkeypatch, tmp_path):
    monkeypatch.setattr(make_plots, "fit_all_arima", lambda *a, **k: (object(), object(), object()))
    monkeypatch.setattr(make_plots, "fit_prophet_models", lambda *a, **k: (object(), object(), object()))
    monkeypatch.setattr(make_plots, "train_xgb_model", lambda *a, **k: (object(), 0.0))
    monkeypatch.setattr(make_plots, "train_lstm_model", lambda *a, **k: (object(), object(), None))
    monkeypatch.setattr(
        make_plots,
        "forecast_arima",
        lambda model, series, p: ff.forecast_arima(DummyArima(), series, p),
    )
    monkeypatch.setattr(
        make_plots,
        "forecast_prophet",
        lambda model, series, p: ff.forecast_arima(DummyArima(), series, p),
    )
    monkeypatch.setattr(
        make_plots,
        "forecast_xgb",
        lambda model, series, p, **kw: ff.forecast_arima(DummyArima(), series, p),
    )
    monkeypatch.setattr(
        make_plots,
        "forecast_lstm",
        lambda model, scaler, series, p, **kw: ff.forecast_arima(DummyArima(), series, p),
    )
    monkeypatch.setattr(
        make_plots,
        "forecast_future_catboost",
        lambda series, freq, horizon=None: pd.DataFrame({"yhat_catboost": [0] * (horizon or 1)}, index=pd.date_range(series.index[-1]+series.index.freq, periods=horizon or 1, freq=series.index.freq)),
    )

    s = sample_series()
    out = tmp_path / "plot.png"
    make_plots.plot_with_forecasts(s, "M", out)
    assert out.exists() and out.stat().st_size > 0


def test_main_pipeline(monkeypatch, tmp_path):
    monkeypatch.setattr(
        make_plots,
        "plot_with_forecasts",
        lambda series, freq, output: Path(output).write_bytes(b"x"),
    )
    monkeypatch.setattr(
        make_plots,
        "plot_metrics",
        lambda metrics, out: Path(out).write_bytes(b"x"),
    )

    series = sample_series()
    metrics = pd.DataFrame({"MAE_monthly": [1], "RMSE_monthly": [2], "MAPE_monthly": [3]}, index=["x"])
    make_plots.main(
        output_dir=str(tmp_path),
        csv_path=None,
        metrics=metrics,
        ts_monthly=series,
        ts_quarterly=series.resample("Q").sum(),
        ts_yearly=series.resample("A").sum(),
    )

    assert (tmp_path / "metrics_comparison.png").exists()


def test_main_requires_csv(tmp_path):
    with pytest.raises(ValueError):
        make_plots.main(output_dir=str(tmp_path), csv_path=None)
