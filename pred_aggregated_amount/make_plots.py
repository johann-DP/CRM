from pathlib import Path

import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .preprocess_timeseries import load_and_aggregate, preprocess_all
from .catboost_forecast import forecast_future_catboost
from .train_xgboost import train_xgb_model
from .train_arima import fit_all_arima
from .lstm_forecast import train_lstm_model
from .prophet_models import fit_prophet_models
from .future_forecast import (
    forecast_arima,
    forecast_prophet,
    forecast_xgb,
    forecast_lstm,
)


def load_original(csv_path: Path) -> pd.DataFrame:
    """Return cleaned DataFrame filtered on won opportunities."""
    df = pd.read_csv(csv_path, dayfirst=True)
    df = df[df["Statut commercial"] == "Gagné"].copy()
    df["Date de fin actualisée"] = pd.to_datetime(
        df["Date de fin actualisée"], errors="coerce", dayfirst=True
    )
    df = df.dropna(subset=["Date de fin actualisée"])
    return df


def plot_scatter(df: pd.DataFrame, out: Path, sort_dates: bool = False) -> None:
    """Plot revenue against closing date."""
    if sort_dates:
        df = df.sort_values("Date de fin actualisée")
    plt.figure(figsize=(12, 6))
    plt.plot(
        df["Date de fin actualisée"],
        df["Total recette réalisé"],
        marker=".",
        linestyle="none",
    )
    plt.xlabel("Date de fin actualisée")
    plt.ylabel("Total recette réalisé")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()


def plot_with_forecasts(ts: pd.Series, freq: str, output: Path) -> None:
    """Plot actual series and forecasts from several models."""
    horizon = 60 if freq == "M" else (20 if freq == "Q" else 5)
    ts_recent = ts.tail(horizon)

    # Fit models on the full series
    arima_m, arima_q, arima_y = fit_all_arima(ts, ts, ts)
    prophet_m, prophet_q, prophet_y = fit_prophet_models(ts, ts, ts)
    xgb_model, _ = train_xgb_model(ts, n_lags=len(ts_recent), add_time_features=True)
    lstm_model, scaler, _ = train_lstm_model(ts, window_size=len(ts_recent))

    freq_code = {"M": "ME", "Q": "QE", "A": "A"}[freq]
    arima_model = {
        "M": arima_m,
        "Q": arima_q,
        "A": arima_y,
    }[freq]
    prophet_model = {
        "M": prophet_m,
        "Q": prophet_q,
        "A": prophet_y,
    }[freq]

    arima_fore = forecast_arima(arima_model, ts, len(ts_recent))
    prophet_fore = forecast_prophet(prophet_model, ts, len(ts_recent))
    xgb_fore = forecast_xgb(
        xgb_model,
        ts,
        len(ts_recent),
        n_lags=len(ts_recent),
        rmse=1.0,
        add_time_features=True,
    )
    lstm_fore = forecast_lstm(
        lstm_model,
        scaler,
        ts,
        len(ts_recent),
        window_size=len(ts_recent),
        rmse=1.0,
    )
    cat_fore = forecast_future_catboost(ts, freq, horizon=len(ts_recent))

    plt.figure(figsize=(14, 7))
    plt.plot(ts_recent.index, ts_recent.values, label="observed", marker="o")
    plt.plot(arima_fore.index, arima_fore["forecast"], label="arima")
    plt.plot(prophet_fore.index, prophet_fore["forecast"], label="prophet")
    plt.plot(xgb_fore.index, xgb_fore["forecast"], label="xgboost")
    plt.plot(lstm_fore.index, lstm_fore["forecast"], label="lstm")
    plt.plot(cat_fore.index, cat_fore["yhat_catboost"], label="catboost")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output, dpi=150)
    plt.close()


def plot_metrics(metrics: pd.DataFrame, out: Path) -> None:
    """Plot grouped bar chart of evaluation metrics."""
    metrics.plot.bar(figsize=(14, 8))
    plt.title("Comparaison des métriques")
    plt.ylabel("Valeur")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()


def main(
    output_dir: str = "output_dir",
    csv_path: str | Path | None = "phase3_cleaned_multivariate.csv",
    metrics: pd.DataFrame | None = None,
    ts_monthly: pd.Series | None = None,
    ts_quarterly: pd.Series | None = None,
    ts_yearly: pd.Series | None = None,
) -> None:
    """Generate all illustrative figures in ``output_dir``.

    Parameters
    ----------
    output_dir:
        Destination folder for the generated PNG files.
    csv_path:
        Path to the cleaned multivariate CSV used to load the original data.
    metrics:
        Optional metrics table to visualize with ``plot_metrics``.
    """

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if ts_monthly is None or ts_quarterly is None or ts_yearly is None:
        if csv_path is None:
            raise ValueError("csv_path is required when time series are not provided")
        df = load_original(Path(csv_path))
        plot_scatter(df, out_path / "recette_vs_date.png", sort_dates=False)
        plot_scatter(df, out_path / "recette_vs_date_sorted.png", sort_dates=True)

        ts = df.set_index("Date de fin actualisée")["Total recette réalisé"]
        monthly = ts.resample("M").sum()
        quarterly = ts.resample("Q").sum()
        yearly = ts.resample("A").sum()
    else:
        monthly, quarterly, yearly = ts_monthly, ts_quarterly, ts_yearly

    plot_with_forecasts(monthly, "M", out_path / "recette_monthly_with_forecasts.png")
    plot_with_forecasts(quarterly, "Q", out_path / "recette_quarterly_with_forecasts.png")
    plot_with_forecasts(yearly, "A", out_path / "recette_yearly_with_forecasts.png")

    if metrics is None:
        # Placeholder metrics if none are provided
        data = {
            "MAE_monthly": [1, 2, 3, 4, 5],
            "RMSE_monthly": [1, 2, 3, 4, 5],
            "MAPE_monthly": [1, 2, 3, 4, 5],
        }
        metrics_df = pd.DataFrame(
            data, index=["catboost", "xgboost", "arima", "lstm", "prophet"]
        )
    else:
        metrics_df = metrics

    plot_metrics(metrics_df, out_path / "metrics_comparison.png")


if __name__ == "__main__":  # pragma: no cover - simple CLI
    main()
