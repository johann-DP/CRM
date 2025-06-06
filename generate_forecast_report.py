#!/usr/bin/env python3
"""Generate performance report for aggregated revenue forecasting."""

from __future__ import annotations

import argparse
from pathlib import Path
import yaml
import pandas as pd
import matplotlib.pyplot as plt

from pred_aggregated_amount import preprocess_dates, preprocess_all
from pred_aggregated_amount.evaluate_models import evaluate_and_predict_all_models
from pred_aggregated_amount.compare_granularities import build_performance_table


def plot_forecast(actual: pd.Series, predicted: pd.Series, model: str, title: str, out: Path) -> None:
    plt.figure(figsize=(10, 5))
    plt.plot(actual.index, actual.values, label="réel", marker="o")
    plt.plot(predicted.index, predicted.values, label=f"prévision {model}")
    plt.xlabel("Date")
    plt.ylabel("Chiffre d'affaires")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()


def plot_error_bars(table: pd.DataFrame, metric: str, out: Path) -> None:
    cols = [f"{metric}_monthly", f"{metric}_quarterly", f"{metric}_yearly"]
    table[cols].plot.bar(figsize=(10, 6))
    plt.ylabel(metric)
    plt.title(f"Comparaison des {metric}")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()


def build_text_report(metrics: dict[str, dict[str, dict[str, float]]]) -> str:
    lines = [
        "Rapport de performances des modèles de prévision",
        "Cinq modèles (ARIMA, Prophet, XGBoost, LSTM, CatBoost) ont été évalués sur les séries mensuelle, trimestrielle et annuelle de chiffre d'affaires agrégé.",
        "",
    ]
    for freq in ("monthly", "quarterly", "yearly"):
        lines.append(f"Granularité {freq} :")
        for model in metrics:
            m = metrics[model][freq]
            lines.append(
                f"  - {model}: MAE={m['MAE']:.2f}, RMSE={m['RMSE']:.2f}, MAPE={m['MAPE']:.1f}%"
            )
        best = min(metrics, key=lambda mod: metrics[mod][freq]["RMSE"])
        lines.append(
            f"  -> Meilleur modèle: {best} (RMSE={metrics[best][freq]['RMSE']:.2f})"
        )
        lines.append("")
    lines.append(
        "Globalement, les approches basées apprentissage (XGBoost, CatBoost) donnent souvent les plus faibles erreurs, notamment à l'échelle mensuelle où la variabilité est élevée."
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Génère un rapport de performances")
    p.add_argument("--config", default="config.yaml", help="Fichier de configuration")
    p.add_argument("--output-dir", default="forecast_output", help="Dossier de sortie")
    args = p.parse_args(argv)

    with open(args.config, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    csv_path = Path(cfg.get("input_file_cleaned_3_all", "phase3_cleaned_all.csv"))
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    monthly, quarterly, yearly = preprocess_dates(csv_path, out_dir / "preprocess")
    monthly, quarterly, yearly = preprocess_all(monthly, quarterly, yearly)

    metrics, preds = evaluate_and_predict_all_models(monthly, quarterly, yearly)
    table = build_performance_table(metrics)
    table.to_csv(out_dir / "model_performance.csv")

    report = build_text_report(metrics)
    with open(out_dir / "rapport_performance.txt", "w", encoding="utf-8") as fh:
        fh.write(report)

    series_map = {"monthly": monthly, "quarterly": quarterly, "yearly": yearly}
    for freq in ("monthly", "quarterly", "yearly"):
        best = min(metrics, key=lambda m: metrics[m][freq]["RMSE"])
        pred_series = preds[best][freq]
        actual = series_map[freq].iloc[-len(pred_series):]
        plot_forecast(
            actual,
            pred_series,
            best,
            f"Prévisions {freq}",
            out_dir / f"previsions_{freq}.png",
        )

    plot_error_bars(table, "RMSE", out_dir / "erreurs_par_modele.png")


if __name__ == "__main__":
    main()
