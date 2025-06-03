"""Utilities to summarise forecast performance across time granularities."""

from __future__ import annotations

from typing import Dict, Mapping

import pandas as pd
import matplotlib.pyplot as plt


MetricDict = Mapping[str, float]
ModelResults = Mapping[str, Mapping[str, MetricDict]]


def build_performance_table(results: ModelResults) -> pd.DataFrame:
    """Return DataFrame of metrics organised by model and frequency.

    Parameters
    ----------
    results:
        Mapping where ``results[model][freq]`` gives a mapping of metric names
        to values. ``freq`` should be one of ``{"monthly", "quarterly", "yearly"}``.

    Returns
    -------
    pd.DataFrame
        Table with models as index and columns like ``MAE_monthly``,
        ``RMSE_quarterly`` etc.
    """
    frames = []
    for freq in ["monthly", "quarterly", "yearly"]:
        data = {
            model: metrics.get(freq, {})
            for model, metrics in results.items()
        }
        df = pd.DataFrame.from_dict(data, orient="index")
        df.columns = [f"{col}_{freq}" for col in df.columns]
        frames.append(df)
    table = pd.concat(frames, axis=1)
    return table


def plot_metric_comparison(table: pd.DataFrame, metric: str) -> None:
    """Display bar chart of ``metric`` for each model and frequency."""
    cols = [f"{metric}_monthly", f"{metric}_quarterly", f"{metric}_yearly"]
    subset = table[cols]
    subset.plot(kind="bar")
    plt.ylabel(metric)
    plt.title(f"Comparison of {metric} across granularities")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()


__all__ = ["build_performance_table", "plot_metric_comparison"]
