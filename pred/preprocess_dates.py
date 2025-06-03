"""Utility functions to fix erroneous closing dates and aggregate revenue."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple, Dict

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor


# ---------------------------------------------------------------------------
# Loading and basic cleaning
# ---------------------------------------------------------------------------

def load_csv(path: str | Path) -> pd.DataFrame:
    """Return DataFrame parsed from ``path`` with date columns coerced."""
    df = pd.read_csv(path)
    for col in ["Date de fin actualisée", "Date de début actualisée", "Date de fin réelle"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce", dayfirst=True)
    return df


def replace_future_dates(df: pd.DataFrame) -> int:
    """Replace dates in 2040 and beyond with ``NaT``.

    Returns the number of replaced values.
    """
    col = "Date de fin actualisée"
    mask = df[col].notna() & (df[col].dt.year >= 2040)
    count = int(mask.sum())
    df.loc[mask, col] = pd.NaT
    assert not (df[col].dropna().dt.year >= 2040).any()
    return count


def copy_real_end_dates(df: pd.DataFrame) -> int:
    """Copy ``Date de fin réelle`` over ``Date de fin actualisée`` when won."""
    col = "Date de fin actualisée"
    mask = (df["Statut commercial"] == "Gagné") & df["Date de fin réelle"].notna()
    count = int(mask.sum())
    df.loc[mask, col] = df.loc[mask, "Date de fin réelle"]
    return count


# ---------------------------------------------------------------------------
# Duration statistics
# ---------------------------------------------------------------------------

def build_history(df: pd.DataFrame) -> Tuple[pd.DataFrame, float]:
    """Return subset with valid dates and the median project duration."""
    hist = df[df["Date de fin réelle"].notna() & df["Date de début actualisée"].notna()].copy()
    hist["duree_jours"] = (
        hist["Date de fin réelle"] - hist["Date de début actualisée"]
    ).dt.days
    median = float(hist["duree_jours"].median())
    return hist, median


def impute_with_median(df: pd.DataFrame, duration: float) -> int:
    """Fill missing closing dates using ``duration`` after the start date."""
    col = "Date de fin actualisée"
    mask = df[col].isna() & df["Date de début actualisée"].notna()
    count = int(mask.sum())
    df.loc[mask, col] = df.loc[mask, "Date de début actualisée"] + pd.to_timedelta(duration, unit="D")
    return count


def train_duration_model(hist: pd.DataFrame) -> Tuple[RandomForestRegressor, list[str]]:
    """Train a random forest on numeric features to predict durations."""
    feature_candidates = [
        "Total recette réalisé",
        "Budget client estimé",
        "Charge prévisionnelle projet",
    ]
    features = [c for c in feature_candidates if c in hist.columns]
    if not features:
        raise ValueError("No suitable feature columns found for regression")

    X = hist[features]
    y = hist["duree_jours"]

    reg = RandomForestRegressor(random_state=0)
    reg.fit(X, y)
    return reg, features


def impute_with_model(
    df: pd.DataFrame, reg: RandomForestRegressor, features: list[str]
) -> int:
    """Predict missing closing dates using ``reg`` and the provided features."""
    col = "Date de fin actualisée"
    mask = df[col].isna() & df["Date de début actualisée"].notna()
    if not mask.any():
        return 0

    X_pred = df.loc[mask, features]
    preds = reg.predict(X_pred)
    df.loc[mask, col] = df.loc[mask, "Date de début actualisée"] + pd.to_timedelta(preds, unit="D")
    return int(mask.sum())


# ---------------------------------------------------------------------------
# Filtering and aggregation
# ---------------------------------------------------------------------------

def filter_won(df: pd.DataFrame) -> pd.DataFrame:
    """Return copy of ``df`` only with won opportunities and valid dates."""
    out = df[df["Statut commercial"] == "Gagné"].copy()
    out = out.dropna(subset=["Date de fin actualisée"])
    return out


def aggregate_revenue(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Aggregate revenue into monthly, quarterly and yearly sums."""
    df = df.set_index("Date de fin actualisée")
    monthly = df["Total recette réalisé"].resample("M").sum().fillna(0)
    quarterly = df["Total recette réalisé"].resample("Q").sum().fillna(0)
    yearly = df["Total recette réalisé"].resample("A").sum().fillna(0)
    return monthly, quarterly, yearly


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

def plot_histograms(df_before: pd.DataFrame, df_after: pd.DataFrame, out_dir: Path) -> None:
    """Save histograms of the closing year before and after correction."""
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 4))
    df_before["Date de fin actualisée"].dt.year.hist(bins=30)
    plt.title("Distribution avant correction")
    plt.xlabel("Année de fin")
    plt.tight_layout()
    plt.savefig(out_dir / "hist_before.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 4))
    df_after["Date de fin actualisée"].dt.year.hist(bins=30)
    plt.title("Distribution après correction")
    plt.xlabel("Année de fin")
    plt.tight_layout()
    plt.savefig(out_dir / "hist_after.png", dpi=150)
    plt.close()


def plot_before_after(ts_before: pd.Series, ts_after: pd.Series, out_dir: Path) -> None:
    """Plot revenue time series before/after correction."""
    plt.figure(figsize=(12, 6))
    ts_before.plot(label="Avant correction")
    ts_after.plot(label="Après correction")
    plt.legend()
    plt.ylabel("Total recette réalisé")
    plt.tight_layout()
    plt.savefig(out_dir / "timeseries_before_after.png", dpi=150)
    plt.close()


def save_summary(info: Dict[str, int], out_dir: Path) -> None:
    """Save a textual summary of the corrections."""
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "correction_summary.csv"
    pd.DataFrame([info]).to_csv(summary_path, index=False)


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def preprocess_dates(csv_path: str | Path, output_dir: str | Path = "output_dir") -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Full pipeline returning aggregated revenue series."""
    out_dir = Path(output_dir)
    df = load_csv(csv_path)
    df_before = df.copy()

    replaced = replace_future_dates(df)
    copied = copy_real_end_dates(df)
    hist, median = build_history(df)
    imputed_median = impute_with_median(df, median)
    reg, features = train_duration_model(hist)
    imputed_model = impute_with_model(df, reg, features)
    # Remove any dates that still fall in 2040 or beyond after imputation
    replaced_after = replace_future_dates(df)

    df_won = filter_won(df)
    monthly, quarterly, yearly = aggregate_revenue(df_won)

    ts_before = (
        df_before.set_index("Date de fin actualisée")["Total recette réalisé"].resample("M").sum().fillna(0)
    )

    plot_histograms(df_before, df, out_dir)
    plot_before_after(ts_before, monthly, out_dir)

    info = {
        "replaced_2050": replaced,
        "replaced_after_impute": replaced_after,
        "copied_real_end": copied,
        "imputed_median": imputed_median,
        "imputed_model": imputed_model,
        "final_rows": len(df_won),
    }
    save_summary(info, out_dir)

    return monthly, quarterly, yearly


__all__ = [
    "load_csv",
    "replace_future_dates",
    "copy_real_end_dates",
    "build_history",
    "impute_with_median",
    "train_duration_model",
    "impute_with_model",
    "filter_won",
    "aggregate_revenue",
    "plot_histograms",
    "plot_before_after",
    "save_summary",
    "preprocess_dates",
]


if __name__ == "__main__":  # pragma: no cover - simple CLI
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess closing dates")
    parser.add_argument("csv", help="Path to phase3_cleaned_multivariate.csv")
    parser.add_argument("--output-dir", default="output_dir", help="Destination for figures and summary")
    args = parser.parse_args()

    preprocess_dates(args.csv, args.output_dir)
