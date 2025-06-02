"""Preprocessing utilities for lead scoring dataset."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple
import numpy as np

import pandas as pd
import yaml
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler


# ---------------------------------------------------------------------------
# Data loading and filtering
# ---------------------------------------------------------------------------

def _load_data(path: Path) -> pd.DataFrame:
    """Read the CSV file and parse the closing date."""
    df = pd.read_csv(path)
    if "Date de clôture" not in df.columns:
        raise ValueError("'Date de clôture' column missing")
    df["Date de clôture"] = pd.to_datetime(
        df["Date de clôture"], dayfirst=True, errors="coerce"
    )
    return df


def _filter_status(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only won/lost opportunities and add ``is_won`` column."""
    df = df[df["Statut_final"].isin(["Gagné", "Perdu"])]
    df = df.dropna(subset=["Date de clôture", "Statut_final"]).copy()
    df["is_won"] = (df["Statut_final"] == "Gagné").astype(int)
    return df


# ---------------------------------------------------------------------------
# Dataset splitting
# ---------------------------------------------------------------------------

def _split_sets(
    df: pd.DataFrame, test_start: str, test_end: str
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return train/validation/test DataFrames based on closing date."""
    start = pd.to_datetime(test_start)
    end = pd.to_datetime(test_end)

    train = df[df["Date de clôture"] < start].copy()
    val = df[(df["Date de clôture"] >= start) & (df["Date de clôture"] <= end)].copy()
    test = df[df["Date de clôture"] > end].copy()
    return train, val, test


# ---------------------------------------------------------------------------
# Feature encoding
# ---------------------------------------------------------------------------

def _encode_features(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    cat_features: list[str],
    num_features: list[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Impute/scale numeric vars and ordinal-encode categorical vars."""
    imp = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)

    X_train_num = imp.fit_transform(train[num_features])
    X_val_num = imp.transform(val[num_features])
    X_test_num = imp.transform(test[num_features])

    X_train_num = scaler.fit_transform(X_train_num)
    X_val_num = scaler.transform(X_val_num)
    X_test_num = scaler.transform(X_test_num)

    X_train_cat = enc.fit_transform(train[cat_features].astype(str))
    X_val_cat = enc.transform(val[cat_features].astype(str))
    X_test_cat = enc.transform(test[cat_features].astype(str))

    cols = num_features + cat_features
    X_train = pd.DataFrame(
        np.column_stack([X_train_num, X_train_cat]),
        columns=cols,
        index=train.index,
    )
    X_val = pd.DataFrame(
        np.column_stack([X_val_num, X_val_cat]),
        columns=cols,
        index=val.index,
    )
    X_test = pd.DataFrame(
        np.column_stack([X_test_num, X_test_cat]),
        columns=cols,
        index=test.index,
    )
    return X_train, X_val, X_test


# ---------------------------------------------------------------------------
# Conversion rate time series
# ---------------------------------------------------------------------------

def _conversion_time_series(df: pd.DataFrame) -> pd.DataFrame:
    df_closed = df[df["Statut_final"].notna()].copy()
    df_closed = df_closed.set_index("Date de clôture")
    ts = df_closed["is_won"].resample("M").agg(["sum", "count"])
    ts["conv_rate"] = ts["sum"] / ts["count"]
    return ts


# ---------------------------------------------------------------------------
# Main preprocessing routine
# ---------------------------------------------------------------------------

def preprocess_lead_scoring(cfg: Dict[str, Dict]) -> None:
    lead_cfg = cfg["lead_scoring"]
    csv_path = Path(lead_cfg["input_path"])
    out_dir = Path(lead_cfg.get("output_dir", cfg.get("output_dir", ".")))
    out_dir.mkdir(parents=True, exist_ok=True)

    df = _load_data(csv_path)
    df = _filter_status(df)

    train, val, test = _split_sets(
        df, lead_cfg["test_start"], lead_cfg["test_end"]
    )

    X_train, X_val, X_test = _encode_features(
        train,
        val,
        test,
        lead_cfg["cat_features"],
        lead_cfg["numeric_features"],
    )
    y_train = train["is_won"]
    y_val = val["is_won"]
    y_test = test["is_won"]

    # Conversion rate time series
    ts_conv = _conversion_time_series(df)
    start = pd.to_datetime(lead_cfg["test_start"])
    ts_conv_rate_train = ts_conv[:start]
    ts_conv_rate_test = ts_conv[start:]

    df_prophet_train = ts_conv_rate_train[["conv_rate"]].rename(columns={"conv_rate": "y"})
    df_prophet_train = df_prophet_train.rename_axis("ds").reset_index()

    # Export datasets
    X_train.to_csv(out_dir / "X_train.csv", index=False)
    y_train.to_csv(out_dir / "y_train.csv", index=False)
    X_val.to_csv(out_dir / "X_val.csv", index=False)
    y_val.to_csv(out_dir / "y_val.csv", index=False)
    X_test.to_csv(out_dir / "X_test.csv", index=False)
    y_test.to_csv(out_dir / "y_test.csv", index=False)

    ts_conv_rate_train.to_csv(out_dir / "ts_conv_rate_train.csv")
    ts_conv_rate_test.to_csv(out_dir / "ts_conv_rate_test.csv")
    df_prophet_train.to_csv(out_dir / "df_prophet_train.csv", index=False)


def preprocess(cfg: Dict[str, Dict]):
    """Return the datasets produced by :func:`preprocess_lead_scoring`.

    This convenience wrapper simply calls :func:`preprocess_lead_scoring` and
    loads the generated CSV files back into memory.  It is used by the main
    pipeline so that downstream functions can directly consume the data without
    dealing with intermediate files.

    Parameters
    ----------
    cfg : dict
        Configuration dictionary containing a ``lead_scoring`` section.

    Returns
    -------
    tuple
        ``(X_train, y_train, X_val, y_val, X_test, y_test, ts_conv_train,
        ts_conv_test, df_prophet_train)`` where each element is a
        :class:`pandas.DataFrame` or :class:`pandas.Series`.
    """

    preprocess_lead_scoring(cfg)

    lead_cfg = cfg.get("lead_scoring", {})
    out_dir = Path(lead_cfg.get("output_dir", cfg.get("output_dir", ".")))

    X_train = pd.read_csv(out_dir / "X_train.csv")
    y_train = pd.read_csv(out_dir / "y_train.csv").squeeze()
    X_val = pd.read_csv(out_dir / "X_val.csv")
    y_val = pd.read_csv(out_dir / "y_val.csv").squeeze()
    X_test = pd.read_csv(out_dir / "X_test.csv")
    y_test = pd.read_csv(out_dir / "y_test.csv").squeeze()

    ts_conv_train = pd.read_csv(
        out_dir / "ts_conv_rate_train.csv", index_col=0, parse_dates=True
    )
    ts_conv_test = pd.read_csv(
        out_dir / "ts_conv_rate_test.csv", index_col=0, parse_dates=True
    )
    df_prophet_train = pd.read_csv(out_dir / "df_prophet_train.csv", parse_dates=["ds"])

    return (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        ts_conv_train,
        ts_conv_test,
        df_prophet_train,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Preprocess lead scoring dataset")
    p.add_argument("--config", default="config.yaml", help="Path to YAML config")
    args = p.parse_args(argv)

    with open(args.config, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    preprocess_lead_scoring(cfg)


if __name__ == "__main__":
    main()
