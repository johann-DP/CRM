"""Preprocessing utilities for lead scoring dataset."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple
import numpy as np

import pandas as pd
import yaml
from joblib import dump, load

try:  # optional dependency for large datasets
    import dask.dataframe as dd
except Exception:  # pragma: no cover - dask not installed
    dd = None
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler


# ---------------------------------------------------------------------------
# Data loading and filtering
# ---------------------------------------------------------------------------


def _load_data(
    path: Path,
    date_col: str,
    date_format: str | None = None,
    dayfirst: bool = False,
) -> pd.DataFrame:
    """Read the CSV file and parse the closing date.

    If the file contains several hundred thousand rows and :mod:`dask` is
    available, ``dask.dataframe`` is used to limit memory usage.  ``date_col``
    specifies the column containing the closing date. ``date_format`` may be
    provided to enforce parsing with :func:`pandas.to_datetime`. If omitted,
    ``dayfirst`` controls the default parsing behaviour.
    """
    if dd is not None:
        try:
            # Rough heuristic based on line count
            with open(path, "r", encoding="utf-8") as fh:
                for i, _ in enumerate(fh):
                    if i > 300_000:
                        break
            big = i > 300_000
        except Exception:  # pragma: no cover - file read issues
            big = False
    else:
        big = False

    if dd is not None and big:
        df = dd.read_csv(path).compute()
    else:
        df = pd.read_csv(path)

    if date_col not in df.columns:
        raise ValueError(f"'{date_col}' column missing")

    if date_format:
        df[date_col] = pd.to_datetime(df[date_col], format=date_format, errors="coerce")
    else:
        df[date_col] = pd.to_datetime(df[date_col], dayfirst=dayfirst, errors="coerce")

    return df


def _clean_closing_dates(df: pd.DataFrame, date_col: str) -> int:
    """Replace unrealistic closing dates with ``NaT``.

    Parameters
    ----------
    df : DataFrame
        Dataset with a ``Date de fin actualisée`` column parsed as datetime.

    Returns
    -------
    int
        Number of replaced values.
    """

    col = date_col
    future_limit = pd.Timestamp("2025-03-01")
    past_limit = pd.Timestamp("1995-01-01")

    mask_future = df[col].notna() & (df[col] > future_limit)
    mask_past = df[col].notna() & (df[col] < past_limit)
    mask = mask_future | mask_past
    count = int(mask.sum())
    df.loc[mask, col] = pd.NaT
    return count


def _filter_status(
    df: pd.DataFrame,
    date_col: str,
    target_col: str,
    positive_label: str,
) -> pd.DataFrame:
    """Keep only won/lost opportunities and add ``is_won`` column."""

    df = df[df[target_col].isin([positive_label, "Perdu"])]
    df = df.dropna(subset=[date_col, target_col]).copy()
    df["is_won"] = (df[target_col] == positive_label).astype(int)
    return df


# ---------------------------------------------------------------------------
# Dataset splitting
# ---------------------------------------------------------------------------


def _split_sets(
    df: pd.DataFrame,
    date_col: str,
    test_start: str,
    test_end: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return train/validation/test DataFrames based on closing date."""
    start = pd.to_datetime(test_start)
    end = pd.to_datetime(test_end)

    train = df[df[date_col] < start].copy()
    val = df[(df[date_col] >= start) & (df[date_col] <= end)].copy()
    test = df[df[date_col] > end].copy()
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
    if len(val) > 0:
        X_val_num = imp.transform(val[num_features])
    else:
        X_val_num = np.empty((0, len(num_features)))
    if len(test) > 0:
        X_test_num = imp.transform(test[num_features])
    else:
        X_test_num = np.empty((0, len(num_features)))

    X_train_num = scaler.fit_transform(X_train_num)
    if len(val) > 0:
        X_val_num = scaler.transform(X_val_num)
    if len(test) > 0:
        X_test_num = scaler.transform(X_test_num)

    X_train_cat = enc.fit_transform(train[cat_features].astype(str))
    if len(val) > 0:
        X_val_cat = enc.transform(val[cat_features].astype(str))
    else:
        X_val_cat = np.empty((0, len(cat_features)))
    if len(test) > 0:
        X_test_cat = enc.transform(test[cat_features].astype(str))
    else:
        X_test_cat = np.empty((0, len(cat_features)))

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


def _conversion_time_series(
    df: pd.DataFrame,
    date_col: str,
    target_col: str,
) -> pd.DataFrame:
    df_closed = df[df[target_col].notna()].copy()
    df_closed = df_closed.set_index(date_col)
    ts = df_closed["is_won"].resample("M").agg(["sum", "count"]).fillna(0.0)
    ts["conv_rate"] = np.where(ts["count"] > 0, ts["sum"] / ts["count"], 0.0)
    return ts


# ---------------------------------------------------------------------------
# Main preprocessing routine
# ---------------------------------------------------------------------------


def preprocess_lead_scoring(cfg: Dict[str, Dict]) -> None:
    """Preprocess the lead scoring dataset and cache intermediate files."""

    lead_cfg = cfg.get("lead_scoring", {})
    if not lead_cfg:
        raise KeyError("'lead_scoring' section missing from configuration")
    csv_path = Path(lead_cfg["input_path"])
    out_dir = Path(lead_cfg.get("output_dir", cfg.get("output_dir", ".")))
    data_dir = out_dir / "data_cache"
    data_dir.mkdir(parents=True, exist_ok=True)

    date_col = lead_cfg.get("date_col", "Date de fin actualisée")
    target_col = lead_cfg.get("target_col", "Statut commercial")
    positive_label = lead_cfg.get("positive_label", "Gagné")
    date_format = lead_cfg.get("date_format")
    dayfirst = lead_cfg.get("dayfirst", False)

    df = _load_data(csv_path, date_col, date_format=date_format, dayfirst=dayfirst)
    cleaned = _clean_closing_dates(df, date_col)
    if cleaned:
        print(f"Dates hors limites remplacées: {cleaned}")
    df = _filter_status(df, date_col, target_col, positive_label)

    train, val, test = _split_sets(
        df,
        date_col,
        lead_cfg["test_start"],
        lead_cfg["test_end"],
    )

    cache_file = data_dir / "encoded_features.joblib"
    if cache_file.exists():
        X_train, X_val, X_test = load(cache_file)
    else:
        X_train, X_val, X_test = _encode_features(
            train,
            val,
            test,
            lead_cfg["cat_features"],
            lead_cfg["numeric_features"],
        )
        dump((X_train, X_val, X_test), cache_file)
    y_train = train["is_won"]
    y_val = val["is_won"]
    y_test = test["is_won"]

    # Conversion rate time series
    ts_conv = _conversion_time_series(df, date_col, target_col)
    start = pd.to_datetime(lead_cfg["test_start"])
    ts_conv_rate_train = ts_conv[:start]
    ts_conv_rate_test = ts_conv[start:]

    df_prophet_train = ts_conv_rate_train[["conv_rate"]].rename(
        columns={"conv_rate": "y"}
    )
    df_prophet_train = df_prophet_train.rename_axis("ds").reset_index()

    # Export datasets for potential reuse
    X_train.to_csv(data_dir / "X_train.csv", index=False)
    y_train.to_csv(data_dir / "y_train.csv", index=False)
    X_val.to_csv(data_dir / "X_val.csv", index=False)
    y_val.to_csv(data_dir / "y_val.csv", index=False)
    X_test.to_csv(data_dir / "X_test.csv", index=False)
    y_test.to_csv(data_dir / "y_test.csv", index=False)

    ts_conv_rate_train.to_csv(data_dir / "ts_conv_rate_train.csv")
    ts_conv_rate_test.to_csv(data_dir / "ts_conv_rate_test.csv")
    df_prophet_train.to_csv(data_dir / "df_prophet_train.csv", index=False)


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
    data_dir = out_dir / "data_cache"

    X_train = pd.read_csv(data_dir / "X_train.csv")
    y_train = pd.read_csv(data_dir / "y_train.csv").squeeze()
    X_val = pd.read_csv(data_dir / "X_val.csv")
    y_val = pd.read_csv(data_dir / "y_val.csv").squeeze()
    X_test = pd.read_csv(data_dir / "X_test.csv")
    y_test = pd.read_csv(data_dir / "y_test.csv").squeeze()

    ts_conv_train = pd.read_csv(
        data_dir / "ts_conv_rate_train.csv", index_col=0, parse_dates=True
    )
    ts_conv_test = pd.read_csv(
        data_dir / "ts_conv_rate_test.csv", index_col=0, parse_dates=True
    )
    df_prophet_train = pd.read_csv(
        data_dir / "df_prophet_train.csv", parse_dates=["ds"]
    )

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
