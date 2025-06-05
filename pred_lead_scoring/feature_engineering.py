"""Feature engineering utilities for lead scoring."""

from __future__ import annotations

from typing import Iterable, List, Tuple, Dict

import pandas as pd
import requests

from .preprocess_lead_scoring import _encode_features


__all__ = [
    "create_internal_features",
    "reduce_categorical_levels",
    "enrich_with_sirene",
    "enrich_with_geo_data",
    "advanced_feature_engineering",
]


def create_internal_features(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    cfg: Dict[str, object],
) -> None:
    """Add month/year and duration features to all datasets.

    The configuration ``cfg`` is updated so that the newly created features are
    appended to ``cfg["numeric_features"]``.
    """
    date_col = cfg.get("date_col")
    new_feats: List[str] = []

    if date_col and date_col in train.columns:
        for df in (train, val, test):
            if date_col in df.columns:
                dates = pd.to_datetime(df[date_col], errors="coerce")
                df["month"] = dates.dt.month.fillna(0).astype(int)
                df["year"] = dates.dt.year.fillna(0).astype(int)
        new_feats.extend(["month", "year"])

    if {"Date de début actualisée", "Date de fin réelle"} <= set(train.columns):
        for df in (train, val, test):
            if {"Date de début actualisée", "Date de fin réelle"} <= set(df.columns):
                start = pd.to_datetime(df["Date de début actualisée"], errors="coerce")
                end = pd.to_datetime(df["Date de fin réelle"], errors="coerce")
                df["duree_entre_debut_fin"] = (end - start).dt.days.fillna(0).astype(float)
        new_feats.append("duree_entre_debut_fin")

    num = cfg.get("numeric_features", [])
    for feat in new_feats:
        if feat not in num:
            num.append(feat)
    cfg["numeric_features"] = num


def reduce_categorical_levels(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    cols: Iterable[str],
    min_freq: int = 5,
) -> None:
    """Group rare categories into ``"Autre"`` for the provided columns."""
    if isinstance(cols, str):
        cols = [cols]

    for col in cols:
        if col not in train.columns:
            continue
        counts = train[col].value_counts(dropna=False)
        frequent = set(counts[counts >= min_freq].index)
        frequent_no_nan = [x for x in frequent if pd.notna(x)]

        for df in (train, val, test):
            if col not in df.columns:
                continue
            series = df[col]
            series = series.where(series.isin(frequent_no_nan), "Autre")
            df[col] = pd.Categorical(series, categories=frequent_no_nan + ["Autre"])


def enrich_with_sirene(
    train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame
) -> None:
    """Enrich datasets with basic SIRENE information.

    Gracefully handles the absence of the ``SIREN`` column by populating the
    target columns with ``"inconnu"``.
    """

    series_list = [s for s in (train.get("SIREN"), val.get("SIREN"), test.get("SIREN")) if s is not None]
    sirens = pd.concat(series_list).dropna().unique() if series_list else []

    info: Dict[str, Tuple[str, str]] = {}
    for siren in sirens:
        url = f"https://entreprise.data.gouv.fr/api/sirene/v3/etablissements/{siren}"
        try:
            resp = requests.get(url, timeout=5)
            if resp.status_code == 200:
                data = resp.json().get("unite_legale", {})
                ap = data.get("activite_principale", "inconnu")
                te = data.get("tranche_effectifs") or data.get("tranche_effectif") or "inconnu"
                info[str(siren)] = (ap, te)
            else:
                info[str(siren)] = ("inconnu", "inconnu")
        except Exception:
            info[str(siren)] = ("inconnu", "inconnu")

    for df in (train, val, test):
        s = df.get("SIREN")
        if s is not None:
            df["secteur_activite"] = s.map(lambda x: info.get(str(x), ("inconnu", "inconnu"))[0])
            df["tranche_effectif"] = s.map(lambda x: info.get(str(x), ("inconnu", "inconnu"))[1])
        else:
            df["secteur_activite"] = "inconnu"
            df["tranche_effectif"] = "inconnu"


def enrich_with_geo_data(
    train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame
) -> None:
    """Add population and region code based on postal code.

    Works even if the ``Code postal`` column is missing from the datasets.
    """

    series_list = [s for s in (train.get("Code postal"), val.get("Code postal"), test.get("Code postal")) if s is not None]
    cps = pd.concat(series_list).dropna().unique() if series_list else []
    info: Dict[str, Tuple[int, str]] = {}
    for cp in cps:
        url = (
            "https://geo.api.gouv.fr/communes?codePostal="
            f"{cp}&fields=population,codeRegion&format=json"
        )
        try:
            resp = requests.get(url, timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                if isinstance(data, list) and data:
                    pop = data[0].get("population", 0)
                    reg = data[0].get("codeRegion")
                else:
                    pop, reg = 0, "nc"
                info[str(cp)] = (int(pop or 0), reg or "nc")
            else:
                info[str(cp)] = (0, "nc")
        except Exception:
            info[str(cp)] = (0, "nc")

    for df in (train, val, test):
        cp = df.get("Code postal")
        if cp is not None:
            df["population_commune"] = cp.map(lambda x: info.get(str(x), (0, "nc"))[0])
            df["code_region"] = cp.map(lambda x: info.get(str(x), (0, "nc"))[1])
        else:
            df["population_commune"] = 0
            df["code_region"] = "nc"


def advanced_feature_engineering(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    cfg: Dict[str, object],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Full feature engineering pipeline used in tests."""
    create_internal_features(train, val, test, cfg)
    enrich_with_sirene(train, val, test)
    enrich_with_geo_data(train, val, test)

    cat_features = cfg.get("cat_features", [])
    num_features = cfg.get("numeric_features", [])

    for col in ["secteur_activite", "tranche_effectif", "code_region"]:
        if col not in cat_features:
            cat_features.append(col)
    if "population_commune" not in num_features:
        num_features.append("population_commune")

    cfg["cat_features"] = cat_features
    cfg["numeric_features"] = num_features

    min_freq = cfg.get("min_cat_freq", 5)
    reduce_categorical_levels(train, val, test, cat_features, min_freq=min_freq)

    return _encode_features(train, val, test, cat_features, num_features)

