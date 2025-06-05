"""Feature engineering utilities for lead scoring."""

from __future__ import annotations

from typing import Iterable, List, Tuple, Dict

import json
import os
import warnings
import ntpath
import tempfile

import pandas as pd
import requests

SIRENE_CACHE: Dict[str, Tuple[str, str]] = {}
GEO_CACHE: Dict[str, Tuple[int, str]] = {}

# hardcoded cache file locations
_SIRENE_CACHE_FILE = r"C:\Users\johan\Documents\sirene_cache.json"
_GEO_CACHE_FILE = r"C:\Users\johan\Documents\geo_cache.json"

from .preprocess_lead_scoring import _encode_features


__all__ = [
    "create_internal_features",
    "reduce_categorical_levels",
    "enrich_with_sirene",
    "enrich_with_geo_data",
    "advanced_feature_engineering",
    "clear_caches",
]


def clear_caches() -> None:
    """Clear cached API responses used during enrichment."""
    SIRENE_CACHE.clear()
    GEO_CACHE.clear()
    for path in (_SIRENE_CACHE_FILE, _GEO_CACHE_FILE):
        try:
            os.remove(path)
        except FileNotFoundError:
            pass
        except Exception as e:  # pragma: no cover - non critical
            warnings.warn(f"Could not remove cache file {path}: {e}")


def _load_cache(file_path: str) -> dict:
    """Load cache data from ``file_path`` if possible."""
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            warnings.warn(f"Cache file {file_path} has invalid format; ignoring.")
            return {}
        for k, v in data.items():
            if isinstance(v, list):
                data[k] = tuple(v)
        return data
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError:
        warnings.warn(
            f"Cache file {file_path} is corrupted or unreadable; starting fresh cache."
        )
        return {}
    except Exception as e:  # pragma: no cover - unexpected errors
        warnings.warn(f"Could not load cache file {file_path}: {e}; ignoring cache.")
        return {}


def _save_cache(file_path: str, data: dict) -> None:
    """Atomically save ``data`` to ``file_path`` and merge concurrent updates."""
    dir_path = ntpath.dirname(file_path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)

    if os.path.exists(file_path):
        try:
            with open(file_path, "r") as f:
                current = json.load(f)
            if isinstance(current, dict):
                for k, v in current.items():
                    if k not in data:
                        data[k] = v
        except Exception:
            pass

    temp_path = file_path + ".tmp"
    try:
        with open(temp_path, "w") as f:
            json.dump(data, f)
        os.replace(temp_path, file_path)
    except Exception as e:  # pragma: no cover - difficult to trigger
        warnings.warn(f"Failed to write cache file {file_path}: {e}")


def _fetch_sirene_data(siren: str) -> Tuple[str, str]:
    """Fetch SIRENE info for ``siren`` via API."""
    url = f"https://entreprise.data.gouv.fr/api/sirene/v3/etablissements/{siren}"
    try:
        resp = requests.get(url, timeout=5)
        if resp.status_code == 200:
            data = resp.json().get("unite_legale", {}) or {}
            ap = data.get("activite_principale", "inconnu")
            te = data.get("tranche_effectifs") or data.get("tranche_effectif") or "inconnu"
            return (ap, te)
        return ("inconnu", "inconnu")
    except Exception:
        return ("inconnu", "inconnu")


def _fetch_geo_data(cp: str) -> Tuple[int, str]:
    """Fetch GEO info for postal code ``cp`` via API."""
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
                return (int(pop or 0), str(reg or "nc"))
            return (0, "nc")
        return (0, "nc")
    except Exception:
        return (0, "nc")


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

    # Count intermediate status changes if a history column is available
    hist_col = next((c for c in train.columns if "statut" in c.lower() and "hist" in c.lower()), None)
    if hist_col:
        for df in (train, val, test):
            if hist_col in df.columns:
                df["nb_changements_statut"] = (
                    df[hist_col]
                    .astype(str)
                    .str.split(r"[>|;/]")
                    .apply(lambda x: len([s for s in x if s and str(s).strip() != ""]) - 1)
                    .fillna(0)
                    .astype(int)
                )
        new_feats.append("nb_changements_statut")

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
    """Enrich datasets with basic SIRENE information using local caching.

    Gracefully handles the absence of the ``SIREN`` column by populating the
    target columns with ``"inconnu"``.
    """

    # load existing cache once
    existing = _load_cache(_SIRENE_CACHE_FILE)
    if existing:
        SIRENE_CACHE.update(existing)

    series_list = [s for s in (train.get("SIREN"), val.get("SIREN"), test.get("SIREN")) if s is not None]
    sirens = pd.concat(series_list).dropna().unique() if series_list else []

    info: Dict[str, Tuple[str, str]] = {}
    missing: List[str] = []
    for siren in sirens:
        key = str(siren)
        if key in SIRENE_CACHE:
            info[key] = SIRENE_CACHE[key]
        else:
            missing.append(key)

    for key in missing:
        data = _fetch_sirene_data(key)
        info[key] = data
        SIRENE_CACHE[key] = data

    if missing or not os.path.exists(_SIRENE_CACHE_FILE):
        _save_cache(_SIRENE_CACHE_FILE, SIRENE_CACHE)

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
    """Add population and region code based on postal code using local caching.

    Works even if the ``Code postal`` column is missing from the datasets.
    """

    existing = _load_cache(_GEO_CACHE_FILE)
    if existing:
        GEO_CACHE.update(existing)

    series_list = [s for s in (train.get("Code postal"), val.get("Code postal"), test.get("Code postal")) if s is not None]
    cps = pd.concat(series_list).dropna().unique() if series_list else []
    info: Dict[str, Tuple[int, str]] = {}
    missing: List[str] = []
    for cp in cps:
        key = str(cp)
        if key in GEO_CACHE:
            info[key] = GEO_CACHE[key]
        else:
            missing.append(key)

    for key in missing:
        data = _fetch_geo_data(key)
        info[key] = data
        GEO_CACHE[key] = data

    if missing or not os.path.exists(_GEO_CACHE_FILE):
        _save_cache(_GEO_CACHE_FILE, GEO_CACHE)

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

    encoding = cfg.get("encoding", "ordinal")
    return _encode_features(train, val, test, cat_features, num_features, encoding=encoding)

