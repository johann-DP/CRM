"""Advanced feature engineering pipeline for lead scoring.

This module defines a set of utilities to build complex features for the
lead scoring dataset. The functions defined here can be combined to create
custom preprocessing pipelines involving external data sources and advanced
encoding strategies.
"""

from __future__ import annotations

from typing import Tuple

import pandas as pd
import requests


def create_internal_features(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    lead_cfg: dict,
) -> None:
    """Create additional features directly from the input datasets.

    Parameters
    ----------
    train, val, test : pd.DataFrame
        Raw datasets split into training, validation and testing sets.
    lead_cfg : dict
        Configuration dictionary describing the lead scoring setup.
    """
    pass


def reduce_categorical_levels(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    cat_cols: list[str],
    min_freq: int,
) -> None:
    """Reduce the number of modalities for categorical variables.

    This function groups rare categories together based on ``min_freq``
    occurrences in ``X_train`` and applies the mapping to the validation
    and test sets.
    """
    pass


def enrich_with_sirene(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    lead_cfg: dict | None = None,
) -> None:
    """Augment the datasets with company information from the SIRENE API.

    The function looks for a ``SIREN`` or ``SIRET`` column in ``X_train``
    and queries the `SIRENE open-data API <https://entreprise.data.gouv.fr/>`_
    to retrieve two attributes about the corresponding company:

    * ``secteur_activite`` – the main activity code (NAF)
    * ``tranche_effectif`` – the employee count bracket

    Results are cached locally to minimise the number of HTTP requests.  In
    case of network failure or missing data, the value ``"inconnu"`` is used.
    When ``lead_cfg`` is provided, the added column names are appended to
    ``lead_cfg["cat_features"]``.

    Parameters
    ----------
    X_train, X_val, X_test : pd.DataFrame
        Datasets that must contain either a ``SIREN`` or ``SIRET`` column.
    lead_cfg : dict, optional
        Configuration dictionary updated in-place with the newly created
        categorical features.
    """

    if "SIREN" in X_train.columns:
        col = "SIREN"
    elif "SIRET" in X_train.columns:
        col = "SIRET"
    else:
        return

    def _fetch_from_api(siren: str, cache: dict[str, tuple[str, str]]) -> tuple[str, str]:
        if siren in cache:
            return cache[siren]

        url = f"https://entreprise.data.gouv.fr/api/sirene/v3/unites_legales/{siren}"
        secteur = "inconnu"
        effectif = "inconnu"
        try:
            resp = requests.get(url, timeout=2)
            if resp.status_code == 200:
                payload = resp.json().get("unite_legale", {})
                secteur = payload.get("activite_principale") or "inconnu"
                effectif = (
                    payload.get("tranche_effectifs")
                    or payload.get("tranche_effectifs_salaries")
                    or "inconnu"
                )
        except requests.RequestException:
            pass

        cache[siren] = (secteur, effectif)
        return secteur, effectif

    cache: dict[str, tuple[str, str]] = {}

    sirens = (
        pd.concat([X_train[col], X_val[col], X_test[col]])
        .dropna()
        .astype(str)
        .str[:9]
        .unique()
    )

    for siren in sirens:
        _fetch_from_api(siren, cache)

    for df in (X_train, X_val, X_test):
        df["secteur_activite"] = (
            df[col]
            .map(lambda x: cache.get(str(x)[:9], ("inconnu", "inconnu"))[0])
            .fillna("inconnu")
        )
        df["tranche_effectif"] = (
            df[col]
            .map(lambda x: cache.get(str(x)[:9], ("inconnu", "inconnu"))[1])
            .fillna("inconnu")
        )

    if lead_cfg is not None:
        cat_feats = list(lead_cfg.get("cat_features", []))
        for feat in ["secteur_activite", "tranche_effectif"]:
            if feat not in cat_feats:
                cat_feats.append(feat)
        lead_cfg["cat_features"] = cat_feats


def enrich_with_geo_data(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
) -> None:
    """Add geographic information such as coordinates or region codes."""
    pass


def advanced_feature_engineering(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    lead_cfg: dict,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return the fully engineered feature matrices.

    This high-level helper orchestrates the various feature engineering
    steps defined in this module and returns processed copies of the
    ``train``, ``val`` and ``test`` datasets.
    """
    pass
