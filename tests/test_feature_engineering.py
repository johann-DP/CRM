import pandas as pd
import numpy as np
import types
import pytest

import feature_engineering as fe

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, MinMaxScaler
from sklearn.feature_selection import chi2, mutual_info_classif


def _patch_sklearn(monkeypatch):
    monkeypatch.setattr(fe, "np", np, False)
    monkeypatch.setattr(fe, "SimpleImputer", SimpleImputer, False)
    monkeypatch.setattr(fe, "OrdinalEncoder", OrdinalEncoder, False)
    monkeypatch.setattr(fe, "StandardScaler", StandardScaler, False)
    monkeypatch.setattr(fe, "MinMaxScaler", MinMaxScaler, False)
    monkeypatch.setattr(fe, "chi2", chi2, False)
    monkeypatch.setattr(fe, "mutual_info_classif", mutual_info_classif, False)


def _fake_requests(monkeypatch):
    sirene_calls = {}
    geo_calls = {}

    def fake_get(url, timeout=5):
        if "entreprise.data.gouv.fr" in url:
            siren = url.split("/")[-1]
            sirene_calls[siren] = sirene_calls.get(siren, 0) + 1
            if siren == "123456789":
                data = {
                    "unite_legale": {
                        "activite_principale": "69123",
                        "tranche_effectifs": "200-500",
                    }
                }
                return types.SimpleNamespace(status_code=200, json=lambda: data)
            return types.SimpleNamespace(status_code=404, json=lambda: {})
        else:
            import re

            cp = re.search(r"codePostal=(\d+)", url).group(1)
            geo_calls[cp] = geo_calls.get(cp, 0) + 1
            if cp == "75001":
                data = [{"population": 1000, "codeRegion": "11"}]
                return types.SimpleNamespace(
                    status_code=200,
                    json=lambda: data,
                    raise_for_status=lambda: None,
                )
            elif cp == "31000":
                data = [{"population": 500, "codeRegion": "76"}]
                return types.SimpleNamespace(
                    status_code=200,
                    json=lambda: data,
                    raise_for_status=lambda: None,
                )
            return types.SimpleNamespace(
                status_code=404,
                json=lambda: [],
                raise_for_status=lambda: None,
            )

    monkeypatch.setattr(
        fe,
        "requests",
        types.SimpleNamespace(get=fake_get, RequestException=Exception),
        False,
    )
    fe.clear_caches()
    return sirene_calls, geo_calls


@pytest.fixture()
def sample_data():
    train = pd.DataFrame(
        {
            "SIREN": [
                "123456789",
                "123456789",
                "987654321",
                "000000000",
                "987654321",
            ],
            "Code postal": [
                "75001",
                "75001",
                "31000",
                "99999",
                "99999",
            ],
            "Date de clôture": [
                "2024-01-10",
                "2024-02-20",
                "2024-03-05",
                "2024-04-15",
                "2024-05-10",
            ],
            "Date de début actualisée": [
                "2024-01-01",
                "2024-02-01",
                None,
                "2024-04-01",
                "2024-05-05",
            ],
            "Date de fin réelle": [
                "2024-01-09",
                "2024-02-10",
                "2024-03-25",
                "2024-04-20",
                None,
            ],
            "category": ["A", "Rare1", "B", "B", "Rare2"],
            "Budget": [100, 200, 150, 400, 120],
            "is_won": [1, 0, 1, 0, 1],
        }
    )
    val = train.iloc[:2].copy()
    test = train.iloc[3:].copy()
    return train, val, test


def test_create_internal_features(sample_data):
    train, val, test = sample_data
    cfg = {"date_col": "Date de clôture", "numeric_features": [], "cat_features": []}
    fe.create_internal_features(train, val, test, cfg)

    for df in (train, val, test):
        assert {"month", "year", "duree_entre_debut_fin"} <= set(df.columns)

    assert train.loc[0, "month"] == 1
    assert train.loc[1, "year"] == 2024
    assert train.loc[0, "duree_entre_debut_fin"] == 8
    assert train.loc[2, "duree_entre_debut_fin"] == 0


def test_reduce_categorical_levels(sample_data):
    train, val, test = sample_data
    fe.reduce_categorical_levels(train, val, test, ["category"], min_freq=2)

    assert list(train["category"]) == ["Autre", "Autre", "B", "B", "Autre"]
    assert list(val["category"]) == ["Autre", "Autre"]
    assert list(test["category"]) == ["B", "Autre"]


def test_reduce_categorical_levels_with_nan():
    train = pd.DataFrame({"category": ["A", "B", np.nan, "B", np.nan]})
    val = train.iloc[:2].copy()
    test = train.iloc[2:].copy()

    fe.reduce_categorical_levels(train, val, test, "category", min_freq=2)

    for df in (train, val, test):
        assert not df["category"].isna().any()
        assert set(df["category"].cat.categories) == {"B", "Autre"}


def test_enrich_with_sirene(sample_data, monkeypatch):
    train, val, test = sample_data
    sirene_calls, geo_calls = _fake_requests(monkeypatch)
    fe.enrich_with_sirene(train, val, test)

    assert {"secteur_activite", "tranche_effectif"} <= set(train.columns)
    assert train.loc[0, "secteur_activite"] == "69123"
    assert train.loc[0, "tranche_effectif"] == "200-500"
    assert train.loc[3, "secteur_activite"] == "inconnu"
    assert sirene_calls["123456789"] == 1
    assert sirene_calls["000000000"] == 1


def test_enrich_with_geo_data(sample_data, monkeypatch):
    train, val, test = sample_data
    _patch_sklearn(monkeypatch)
    sirene_calls, geo_calls = _fake_requests(monkeypatch)
    fe.enrich_with_geo_data(train, val, test)

    assert {"population_commune", "code_region"} <= set(train.columns)
    assert train.loc[0, "population_commune"] == 1000
    assert train.loc[0, "code_region"] == "11"
    assert train.loc[3, "population_commune"] == 0
    assert geo_calls["75001"] == 1
    assert geo_calls["99999"] == 1


def test_enrichment_caching(sample_data, monkeypatch):
    train, val, test = sample_data
    _patch_sklearn(monkeypatch)
    sirene_calls, geo_calls = _fake_requests(monkeypatch)
    fe.clear_caches()
    fe.enrich_with_sirene(train, val, test)
    fe.enrich_with_geo_data(train, val, test)
    # second run should not trigger additional API calls
    fe.enrich_with_sirene(train, val, test)
    fe.enrich_with_geo_data(train, val, test)
    assert sirene_calls["123456789"] == 1
    assert geo_calls["75001"] == 1


def test_enrich_with_sirene_missing_column(monkeypatch):
    train = pd.DataFrame({"Budget": [1, 2]})
    val = train.copy()
    test = train.copy()
    _fake_requests(monkeypatch)

    fe.enrich_with_sirene(train, val, test)

    for df in (train, val, test):
        assert {"secteur_activite", "tranche_effectif"} <= set(df.columns)
        assert (df["secteur_activite"] == "inconnu").all()
        assert (df["tranche_effectif"] == "inconnu").all()


def test_enrich_with_geo_data_missing_column(monkeypatch):
    train = pd.DataFrame({"Budget": [1, 2]})
    val = train.copy()
    test = train.copy()
    _fake_requests(monkeypatch)

    fe.enrich_with_geo_data(train, val, test)

    for df in (train, val, test):
        assert {"population_commune", "code_region"} <= set(df.columns)
        assert (df["population_commune"] == 0).all()
        assert (df["code_region"] == "nc").all()


def test_advanced_feature_engineering_missing_columns(monkeypatch):
    train = pd.DataFrame(
        {
            "Date de clôture": ["2024-01-10", "2024-02-20"],
            "category": ["A", "B"],
            "Budget": [100, 200],
            "is_won": [1, 0],
        }
    )
    val = train.iloc[:1].copy()
    test = train.iloc[1:].copy()

    _patch_sklearn(monkeypatch)
    _fake_requests(monkeypatch)

    cfg = {
        "date_col": "Date de clôture",
        "cat_features": ["category"],
        "numeric_features": ["Budget"],
        "min_cat_freq": 1,
        "target_col": "is_won",
    }

    X_train, X_val, X_test = fe.advanced_feature_engineering(
        train.copy(), val.copy(), test.copy(), cfg
    )

    assert not X_train.isna().any().any()
    assert "secteur_activite" in X_train.columns


def test_advanced_feature_engineering(sample_data, monkeypatch):
    train, val, test = sample_data
    _patch_sklearn(monkeypatch)
    sirene_calls, geo_calls = _fake_requests(monkeypatch)

    cfg = {
        "date_col": "Date de clôture",
        "cat_features": ["category"],
        "numeric_features": ["Budget"],
        "min_cat_freq": 2,
        "target_col": "is_won",
    }

    X_train, X_val, X_test = fe.advanced_feature_engineering(
        train.copy(), val.copy(), test.copy(), cfg
    )

    expected_cols = {
        "month",
        "year",
        "duree_entre_debut_fin",
        "secteur_activite",
        "tranche_effectif",
        "population_commune",
        "code_region",
    }
    assert expected_cols <= set(cfg["numeric_features"] + cfg["cat_features"])
    assert list(X_train.columns) == cfg["numeric_features"] + cfg["cat_features"]
    assert X_train.shape[1] == len(cfg["numeric_features"]) + len(cfg["cat_features"])

    num_means = X_train[cfg["numeric_features"]].mean().abs()
    assert (num_means < 1e-6).all()
    num_stds = X_train[cfg["numeric_features"]].std(ddof=0)
    assert (np.isclose(num_stds, 1, atol=1e-6) | np.isclose(num_stds, 0, atol=1e-6)).all()
    assert not X_train.isna().any().any()
    assert sirene_calls["123456789"] == 1
    assert geo_calls["75001"] == 1


def test_run_lead_scoring_pipeline(tmp_path, monkeypatch):
    out_dir = tmp_path / "out"
    cfg = {"lead_scoring": {"output_dir": str(out_dir)}}
    cfg_path = tmp_path / "cfg.yaml"
    import yaml
    cfg_path.write_text(yaml.safe_dump(cfg))

    import pandas as pd
    from pred_lead_scoring import run_lead_scoring

    def fake_preprocess(cfg):
        X = pd.DataFrame({"x": [1, 2]})
        y = pd.Series([0, 1])
        ts = pd.DataFrame({"conv_rate": [0.1, 0.2]})
        return X, y, X, y, X, y, ts, ts, ts

    monkeypatch.setattr(run_lead_scoring, "preprocess", fake_preprocess)
    for name in [
        "train_xgboost_lead",
        "train_catboost_lead",
        "train_logistic_lead",
        "train_mlp_lead",
        "train_ensemble_lead",
        "train_arima_conv_rate",
        "train_prophet_conv_rate",
    ]:
        monkeypatch.setattr(run_lead_scoring, name, lambda *a, **k: None)

    metrics = pd.DataFrame({"model": ["xgb"], "auc": [0.5]})

    def fake_eval(cfg, X_test, y_test, ts):
        (out_dir / "data_cache").mkdir(parents=True, exist_ok=True)
        for name in [
            "X_train.csv",
            "y_train.csv",
            "X_val.csv",
            "y_val.csv",
            "X_test.csv",
            "y_test.csv",
        ]:
            (out_dir / "data_cache" / name).write_text("dummy")
        return metrics

    monkeypatch.setattr(run_lead_scoring, "evaluate_lead_models", fake_eval)
    monkeypatch.setattr(run_lead_scoring, "plot_results", lambda *a, **k: None)

    run_lead_scoring.main(["--config", str(cfg_path)])

    assert (out_dir / "data_cache" / "X_train.csv").exists()

