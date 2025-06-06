import pandas as pd
import pytest
import importlib

pdp = importlib.import_module("pred_aggregated_amount.preprocess_dates")



def test_load_csv_parses_dates(tmp_path):
    csv_content = """Date de fin actualisée,Date de début actualisée,Date de fin réelle,Statut commercial
2024-01-10,2023-12-01,2024-01-11,Gagné
2025-04-01,2024-04-01,2025-04-05,Perdu
"""
    path = tmp_path / "data.csv"
    path.write_text(csv_content)

    df = pdp.load_csv(path)
    for col in ["Date de fin actualisée", "Date de début actualisée", "Date de fin réelle"]:
        assert pd.api.types.is_datetime64_any_dtype(df[col])

    with pytest.raises(FileNotFoundError):
        pdp.load_csv(path.with_name("missing.csv"))


def test_replace_and_remove_extreme_dates():
    df = pd.DataFrame({
        "Date de fin actualisée": [
            "2026-01-01", "1990-01-01", "2024-06-01"
        ]
    })
    df["Date de fin actualisée"] = pd.to_datetime(df["Date de fin actualisée"])

    replaced_future = pdp.replace_future_dates(df)
    replaced_old = pdp.remove_old_dates(df)

    assert replaced_future == 1
    assert replaced_old == 1
    assert pd.isna(df.loc[0, "Date de fin actualisée"])
    assert pd.isna(df.loc[1, "Date de fin actualisée"])
    assert df.loc[2, "Date de fin actualisée"] == pd.Timestamp("2024-06-01")


def test_copy_real_end_dates():
    df = pd.DataFrame({
        "Date de fin actualisée": [pd.NaT, pd.Timestamp("2024-05-01")],
        "Date de fin réelle": [pd.Timestamp("2024-04-30"), pd.NaT],
        "Statut commercial": ["Gagné", "Perdu"],
    })

    count = pdp.copy_real_end_dates(df)
    assert df.loc[0, "Date de fin actualisée"] == pd.Timestamp("2024-04-30")
    assert count == 1
    assert df.loc[1, "Date de fin actualisée"] == pd.Timestamp("2024-05-01")


def test_build_history_and_median():
    df = pd.DataFrame({
        "Date de début actualisée": ["2024-01-01", "2024-02-01", None],
        "Date de fin réelle": ["2024-01-11", None, "2024-03-15"],
    })
    df = df.apply(pd.to_datetime)

    hist, median = pdp.build_history(df)
    assert len(hist) == 1
    manual_duration = (pd.Timestamp("2024-01-11") - pd.Timestamp("2024-01-01")).days
    assert hist.loc[0, "duree_jours"] == manual_duration
    assert median == float(manual_duration)


def test_impute_with_median():
    df = pd.DataFrame({
        "Date de début actualisée": pd.to_datetime(["2024-01-01", "2024-02-01"]),
        "Date de fin actualisée": [pd.NaT, pd.NaT],
    })

    count = pdp.impute_with_median(df, 10)
    assert count == 2
    assert df.loc[0, "Date de fin actualisée"] == pd.Timestamp("2024-01-11")
    assert df.loc[1, "Date de fin actualisée"] == pd.Timestamp("2024-02-11")


def test_train_duration_model_no_features():
    hist = pd.DataFrame({
        "Date de début actualisée": pd.to_datetime(["2024-01-01"]),
        "Date de fin réelle": pd.to_datetime(["2024-01-02"]),
        "duree_jours": [1],
    })
    with pytest.raises(ValueError):
        pdp.train_duration_model(hist)


def test_impute_with_model_dummy():
    df = pd.DataFrame({
        "Date de début actualisée": pd.to_datetime(["2024-01-01"]),
        "Date de fin actualisée": [pd.NaT],
        "feat1": [1],
    })

    class DummyReg:
        def predict(self, X):
            return [5]

    reg = DummyReg()
    count = pdp.impute_with_model(df, reg, ["feat1"])
    assert count == 1
    assert df.loc[0, "Date de fin actualisée"] == pd.Timestamp("2024-01-06")


def test_filter_won():
    df = pd.DataFrame({
        "Date de fin actualisée": pd.to_datetime(["2024-01-10", None, "2024-01-05"]),
        "Statut commercial": ["Gagné", "Gagné", "Perdu"],
        "Total recette réalisé": [100, 200, 300],
    })
    result = pdp.filter_won(df)
    assert list(result.index) == [pd.Timestamp("2024-01-10")]
    assert list(result["Total recette réalisé"]) == [100]


def test_save_summary(tmp_path):
    info = {"a": 1, "b": 2}
    pdp.save_summary(info, tmp_path)
    out_file = tmp_path / "correction_summary.csv"
    assert out_file.exists()
    saved = pd.read_csv(out_file)
    assert list(saved.columns) == list(info.keys())
    assert saved.iloc[0].tolist() == [1, 2]

