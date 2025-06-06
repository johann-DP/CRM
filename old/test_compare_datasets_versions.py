import pandas as pd
from pathlib import Path

import dataset_comparison as dc


def sample_datasets():
    df1 = pd.DataFrame({
        "Code": [1, 2, 3, 4],
        "Date de début actualisée": ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"],
        "Date de fin réelle": ["2024-01-05", "2024-01-06", "2024-01-07", "2024-01-08"],
        "Total recette réalisé": [1000, 10000, 1500, 2000],
        "Budget client estimé": [1100, 10500, 1600, 2100],
        "Charge prévisionnelle projet": [800, 9500, 1300, 1700],
        "Statut commercial": ["Gagné", "Perdu", "Gagné", "Gagné"],
        "Type opportunité": ["T1", "T2", "T1", "T1"],
    })
    df2 = df1.copy()
    df2["flag_multivariate"] = [False, True, False, False]
    return {"phase1": df1, "phase3": df2}


def test_compare_versions_basic():
    datasets = sample_datasets()
    res = dc.compare_datasets_versions(datasets, min_modalite_freq=1)

    combined = res["metrics"]
    assert set(combined["dataset_version"]) == {"phase1", "phase3"}
    assert "pca" in combined["method"].values

    details = res["details"]
    assert set(details) == {"phase1", "phase3"}
    for d in details.values():
        assert isinstance(d["figures"], dict)
        assert not d["metrics"].empty


def test_compare_versions_output_dir(tmp_path, monkeypatch):
    datasets = sample_datasets()

    def dummy_generate(*args, output_dir=None, **kwargs):
        if output_dir is not None:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            (Path(output_dir) / "dummy.txt").write_text("x")
        return {}

    from dataset_comparison import generate_figures as real_generate
    monkeypatch.setattr("dataset_comparison.generate_figures", dummy_generate)

    res = dc.compare_datasets_versions(datasets, min_modalite_freq=1, output_dir=tmp_path)
    for name in datasets:
        assert (tmp_path / name / "dummy.txt").is_file()
    monkeypatch.setattr("dataset_comparison.generate_figures", real_generate)


def test_compare_versions_monkeypatched(monkeypatch):
    datasets = sample_datasets()

    def fake_eval(*args, **kwargs):
        return pd.DataFrame({"variance_cumulee_%": [0.1]}, index=["dummy"])

    monkeypatch.setattr("dataset_comparison.evaluate_methods", fake_eval)
    res = dc.compare_datasets_versions(datasets, min_modalite_freq=1)
    assert set(res["metrics"]["dataset_version"]) == set(datasets)
    assert (res["metrics"].groupby("dataset_version").size() == 1).all()
