import importlib
from pathlib import Path
import pandas as pd

import pytest


def _make_sample_config(tmp_path: Path) -> dict[str, str]:
    raw = pd.DataFrame(
        {
            "Date Op": ["2024-01-01", "2024-02-01"],
            "Total recette realise": ["1 000", "2 500"],
            "Categorie": ["A", "B"],
        }
    )
    raw_path = tmp_path / "raw.csv"
    raw.to_csv(raw_path, index=False)

    phase1_path = tmp_path / "phase1.csv"
    raw.to_csv(phase1_path, index=False)

    phase2_path = tmp_path / "phase2.csv"
    raw.iloc[:1].to_csv(phase2_path, index=False)

    phase3_path = tmp_path / "phase3.csv"
    raw.to_csv(phase3_path, index=False)

    return {
        "input_file": str(raw_path),
        "phase1_file": str(phase1_path),
        "phase2_file": str(phase2_path),
        "phase3_file": str(phase3_path),
    }


@pytest.fixture()
def sample_files(tmp_path: Path) -> dict[str, str]:
    return _make_sample_config(tmp_path)


@pytest.fixture()
def sample_files_with_dict(tmp_path: Path) -> dict[str, str]:
    cfg = _make_sample_config(tmp_path)
    mapping = pd.DataFrame(
        {
            "original": ["Date Op", "Total recette realise", "Categorie"],
            "renamed": ["date", "recette", "categorie"],
        }
    )
    dict_path = tmp_path / "dict.xlsx"
    mapping.to_excel(dict_path, index=False)
    cfg["data_dictionary"] = str(dict_path)
    return cfg


def test_load_datasets_types(sample_files):
    mod = importlib.import_module("phase4v3")
    datasets = mod.load_datasets(sample_files)

    assert set(datasets) >= {"raw", "phase1", "phase2", "phase3"}
    raw_df = datasets["raw"]
    assert pd.api.types.is_datetime64_any_dtype(raw_df["Date Op"])
    assert pd.api.types.is_numeric_dtype(raw_df["Total recette realise"])
    assert datasets["phase1"].shape[0] == 2


def test_load_datasets_structure(sample_files):
    mod = importlib.import_module("phase4v3")
    datasets = mod.load_datasets(sample_files)

    expected_cols = ["Date Op", "Total recette realise", "Categorie"]
    expected_rows = {"raw": 2, "phase1": 2, "phase2": 1, "phase3": 2}
    for key, rows in expected_rows.items():
        assert key in datasets
        df = datasets[key]
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == expected_cols
        assert df.shape[0] == rows


def test_load_datasets_with_dictionary(sample_files_with_dict):
    mod = importlib.import_module("phase4v3")
    datasets = mod.load_datasets(sample_files_with_dict)

    expected = ["date", "recette", "categorie"]
    for df in datasets.values():
        assert list(df.columns) == expected


def test_load_datasets_global_config(sample_files):
    mod = importlib.import_module("phase4v3")
    mod.CONFIG.clear()
    mod.CONFIG.update(sample_files)
    datasets = mod.load_datasets()
    assert "raw" in datasets and isinstance(datasets["raw"], pd.DataFrame)


def test_run_pipeline(tmp_path: Path, sample_files):
    mod = importlib.import_module("phase4v3")
    cfg = dict(sample_files)
    cfg.update({"output_dir": str(tmp_path), "dataset": "raw"})

    out = mod.run_pipeline(cfg)

    assert "metrics" in out
    assert isinstance(out["metrics"], pd.DataFrame)
    assert (tmp_path / "metrics.csv").exists()
