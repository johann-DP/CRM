from pathlib import Path
from typing import Dict
import pandas as pd

import pytest

import dataset_loader as dl
import phase4


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
def sample_files(tmp_path: Path) -> Dict[str, str]:
    """Return a minimal configuration with paths to sample CSV files."""
    return _make_sample_config(tmp_path)


@pytest.fixture()
def sample_files_with_dict(tmp_path: Path):
    raw = pd.DataFrame(
        {
            "Date Op": ["2024-01-01"],
            "Total recette realise": ["1 000"],
            "Categorie": ["A"],
        }
    )
    raw_path = tmp_path / "raw.csv"
    raw.to_csv(raw_path, index=False)

    phase1_path = tmp_path / "phase1.csv"
    raw.to_csv(phase1_path, index=False)
    phase2_path = tmp_path / "phase2.csv"
    raw.iloc[:0].to_csv(phase2_path, index=False)
    phase3_path = tmp_path / "phase3.csv"
    raw.to_csv(phase3_path, index=False)

    mapping = pd.DataFrame(
        {
            "original": ["Date Op", "Total recette realise", "Categorie"],
            "renamed": ["Date", "Recette", "Cat"],
        }
    )
    dict_path = tmp_path / "dict.xlsx"
    mapping.to_excel(dict_path, index=False)

    return {
        "input_file": str(raw_path),
        "phase1_file": str(phase1_path),
        "phase2_file": str(phase2_path),
        "phase3_file": str(phase3_path),
        "data_dictionary": str(dict_path),
    }


@pytest.fixture()
def sample_files_with_dict(tmp_path: Path, sample_files):
    mapping = pd.DataFrame({
        "original": ["Date Op", "Total recette realise", "Categorie"],
        "clean": ["Date", "Recette", "Cat"],
    })
    dict_path = tmp_path / "dict.xlsx"
    mapping.to_excel(dict_path, index=False)
    cfg = dict(sample_files)
    cfg["data_dictionary"] = str(dict_path)
    return cfg


def test_load_datasets_types(sample_files):
    datasets = dl.load_datasets(sample_files)

    assert set(datasets) >= {"raw", "phase1", "phase2", "phase3"}
    raw_df = datasets["raw"]
    assert pd.api.types.is_datetime64_any_dtype(raw_df["Date Op"])
    assert pd.api.types.is_numeric_dtype(raw_df["Total recette realise"])
    assert datasets["phase1"].shape[0] == 2


def test_load_datasets_structure(sample_files):
    datasets = dl.load_datasets(sample_files)

    expected_cols = ["Date Op", "Total recette realise", "Categorie"]
    expected_rows = {"raw": 2, "phase1": 2, "phase2": 1, "phase3": 2}
    for key, rows in expected_rows.items():
        assert key in datasets
        df = datasets[key]
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == expected_cols
        assert df.shape[0] == rows


def test_column_mapping(sample_files_with_dict):
    datasets = dl.load_datasets(sample_files_with_dict)

    assert list(datasets["raw"].columns) == ["Date", "Recette", "Cat"]
    for key in ["phase1", "phase2", "phase3"]:
        assert list(datasets[key].columns) == ["Date", "Recette", "Cat"]


def test_default_config_usage(sample_files):
    dl.CONFIG.clear()
    dl.CONFIG.update(sample_files)
    datasets = dl.load_datasets()
    assert set(datasets) >= {"raw", "phase1", "phase2", "phase3"}


def test_run_pipeline(tmp_path: Path, sample_files):
    cfg = dict(sample_files)
    cfg.update({"output_dir": str(tmp_path), "dataset": "raw", "methods": ["pca"]})

    out = phase4.run_pipeline(cfg)

    assert "metrics" in out
    assert isinstance(out["metrics"], pd.DataFrame)
    assert (tmp_path / "metrics.csv").exists()


def test_load_datasets_mapping(sample_files_with_dict):
    datasets = dl.load_datasets(sample_files_with_dict)

    for df in datasets.values():
        assert list(df.columns) == ["Date", "Recette", "Cat"]


def test_load_datasets_global_config(sample_files):
    dl.CONFIG.clear()
    dl.CONFIG.update(sample_files)

    datasets = dl.load_datasets()
    assert set(datasets) >= {"raw", "phase1", "phase2", "phase3"}
