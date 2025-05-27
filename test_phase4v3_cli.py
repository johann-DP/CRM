import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


def create_dataset(path: Path) -> None:
    df = pd.DataFrame({
        "Total recette actualisé": [
            1000,
            1500,
            1200,
            1300,
            1400,
            1100,
            1250,
            1350,
            1450,
            1550,
            1600,
        ],
        "Statut commercial": [
            "gagné",
            "en cours",
            "en attente",
            "gagné",
            "gagné",
            "en cours",
            "gagné",
            "en cours",
            "gagné",
            "en attente",
            "gagné",
        ],
        "Date de début actualisée": [
            "2024-01-01",
            "2024-02-01",
            "2024-03-01",
            "2024-04-01",
            "2024-05-01",
            "2024-06-01",
            "2024-07-01",
            "2024-08-01",
            "2024-09-01",
            "2024-10-01",
            "2024-11-01",
        ],
    })
    df.to_csv(path, index=False)


def test_phase4v3_cli(tmp_path: Path):
    raw_path = tmp_path / "raw.csv"
    create_dataset(raw_path)
    phase1 = tmp_path / "phase1.csv"
    phase2 = tmp_path / "phase2.csv"
    phase3 = tmp_path / "phase3.csv"
    for p in [phase1, phase2, phase3]:
        create_dataset(p)
    cfg = {
        "input_file": str(raw_path),
        "phase1_file": str(phase1),
        "phase2_file": str(phase2),
        "phase3_file": str(phase3),
        "dataset": "raw",
        "output_dir": str(tmp_path),
        "output_pdf": str(tmp_path / "out.pdf"),
        "methods": ["pca"],
    }
    cfg_path = tmp_path / "cfg.json"
    with cfg_path.open("w", encoding="utf-8") as fh:
        json.dump(cfg, fh)
    res = subprocess.run([
        sys.executable,
        "phase4v3.py",
        "--config",
        str(cfg_path),
    ], cwd=Path(__file__).resolve().parent, capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    assert (tmp_path / "metrics.csv").exists()
    assert (tmp_path / "out.pdf").exists()
