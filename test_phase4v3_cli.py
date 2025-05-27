import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


def test_phase4v3_cli(tmp_path: Path):
    df = pd.DataFrame({
        "Date de debut": ["2024-01-01", "2024-02-01", "2024-03-01", "2024-04-01", "2024-05-01"],
        "Total recette réalisé": [1000, 2500, 1500, 2000, 3000],
        "Catégorie": ["A", "B", "A", "B", "A"],
    })
    data_path = tmp_path / "raw.csv"
    df.to_csv(data_path, index=False)

    cfg = {
        "input_file": str(data_path),
        "dataset": "raw",
        "output_dir": str(tmp_path / "out"),
        "output_pdf": str(tmp_path / "out" / "report.pdf"),
        "methods": ["pca"],
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(cfg))

    result = subprocess.run(
        [sys.executable, "phase4v3.py", "--config", str(config_path)],
        cwd=Path(__file__).resolve().parent,
        check=True,
    )
    assert result.returncode == 0
    metrics_path = tmp_path / "out" / "metrics.csv"
    pdf_path = tmp_path / "out" / "report.pdf"
    assert metrics_path.exists()
    assert pdf_path.exists()
