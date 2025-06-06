import pandas as pd
from pathlib import Path

from phase4bis.export_mfa_inertias import export_mfa_inertias


def test_export_mfa_inertias(tmp_path: Path) -> None:
    df = pd.DataFrame({
        "A": [1, 2, 3, 4],
        "B": [4, 3, 2, 1],
        "cat1": ["x", "y", "x", "y"],
        "cat2": ["u", "v", "u", "v"],
    })
    csv_path = tmp_path / "raw.csv"
    df.to_csv(csv_path, index=False)

    cfg = {
        "input_file": str(csv_path),
        "dataset": "raw",
        "min_modalite_freq": 1,
        "mfa": {"n_components": 2, "groups": [["A", "B"], ["cat1", "cat2"]]},
    }
    out = tmp_path / "inerties.csv"

    table = export_mfa_inertias(cfg, output=out)
    assert out.exists()
    loaded = pd.read_csv(out)
    assert "variance_expliquee_pct" in loaded.columns
    assert len(loaded) == len(table) == 2
