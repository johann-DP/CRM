import pandas as pd
import numpy as np
import phase4
from pathlib import Path


def test_run_pipeline_respects_optimize(tmp_path, monkeypatch):
    df = pd.DataFrame({
        "num": [1, 2, 3, 4, 5, 6],
        "cat": ["a", "b", "a", "b", "c", "a"],
    })
    csv_path = tmp_path / "raw.csv"
    df.to_csv(csv_path, index=False)

    cfg = {
        "input_file": str(csv_path),
        "dataset": "raw",
        "output_dir": str(tmp_path / "out"),
        "methods": ["famd"],
        "optimize_params": False,
        "min_modalite_freq": 1,
        "famd": {"n_components": 2},
    }

    called = {}

    def fake_run_famd(df_active, quant_vars, qual_vars, n_components=None, *, optimize=False, **kwargs):
        called["n_components"] = n_components
        called["optimize"] = optimize
        cols = n_components or 1
        emb = pd.DataFrame(
            np.zeros((len(df_active), cols)),
            index=df_active.index,
            columns=[f"F{i+1}" for i in range(cols)],
        )
        return {"embeddings": emb}

    monkeypatch.setattr(phase4, "run_famd", fake_run_famd)
    monkeypatch.setattr(phase4, "evaluate_methods", lambda *a, **k: pd.DataFrame())
    monkeypatch.setattr(phase4, "plot_methods_heatmap", lambda *a, **k: None)
    monkeypatch.setattr(phase4, "generate_figures", lambda *a, **k: {})
    phase4.run_pipeline(cfg)

    assert called.get("optimize") is False
    assert called.get("n_components") == 2


def test_run_pipeline_parallel_calls(monkeypatch, tmp_path):
    calls = {}

    def fake_run_pipeline(cfg):
        calls[cfg["dataset"]] = cfg["output_dir"]
        return {}

    monkeypatch.setattr(phase4, "run_pipeline", fake_run_pipeline)

    cfg = {"output_dir": str(tmp_path / "out"), "input_file": "dummy"}
    datasets = ["raw", "cleaned_1"]

    res = phase4.run_pipeline_parallel(cfg, datasets, n_jobs=1)

    assert set(res) == set(datasets)
    for name in datasets:
        assert Path(calls[name]).name == name
