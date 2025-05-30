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
    monkeypatch.setattr(phase4, "save_segment_analysis_figures", lambda *a, **k: None)
    phase4.run_pipeline(cfg)

    assert called.get("optimize") is False
    assert called.get("n_components") == 2


def test_run_pipeline_parallel_calls(monkeypatch, tmp_path):
    calls = {}

    def fake_run_pipeline(cfg):
        calls[cfg["dataset"]] = cfg["output_dir"]
        return {}

    class FakeParallel:
        def __init__(self, n_jobs=None, backend=None):
            calls["n_jobs"] = n_jobs
            calls["backend"] = backend
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            pass

        def __call__(self, tasks):
            return [task() for task in tasks]

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    def fake_delayed(func):
        def wrapper(*args, **kwargs):
            return lambda: func(*args, **kwargs)
        return wrapper

    monkeypatch.setattr(phase4, "run_pipeline", fake_run_pipeline)
    monkeypatch.setattr(phase4, "plot_general_heatmap", lambda *a, **k: None)
    monkeypatch.setattr(phase4, "Parallel", FakeParallel)
    monkeypatch.setattr(phase4, "delayed", fake_delayed)

    cfg = {"output_dir": str(tmp_path / "out"), "input_file": "dummy"}
    datasets = ["raw", "cleaned_1"]

    res = phase4.run_pipeline_parallel(
        cfg, datasets, n_jobs=2, backend="multiprocessing"
    )

    assert set(res) == set(datasets)
    for name in datasets:
        assert Path(calls[name]).name == name
    assert calls["n_jobs"] == 2
    assert calls["backend"] == "multiprocessing"


def test_run_pipeline_parallel_builds_report(monkeypatch, tmp_path):
    build_calls = {}

    def fake_run_pipeline(cfg):
        Path(cfg["output_dir"]).mkdir(parents=True, exist_ok=True)
        return {}

    class FakeParallel:
        def __init__(self, n_jobs=None, backend=None):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            pass

        def __call__(self, tasks):
            return [task() for task in tasks]

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    def fake_delayed(func):
        def wrapper(*args, **kwargs):
            return lambda: func(*args, **kwargs)
        return wrapper

    def fake_export(figs, tables, pdf_path):
        build_calls["args"] = (figs, tables, pdf_path)
        Path(pdf_path).parent.mkdir(parents=True, exist_ok=True)
        Path(pdf_path).write_text("final")
        return Path(pdf_path)

    monkeypatch.setattr(phase4, "run_pipeline", fake_run_pipeline)
    monkeypatch.setattr(phase4, "plot_general_heatmap", lambda *a, **k: None)
    monkeypatch.setattr(phase4, "export_report_to_pdf", fake_export)
    monkeypatch.setattr(phase4, "Parallel", FakeParallel)
    monkeypatch.setattr(phase4, "delayed", fake_delayed)

    cfg = {
        "output_dir": str(tmp_path / "out"),
        "input_file": "dummy",
        "output_pdf": str(tmp_path / "out" / "phase4_report.pdf"),
    }
    datasets = ["raw", "cleaned_1", "cleaned_3_univ", "cleaned_3_multi"]

    phase4.run_pipeline_parallel(cfg, datasets)

    figs, tables, pdf_path = build_calls.get("args")
    assert isinstance(figs, dict)
    assert Path(pdf_path) == Path(cfg["output_pdf"])
    