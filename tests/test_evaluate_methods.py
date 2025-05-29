import pandas as pd
import numpy as np
import pytest

import phase4_functions as pf


def sample_df() -> pd.DataFrame:
    return pd.DataFrame({
        "num1": [0, 1, 2, 3],
        "num2": [3, 2, 1, 0],
        "cat": ["a", "b", "a", "b"],
    })


def test_evaluate_and_plot(tmp_path, monkeypatch):
    df = sample_df()
    quant_vars = ["num1", "num2"]
    qual_vars = ["cat"]

    emb_a = pd.DataFrame(
        [[0, 0], [1, 0], [0, 1], [1, 1]],
        index=df.index,
        columns=["d1", "d2"],
    )
    emb_b = pd.DataFrame(
        [[0, 0], [1, 1], [2, 0], [0, 2]],
        index=df.index,
        columns=["d1", "d2"],
    )

    results = {
        "A": {"embeddings": emb_a, "inertia": [2/3, 1/3], "runtime_s": 0.1},
        "B": {"embeddings": emb_b, "inertia": [1/3, 1/6], "runtime_s": 0.2},
    }

    monkeypatch.setattr(pf, "trustworthiness", lambda *args, **kwargs: 0.5)

    metrics = pf.evaluate_methods(
        results, df, quant_vars, qual_vars, k_range=range(2, 3)
    )

    # cluster labels added
    for info in results.values():
        assert "cluster_labels" in info
        assert len(info["cluster_labels"]) == len(df)

    # check table structure
    expected_cols = {
        "variance_cumulee_%",
        "nb_axes_kaiser",
        "silhouette",
        "dunn_index",
        "trustworthiness",
        "continuity",
        "runtime_seconds",
        "cluster_algo",
    }
    assert set(metrics.columns) == expected_cols
    assert list(metrics.index) == ["A", "B"]

    # variance cumulative check
    assert metrics.loc["A", "variance_cumulee_%"] == pytest.approx(100.0)
    assert metrics.loc["B", "variance_cumulee_%"] == pytest.approx(50.0)

    # metric ranges
    assert metrics["silhouette"].between(-1, 1).all()
    for val in metrics["trustworthiness"]:
        assert np.isnan(val) or 0 <= val <= 1

    pf.plot_methods_heatmap(metrics, tmp_path)
    assert (tmp_path / "methods_heatmap.png").exists()
