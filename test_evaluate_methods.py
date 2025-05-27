import pandas as pd
import numpy as np

from factor_methods import run_pca
from evaluate_methods import evaluate_methods, plot_methods_heatmap


def sample_df() -> pd.DataFrame:
    return pd.DataFrame({
        "num1": [1, 2, 3, 4, 5],
        "num2": [5, 4, 3, 2, 1],
        "cat": ["a", "b", "a", "b", "a"],
    })


def test_evaluate_and_plot(tmp_path):
    df = sample_df()
    quant_vars = ["num1", "num2"]
    qual_vars = ["cat"]
    pca_res = run_pca(df, quant_vars, n_components=2)
    dummy_emb = pd.DataFrame(
        np.linspace(0, 1, 10).reshape(5, 2), index=df.index, columns=["d1", "d2"]
    )
    results = {
        "pca": {
            "embeddings": pca_res["embeddings"],
            "inertia": pca_res["inertia"],
            "runtime_s": 0.1,
        },
        "dummy": {
            "embeddings": dummy_emb,
            "runtime_s": 0.2,
        },
    }
    metrics = evaluate_methods(results, df, quant_vars, qual_vars, n_clusters=2)

    assert set(metrics.columns) == {
        "variance_cumulee_%",
        "nb_axes_kaiser",
        "silhouette",
        "dunn_index",
        "trustworthiness",
        "continuity",
        "runtime_seconds",
    }
    assert "pca" in metrics.index

    plot_methods_heatmap(metrics, tmp_path)
    assert (tmp_path / "methods_heatmap.png").exists()

