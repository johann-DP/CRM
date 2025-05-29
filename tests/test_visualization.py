import pandas as pd
import numpy as np
import phase4_functions as pf
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def test_plot_scree_file(tmp_path):
    arr = np.array([2.0, 1.0, 0.5])
    out = tmp_path / "scree.png"
    pf.plot_scree(arr, "PCA", out)
    assert out.exists() and out.stat().st_size > 0


def test_plot_correlation_circle_file(tmp_path):
    df = pd.DataFrame(np.random.rand(10, 3), columns=["A", "B", "C"])
    model = PCA(n_components=2).fit(df)
    out = tmp_path / "circle.png"
    pf.plot_correlation_circle(model, ["A", "B", "C"], out)
    assert out.exists() and out.stat().st_size > 0


def test_plot_embedding_file(tmp_path):
    coords = pd.DataFrame({"F1": [0, 1], "F2": [1, 0]})
    out = tmp_path / "emb.png"
    pf.plot_embedding(coords, color_by=["x", "y"], title="test", output_path=out)
    assert out.exists() and out.stat().st_size > 0


def test_generate_figures_handles_3d(tmp_path):
    df_active = pd.DataFrame({
        "num1": [1, 2, 3],
        "num2": [4, 5, 6],
        "cat": ["a", "b", "a"],
    })
    factor_results = {
        "pca": {
            "embeddings": pd.DataFrame(
                np.random.rand(3, 3), columns=["F1", "F2", "F3"]
            ),
            "cluster_labels": np.array([0, 1, 0]),
        }
    }
    nonlin_results = {
        "umap": {
            "embeddings": pd.DataFrame(
                np.random.rand(3, 3), columns=["U1", "U2", "U3"]
            ),
            "cluster_labels": np.array([0, 1, 0]),
        }
    }
    figs = pf.generate_figures(
        factor_results,
        nonlin_results,
        df_active,
        ["num1", "num2"],
        ["cat"],
        output_dir=None,
        cluster_k=2,
    )
    assert figs
    assert "pca_cluster_comparison" in figs


def test_plot_cluster_grid():
    emb = pd.DataFrame(np.random.rand(10, 2), columns=["X", "Y"])
    km_labels, km_k = pf.tune_kmeans_clusters(emb.values, range(2, 3))
    ag_labels, ag_k = pf.tune_agglomerative_clusters(emb.values, range(2, 3))
    db_labels, db_eps = pf.tune_dbscan_clusters(emb.values, eps_values=[0.5])
    fig = pf.plot_cluster_grid(
        emb,
        km_labels,
        ag_labels,
        db_labels,
        "test",
        km_k,
        ag_k,
        db_eps,
    )
    assert isinstance(fig, plt.Figure)
