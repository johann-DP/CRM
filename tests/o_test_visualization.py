import pandas as pd
import numpy as np
import phase4.functions as pf
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
        segment_col="cat",
    )
    assert figs


def test_cluster_segment_table_and_heatmap():
    labels = np.array([0, 0, 1, 1, 0])
    segs = pd.Series(["A", "B", "A", "A", "B"])
    tab = pf.cluster_segment_table(labels, segs)
    assert tab.loc[0, "A"] == 1
    fig = pf.plot_cluster_segment_heatmap(tab, "test")
    assert hasattr(fig, "savefig")


def test_cluster_confusion_table_and_heatmap():
    a = np.array([0, 0, 1, 1])
    b = np.array([1, 0, 1, 0])
    tab = pf.cluster_confusion_table(a, b)
    assert tab.loc[0, 0] == 1
    fig = pf.plot_cluster_confusion_heatmap(tab, "test")
    assert hasattr(fig, "savefig")


def test_plot_clusters_by_k():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(20, 2)), columns=["F1", "F2"])
    fig = pf.plot_clusters_by_k(X, "kmeans", [2, 3], "pca")
    assert hasattr(fig, "savefig")


def test_cluster_evaluation_and_stability_plots():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, 2))
    curves = {}
    opts = {}
    for method in ["kmeans", "agglomerative", "gmm", "spectral"]:
        df, best = pf.cluster_evaluation_metrics(X, method, range(2, 4))
        curves[method] = df
        opts[method] = best
        fig = pf.plot_cluster_evaluation(df, method, best)
        assert hasattr(fig, "savefig")
    comb = pf.plot_combined_silhouette(curves, opts)
    assert hasattr(comb, "savefig")

    for method in ["kmeans", "agglomerative", "gmm", "spectral"]:
        labels, best_k, table = pf.optimize_clusters(method, X, range(2, 4))
        assert len(labels) == X.shape[0]
        assert best_k in table["k"].values

    metrics = {
        "d1": {"pca_axis_corr_mean": 0.8, "pca_var_first_axis_mean": 0.5},
        "d2": {"pca_axis_corr_mean": 0.9, "pca_var_first_axis_mean": 0.6},
    }
    figs = pf.plot_pca_stability_bars(metrics)
    for fig in figs.values():
        assert hasattr(fig, "savefig")


def test_plot_scatter_ellipses(tmp_path):
    coords = pd.DataFrame({"X": [0, 1, 0, 1], "Y": [0, 0, 1, 1]})
    labels = pd.Series([0, 0, 1, 1])
    out = tmp_path / "ell.png"
    pf.plot_scatter_ellipses(coords, labels, output_path=out)
    assert out.exists() and out.stat().st_size > 0
