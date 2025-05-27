import pandas as pd
import matplotlib
matplotlib.use("Agg")

from visualization import generate_figures


def test_generate_figures_basic(tmp_path):
    df = pd.DataFrame({
        "num1": [1, 2, 3],
        "num2": [4, 5, 6],
        "cat": ["a", "b", "a"],
    })

    factor_results = {
        "pca": {
            "embeddings": pd.DataFrame(
                [[0.1, 0.2, 0.3], [0.0, -0.1, 0.2], [0.2, 0.1, 0.0]],
                index=df.index,
                columns=["F1", "F2", "F3"],
            ),
            "loadings": pd.DataFrame(
                [[0.7, 0.2], [0.1, 0.9]],
                index=["num1", "num2"],
                columns=["F1", "F2"],
            ),
            "inertia": pd.Series([0.6, 0.3], index=["F1", "F2"]),
        }
    }

    nonlin_results = {
        "umap": {
            "embeddings": pd.DataFrame(
                [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
                index=df.index,
                columns=["UMAP1", "UMAP2"],
            )
        }
    }

    figs = generate_figures(
        factor_results,
        nonlin_results,
        df,
        ["num1", "num2"],
        ["cat"],
        output_dir=tmp_path,
    )
    assert "pca_correlation" in figs
    assert "pca_scatter_2d" in figs
    assert "umap_scatter_2d" in figs
    for f in figs.values():
        assert hasattr(f, "savefig")


def test_generate_figures_missing_f2(tmp_path):
    df = pd.DataFrame({
        "num1": [1, 2, 3],
        "num2": [4, 5, 6],
        "cat": ["a", "b", "a"],
    })

    factor_results = {
        "pca": {
            "embeddings": pd.DataFrame(
                [[0.1, 0.2], [0.0, -0.1], [0.2, 0.1]],
                index=df.index,
                columns=["F1", "F2"],
            ),
            "loadings": pd.DataFrame(
                [[0.7], [0.1]],
                index=["num1", "num2"],
                columns=["F1"],
            ),
            "inertia": pd.Series([1.0], index=["F1"]),
        }
    }

    figs = generate_figures(
        factor_results,
        {},
        df,
        ["num1", "num2"],
        ["cat"],
        output_dir=tmp_path,
    )
    # scatter plot should still be produced
    assert "pca_scatter_2d" in figs
    # correlation plot cannot be generated with a single axis
    assert "pca_correlation" not in figs


def test_generate_figures_clusters(tmp_path):
    df = pd.DataFrame({
        "num1": [1, 2, 3, 4],
        "num2": [4, 3, 2, 1],
        "cat": ["a", "b", "a", "b"],
    })

    factor_results = {
        "pca": {
            "embeddings": pd.DataFrame(
                [[0.1, 0.2], [0.0, -0.1], [0.2, 0.1], [-0.2, -0.1]],
                index=df.index,
                columns=["F1", "F2"],
            )
        }
    }

    figs = generate_figures(
        factor_results,
        {},
        df,
        ["num1", "num2"],
        ["cat"],
        output_dir=tmp_path,
        cluster_k=2,
    )
    assert "pca_clusters" in figs
    out = tmp_path / "cluster.png"
    figs["pca_clusters"].savefig(out)
    assert out.exists()
