import pandas as pd
import matplotlib
matplotlib.use("Agg")

from visualization import generate_figures


def test_generate_figures_basic():
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

    figs = generate_figures(factor_results, nonlin_results, df, ["num1", "num2"], ["cat"])
    assert "pca_correlation" in figs
    assert "pca_scatter_2d" in figs
    assert "umap_scatter_2d" in figs
    for f in figs.values():
        assert hasattr(f, "savefig")
