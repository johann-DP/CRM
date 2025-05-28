import pandas as pd
import numpy as np
import phase4_functions as pf
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
