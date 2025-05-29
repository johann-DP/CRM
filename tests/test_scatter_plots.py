import pandas as pd
import numpy as np
import phase4_functions as pf


def test_generate_scatter_plots(tmp_path):
    emb = pd.DataFrame(np.random.rand(5, 3), columns=["C1", "C2", "C3"])
    out = pf.generate_scatter_plots(emb, "ds", "pca", tmp_path)
    # expect baseline and three clustering methods (2D/3D)
    assert set(out) >= {
        "no_cluster_2d",
        "kmeans_2d",
        "agglomerative_2d",
        "gmm_2d",
    }
    for path in out.values():
        assert path.exists() and path.stat().st_size > 0
