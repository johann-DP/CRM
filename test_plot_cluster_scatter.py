import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
from visualization import plot_cluster_scatter


def test_plot_cluster_scatter_basic():
    emb = pd.DataFrame({"x": [0, 1, 0, 1], "y": [0, 0, 1, 1]})
    labels = np.array([0, 0, 1, 1])
    fig = plot_cluster_scatter(emb, labels, "test")
    assert hasattr(fig, "savefig")

