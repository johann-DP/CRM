import pandas as pd
import numpy as np
from block4_factor_methods import run_mca


def test_run_mca_handles_missing():
    df = pd.DataFrame({
        "cat1": ["a", "b", np.nan, "c", np.inf],
        "cat2": ["x", np.inf, "y", "x", "y"],
    })
    result = run_mca(df, ["cat1", "cat2"])
    emb = result["embeddings"]
    assert emb.shape[0] == len(df)
