import phase4


def test_method_params_merging():
    cfg = {
        "umap": {"n_neighbors": 50},
        "umap_min_dist": 0.2,
        "pacmap_MN_ratio": 0.7,
    }
    umap_params = phase4._method_params("umap", cfg)
    assert umap_params["n_neighbors"] == 50
    assert umap_params["min_dist"] == 0.2

    pacmap_params = phase4._method_params("pacmap", cfg)
    assert pacmap_params["MN_ratio"] == 0.7
    assert "n_neighbors" in pacmap_params  # default from BEST_PARAMS

