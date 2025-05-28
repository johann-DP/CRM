import pandas as pd
import phase4_functions as pf


def test_compare_datasets_versions_outlier(tmp_path):
    base = pd.DataFrame({
        "Code": range(1, 7),
        "num1": [1, 2, 3, 4, 5, 100],
        "num2": [5, 4, 3, 2, 1, 0],
        "cat": ["A", "A", "B", "B", "A", "B"],
        "Statut commercial": ["Gagné"] * 6,
    })
    clean = base[base["num1"] != 100]
    datasets = {"with_outlier": base, "clean": clean}

    result = pf.compare_datasets_versions(datasets, exclude_lost=False, output_dir=tmp_path)
    metrics = result["metrics"]

    assert set(metrics["dataset_version"]) == {"with_outlier", "clean"}
    pca_metrics = metrics[metrics["method"] == "pca"].set_index("dataset_version")
    assert pca_metrics.loc["clean", "variance_cumulee_%"] > pca_metrics.loc["with_outlier", "variance_cumulee_%"]

    for name in datasets:
        assert (tmp_path / name).is_dir()
