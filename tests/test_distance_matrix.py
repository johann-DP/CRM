import pandas as pd
import generate_distance_matrix as gdm


def test_compute_distance_matrix_basic():
    df = pd.DataFrame({
        "num1": [0, 1],
        "cat": ["a", "b"],
    }, index=[10, 20])
    quant_vars = ["num1"]
    qual_vars = ["cat"]
    mat = gdm.compute_distance_matrix(df, quant_vars, qual_vars)
    assert list(mat.index) == [10, 20]
    assert list(mat.columns) == [10, 20]
    assert mat.loc[10, 10] == 0
    assert mat.loc[20, 20] == 0
    assert mat.loc[10, 20] == mat.loc[20, 10] != 0
