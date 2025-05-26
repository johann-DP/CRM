import ast
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import logging
import umap
from typing import Any, Dict


def load_run_umap(path="nonlinear_methods.py"):
    """Load run_umap from file without executing the rest."""
    source = open(path, "r", encoding="utf-8").read()
    module = ast.parse(source, filename=path)
    namespace = {
        "pd": pd,
        "np": np,
        "StandardScaler": StandardScaler,
        "OneHotEncoder": OneHotEncoder,
        "logging": logging,
        "umap": umap,
        "Dict": Dict,
        "Any": Any,
        "time": __import__("time"),
    }
    for node in module.body:
        if isinstance(node, ast.FunctionDef) and node.name in {"_to_numeric_matrix", "run_umap"}:
            code = ast.get_source_segment(source, node)
            exec(compile(code, path, "exec"), namespace)
    return namespace["run_umap"]


def test_run_umap_with_random_state():
    run_umap = load_run_umap()
    df = pd.DataFrame({
        "num1": [1, 2, 3, 4, 5],
        "num2": [5, 4, 3, 2, 1],
        "cat1": ["a", "b", "a", "b", "a"],
    })
    result = run_umap(df, random_state=0)
    emb = result["embeddings"]
    assert emb.shape == (5, 2)
