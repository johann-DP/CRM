import ast
import pandas as pd
import logging
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from typing import List, Tuple, Any
import phate


def load_run_phate(path="phase4v2.py"):
    """Load run_phate function from file without executing rest."""
    source = open(path, "r", encoding="utf-8").read()
    module = ast.parse(source, filename=path)
    namespace = {
        "pd": pd,
        "StandardScaler": StandardScaler,
        "OneHotEncoder": OneHotEncoder,
        "logging": logging,
        "np": np,
        "phate": phate,
        "List": List,
        "Tuple": Tuple,
        "Any": Any,
        "Path": Path,
    }
    for node in module.body:
        if isinstance(node, ast.FunctionDef) and node.name == "run_phate":
            func_code = ast.get_source_segment(source, node)
            exec(compile(func_code, path, "exec"), namespace)
    if "run_phate" in namespace:
        return namespace["run_phate"]
    raise RuntimeError("run_phate not found")


def main():
    run_phate = load_run_phate()
    df = pd.DataFrame({
        "num1": [1, 2, 3, 4, 5],
        "num2": [5, 4, 3, 2, 1],
        "cat1": ["a", "b", "a", "b", "a"],
    })
    op, coords = run_phate(
        df,
        ["num1", "num2"],
        ["cat1"],
        Path("ignore"),
        n_components=2,
    )
    print("Coords shape", coords.shape)


if __name__ == "__main__":
    main()
