import ast
import pandas as pd
import prince
from sklearn.preprocessing import StandardScaler
import logging
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Optional, Tuple
import pytest


def load_run_pcamix(path='phase4v2.py'):
    """Load run_pcamix function from file without executing rest."""
    source = open(path, 'r', encoding='utf-8').read()
    module = ast.parse(source, filename=path)
    namespace = {
        'pd': pd,
        'prince': prince,
        'StandardScaler': StandardScaler,
        'logging': logging,
        'np': np,
        'plt': plt,
        'List': List,
        'Optional': Optional,
        'Tuple': Tuple,
        'Path': Path,
    }
    for node in module.body:
        if isinstance(node, ast.FunctionDef) and node.name in {'get_explained_inertia', 'run_pcamix'}:
            func_code = ast.get_source_segment(source, node)
            exec(compile(func_code, path, 'exec'), namespace)
    if 'run_pcamix' in namespace:
        return namespace['run_pcamix']
    raise RuntimeError('run_pcamix not found')


def test_run_pcamix_basic(tmp_path: Path):
    run_pcamix = load_run_pcamix()
    df = pd.DataFrame({
        'num1': [1, 2, 3, 4, 5],
        'num2': [5, 4, 3, 2, 1],
        'cat1': ['a', 'b', 'a', 'b', 'a']
    })
    model, inertia, rows, cols = run_pcamix(
        df,
        ['num1', 'num2'],
        ['cat1'],
        output_dir=tmp_path,
        n_components=None,
        optimize=False,
    )
    assert rows.shape[0] == 5
    assert rows.shape[1] == cols.shape[1]
    assert cols.shape[0] == 3
    assert pytest.approx(float(inertia.sum()), rel=1e-6) == 1.0
