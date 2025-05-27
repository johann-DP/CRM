import pytest
import ast
import pandas as pd
import prince
from sklearn.preprocessing import StandardScaler
import logging
import numpy as np
from typing import List, Optional, Tuple


def load_run_famd(path='phase4v2.py'):
    """Load only the run_famd function from the given file without executing the rest."""
    source = open(path, 'r', encoding='utf-8').read()
    module = ast.parse(source, filename=path)
    namespace = {
        'pd': pd,
        'prince': prince,
        'StandardScaler': StandardScaler,
        'logging': logging,
        'np': np,
        'List': List,
        'Optional': Optional,
        'Tuple': Tuple,
    }
    for node in module.body:
        if isinstance(node, ast.FunctionDef) and node.name in {'get_explained_inertia', 'run_famd'}:
            func_code = ast.get_source_segment(source, node)
            exec(compile(func_code, path, 'exec'), namespace)
    if 'run_famd' in namespace:
        return namespace['run_famd']
    raise RuntimeError('run_famd not found')


def test_run_famd_basic():
    run_famd = load_run_famd()
    df = pd.DataFrame({
        'num1': [1, 2, 3, 4, 5],
        'num2': [5, 4, 3, 2, 1],
        'cat1': ['a', 'b', 'a', 'b', 'a']
    })
    famd, inertia, rows, cols, contrib = run_famd(df, ['num1', 'num2'], ['cat1'], n_components=2)
    assert rows.shape == (5, 2)
    assert cols.shape == (3, 2)
    assert contrib.shape == (3, 2)
    assert pytest.approx(float(inertia.sum()), rel=1e-6) == 1.0


def test_run_famd_with_datetimes():
    run_famd = load_run_famd()
    df = pd.DataFrame({
        'num': [1, 2, 3, 4],
        'date': pd.to_datetime(['2020-01-01', '2020-01-02', '2020-01-03', '2020-01-04']),
        'cat': ['a', 'b', 'a', 'b'],
    })
    famd, inertia, rows, cols, contrib = run_famd(df, ['num'], ['date', 'cat'], n_components=2)
    assert rows.shape[0] == 4

