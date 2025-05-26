import ast
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import logging
from pathlib import Path
from typing import List, Optional, Tuple, Sequence, Any
import pytest


def load_run_pacmap(path='phase4v2.py'):
    """Load run_pacmap function from the given file without executing the rest."""
    source = open(path, 'r', encoding='utf-8').read()
    module = ast.parse(source, filename=path)
    namespace = {
        'pd': pd,
        'np': np,
        'logging': logging,
        'StandardScaler': StandardScaler,
        'OneHotEncoder': OneHotEncoder,
        'List': List,
        'Optional': Optional,
        'Tuple': Tuple,
        'Sequence': Sequence,
        'Any': Any,
        'Path': Path,
        'pacmap': None,
    }
    for node in module.body:
        if isinstance(node, ast.FunctionDef) and node.name == 'run_pacmap':
            func_code = ast.get_source_segment(source, node)
            exec(compile(func_code, path, 'exec'), namespace)
    return namespace['run_pacmap']


def test_run_pacmap_no_module():
    run_pacmap = load_run_pacmap()
    df = pd.DataFrame({
        'num1': [1, 2, 3],
        'num2': [3, 2, 1],
        'cat1': ['a', 'b', 'a']
    })
    model, emb = run_pacmap(df, ['num1', 'num2'], ['cat1'], Path('ignore'))
    assert model is None
    assert emb.empty
