import os
from phase4 import set_blas_threads


def test_set_blas_threads_sets_env(monkeypatch):
    for var in [
        'OPENBLAS_NUM_THREADS',
        'MKL_NUM_THREADS',
        'OMP_NUM_THREADS',
    ]:
        monkeypatch.delenv(var, raising=False)
    set_blas_threads(3)
    assert os.environ['OPENBLAS_NUM_THREADS'] == '3'
    assert os.environ['MKL_NUM_THREADS'] == '3'
    assert os.environ['OMP_NUM_THREADS'] == '3'

