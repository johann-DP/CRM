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


def test_set_blas_threads_caps_openblas(monkeypatch):
    monkeypatch.delenv('OPENBLAS_NUM_THREADS', raising=False)
    set_blas_threads(64)
    assert os.environ['OPENBLAS_NUM_THREADS'] == '24'


def test_set_blas_threads_uses_threadpoolctl(monkeypatch):
    calls = []

    def fake_limits(n):
        calls.append(n)

    monkeypatch.setattr('phase4.threadpool_limits', fake_limits)
    set_blas_threads(8)
    assert calls == [8]
