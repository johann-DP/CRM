import importlib
import warnings

import phase4
import phase4_functions as pf


def _has_tight_layout_filter():
    return any(
        f[0] == 'ignore'
        and getattr(f[1], 'pattern', str(f[1])).startswith('Tight layout not applied')
        for f in warnings.filters
    )


def test_tight_layout_filter_phase4(monkeypatch):
    monkeypatch.delenv('OPENBLAS_NUM_THREADS', raising=False)
    importlib.reload(phase4)
    assert _has_tight_layout_filter()


def test_tight_layout_filter_pf():
    importlib.reload(pf)
    assert _has_tight_layout_filter()
