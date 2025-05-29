import importlib
import warnings

import phase4
import phase4_functions as pf

def _reload():
    warnings.resetwarnings()
    importlib.reload(phase4)
    importlib.reload(pf)

def _has_filter(substring):
    for action, message, *_ in warnings.filters:
        pattern = getattr(message, 'pattern', message)
        if action == 'ignore' and pattern and substring in str(pattern):
            return True
    return False


def test_openpyxl_warning_suppressed():
    _reload()
    assert _has_filter('Workbook contains no default style')


def test_tight_layout_warning_suppressed():
    _reload()
    assert _has_filter('Tight layout not applied')
