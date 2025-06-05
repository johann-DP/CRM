"""Backward compatible wrapper for lead scoring feature engineering."""

from pred_lead_scoring import feature_engineering as _fe

# local requests object to allow monkeypatching
requests = _fe.requests


def _call_with_patched_requests(func, *args, **kwargs):
    """Call ``func`` while ensuring it uses this module's ``requests`` object."""
    orig = _fe.requests
    _fe.requests = requests
    try:
        return func(*args, **kwargs)
    finally:
        _fe.requests = orig


def create_internal_features(*args, **kwargs):
    return _call_with_patched_requests(_fe.create_internal_features, *args, **kwargs)


def reduce_categorical_levels(*args, **kwargs):
    return _fe.reduce_categorical_levels(*args, **kwargs)


def enrich_with_sirene(*args, **kwargs):
    return _call_with_patched_requests(_fe.enrich_with_sirene, *args, **kwargs)


def enrich_with_geo_data(*args, **kwargs):
    return _call_with_patched_requests(_fe.enrich_with_geo_data, *args, **kwargs)


def advanced_feature_engineering(*args, **kwargs):
    return _call_with_patched_requests(_fe.advanced_feature_engineering, *args, **kwargs)


def clear_caches(*args, **kwargs):
    return _fe.clear_caches(*args, **kwargs)


__all__ = [
    "create_internal_features",
    "reduce_categorical_levels",
    "enrich_with_sirene",
    "enrich_with_geo_data",
    "advanced_feature_engineering",
    "clear_caches",
]

