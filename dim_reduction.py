"""Wrappers for dimensionality reduction methods."""

from __future__ import annotations

from typing import Dict, Any, List, Sequence, Optional

from block4_factor_methods import (
    run_pca,
    run_mca,
    run_famd,
    run_mfa,
)
from nonlinear_methods import run_umap, run_phate, run_pacmap


def run_all_factor_methods(
    df_active: Any,
    quant_vars: List[str],
    qual_vars: List[str],
    *,
    groups: Optional[Sequence[Sequence[str]]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Execute PCA, MCA, FAMD and optionally MFA."""
    results = {
        "PCA": run_pca(df_active, quant_vars),
        "MCA": run_mca(df_active, qual_vars),
        "FAMD": run_famd(df_active, quant_vars, qual_vars),
    }
    if groups is not None:
        results["MFA"] = run_mfa(df_active, groups)
    return results


def run_all_nonlin(df_active: Any) -> Dict[str, Dict[str, Any]]:
    """Run UMAP, PHATE and PaCMAP on the given dataset."""
    results: Dict[str, Dict[str, Any]] = {}
    for name, func in (("UMAP", run_umap), ("PHATE", run_phate), ("PACMAP", run_pacmap)):
        try:
            results[name] = func(df_active)
        except Exception as exc:  # pragma: no cover - missing deps
            results[name] = {"error": str(exc)}
    return results
