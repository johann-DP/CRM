from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from joblib import Parallel, delayed


def _run_pipeline_single(config: Dict[str, Any], name: str) -> tuple[str, Dict[str, Any]]:
    """Helper for :func:`run_pipeline_parallel` executing a single dataset."""

    import phase4  # local import to avoid circular dependency

    cfg = dict(config)
    cfg["dataset"] = name
    if "output_dir" in cfg:
        base = Path(cfg["output_dir"])
        cfg["output_dir"] = str(base / name)
    # intermediate PDFs are no longer generated
    cfg.pop("output_pdf", None)
    result = phase4.run_pipeline(cfg)
    # Drop heavy figure objects to avoid pickling overhead
    if isinstance(result, dict) and "figures" in result:
        result.pop("figures", None)
    return name, result


def run_pipeline_parallel(
    config: Dict[str, Any],
    datasets: Sequence[str],
    *,
    n_jobs: Optional[int] = None,
    backend: str = "multiprocessing",
) -> Dict[str, Dict[str, Any]]:
    """Run :func:`phase4.run_pipeline` on several datasets in parallel."""

    n_jobs = n_jobs or len(datasets)
    with Parallel(n_jobs=n_jobs, backend=backend) as parallel:
        results = parallel(
            delayed(_run_pipeline_single)(config, ds) for ds in datasets
        )
    return dict(results)
