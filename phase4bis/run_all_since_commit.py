#!/usr/bin/env python3
"""Run all analysis utilities located in this directory.

The helper discovers every Python script in the ``phase4bis`` folder and
executes them sequentially or in parallel when ``--jobs`` is greater than
one.  Some scripts need an example CSV file; ``sample_dataset.csv`` from
this directory is provided for this purpose.

Usage::

    python -m phase4bis.run_all_since_commit \
        [--config config.yaml] [--jobs N]

Results of the executed scripts are written either in the working
directory or under the ``output_dir`` defined in ``config.yaml``.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import subprocess
from pathlib import Path
from typing import Any, Mapping
import sys

import yaml


OUTPUT_DIR = Path.cwd()


def run(cmd: list[str]) -> bool:
    """Run ``cmd`` in :data:`OUTPUT_DIR` and return ``True`` on success."""
    print("$", " ".join(cmd))
    completed = subprocess.run(cmd, cwd=OUTPUT_DIR)
    if completed.returncode != 0:
        print(f"Command failed with exit code {completed.returncode}")
        return False
    return True




def _needs_config(path: Path) -> bool:
    """Return True if ``path`` CLI accepts a ``--config`` option."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return False
    return "--config" in text


def _scripts_in_dir(folder: Path) -> list[Path]:
    """Return Python scripts located in ``folder`` sorted by name."""
    return sorted(
        p
        for p in folder.glob("*.py")
        if p.name != Path(__file__).name
    )


_SAMPLE = Path(__file__).resolve().parent / "sample_dataset.csv"

_EXTRA_ARGS: dict[str, list[str]] = {
    "clustering_quality_indices.py": [str(_SAMPLE)],
    "export_pca_inertias.py": [str(_SAMPLE)],
    "famd_full_analysis.py": [str(_SAMPLE)],
    "famd_cos2_heatmap.py": [str(_SAMPLE)],
    "famd_individuals.py": [str(_SAMPLE)],
    "compare_umap_clusters.py": ["--raw", str(_SAMPLE), "--prepared", str(_SAMPLE)],
}


def _command(path: Path, config: Path) -> list[str] | None:
    """Return command list to execute ``path`` or ``None`` to skip."""
    if path.name == "cluster_confusion_heatmap.py":
        # This script expects two clustering label files which are not
        # available in the repository; skip automatic execution.
        return None

    cmd = [sys.executable, str(path.resolve())]
    if _needs_config(path):
        cmd += ["--config", str(config)]
    extra = _EXTRA_ARGS.get(path.name)
    if extra:
        cmd += extra
    return cmd


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Run all utilities from phase4bis")
    p.add_argument("--config", default="config.yaml", help="Configuration file")
    p.add_argument("--jobs", type=int, default=12, help="Number of parallel scripts")
    args = p.parse_args(argv)

    cfg_path = Path(args.config)

    folder = Path(__file__).resolve().parent
    scripts = [_command(pth, cfg_path) for pth in _scripts_in_dir(folder)]
    scripts = [cmd for cmd in scripts if cmd]

    if not scripts:
        print(f"No scripts found in {folder}")
        return

    global OUTPUT_DIR

    def _load_config(path: Path) -> Mapping[str, Any]:
        with open(path, "r", encoding="utf-8") as fh:
            if path.suffix.lower() in {".yaml", ".yml"}:
                return yaml.safe_load(fh)
            return json.load(fh)

    cfg = _load_config(cfg_path)
    OUTPUT_DIR = Path(cfg.get("output_dir", "."))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results: list[bool]
    if args.jobs > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as exe:
            results = list(exe.map(run, scripts))
    else:
        results = [run(cmd) for cmd in scripts]

    successes = sum(results)
    failures = len(results) - successes
    print(f"{successes} script(s) succeeded, {failures} failed")

    print(
        "Results are available either in the current directory or under "
        "the 'output_dir' configured in config.yaml."
    )


if __name__ == "__main__":  # pragma: no cover - CLI helper
    main()
