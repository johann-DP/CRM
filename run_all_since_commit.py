#!/usr/bin/env python3
"""Run every script introduced since a given commit.

The helper inspects the Git history and executes all Python scripts that
appeared after the specified base commit.  Scripts are executed
concurrently when ``--jobs`` is greater than one.  A few utilities expect
an input CSV file; ``sample_dataset.csv`` from the repository root is
used for these.

Usage::

    python run_all_since_commit.py [--since b362e454] [--config config.yaml]
                                  [--jobs N]

Results of the executed scripts are written either in the working
directory or under the ``output_dir`` defined in ``config.yaml``.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import subprocess
from pathlib import Path


def run(cmd: list[str]) -> bool:
    """Run ``cmd`` and return ``True`` on success."""
    print("$", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        print(f"Command failed with exit code {exc.returncode}")
        return False
    return True


def _new_scripts(base: str) -> list[Path]:
    """Return Python files added after ``base``."""
    out = subprocess.check_output(
        ["git", "diff", "--name-status", "--diff-filter=A", f"{base}..HEAD"],
        text=True,
    )
    paths: list[Path] = []
    for line in out.splitlines():
        _status, name = line.split("\t", 1)
        if not name.endswith(".py"):
            continue
        if name.startswith("tests/"):
            continue
        if name == Path(__file__).name:
            continue
        paths.append(Path(name))
    return paths


def _needs_config(path: Path) -> bool:
    """Return True if ``path`` CLI accepts a ``--config`` option."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return False
    return "--config" in text


_SAMPLE = Path("sample_dataset.csv")

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

    cmd = ["python", str(path)]
    if _needs_config(path):
        cmd += ["--config", str(config)]
    extra = _EXTRA_ARGS.get(path.name)
    if extra:
        cmd += extra
    return cmd


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Run scripts added since a commit")
    p.add_argument("--since", default="b362e454", help="Base commit")
    p.add_argument("--config", default="config.yaml", help="Configuration file")
    p.add_argument("--jobs", type=int, default=1, help="Number of parallel scripts")
    args = p.parse_args(argv)

    cfg = Path(args.config)

    scripts = [_command(pth, cfg) for pth in _new_scripts(args.since)]
    scripts = [cmd for cmd in scripts if cmd]

    if not scripts:
        print("No new scripts since", args.since)
        return

    if args.jobs > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as exe:
            results = list(exe.map(run, scripts))
    else:
        results = [run(cmd) for cmd in scripts]

    failures = len([r for r in results if not r])
    if failures:
        print(f"{failures} script(s) failed")
    else:
        print("All scripts completed successfully")

    print("Results are available either in the current directory or under" " the 'output_dir' configured in config.yaml.")


if __name__ == "__main__":  # pragma: no cover - CLI helper
    main()
