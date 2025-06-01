#!/usr/bin/env python3
"""Run all analysis steps sequentially or in parallel.

This helper executes Phase 2, Phase 3 and Phase 4 of the CRM analysis
using the configuration file of the project.  Phase 4 can run the
requested datasets sequentially or in parallel depending on the
``--jobs`` argument.

Usage::

    python run_all_since_commit.py [--config config.yaml] [--jobs N]

All generated files are stored in the output directory configured in
``config.yaml``.
"""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def run(cmd: list[str]) -> None:
    """Run ``cmd`` and forward output."""
    print("$", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Execute all analysis steps")
    p.add_argument("--config", default="config.yaml", help="Configuration file")
    p.add_argument("--jobs", type=int, default=1, help="Parallel jobs for phase4")
    args = p.parse_args(argv)

    cfg = Path(args.config)
    if not cfg.exists():
        raise FileNotFoundError(cfg)

    # Phase 2 and 3 rely on paths hard-coded in the config.
    run(["python", "phase2.py"])
    run(["python", "phase3.py"])

    # Phase 4 supports parallel execution across datasets
    cmd = ["python", "phase4.py", "--config", str(cfg), "--dataset-jobs", str(args.jobs)]
    run(cmd)

    # Build the consolidated PDF report from the generated images
    run(["python", "generate_phase4_report.py", "--config", str(cfg)])

    print("Results are saved under the directory specified by 'output_dir' in the config file.")


if __name__ == "__main__":  # pragma: no cover - CLI helper
    main()
