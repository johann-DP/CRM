#!/usr/bin/env python3
"""Run all fine tuning scripts in parallel."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import subprocess

# List of fine tuning scripts to execute for each algorithm
# Order follows the project's main Phase 4 scripts
SCRIPTS = [
    "phase4_famd.py",        # FAMD
    "fine_tuning_mca.py",    # MCA
    "fine_tuning_mfa.py",    # MFA
    "pacmap_fine_tune.py",   # PaCMAP
    "fine_tune_pca.py",      # PCA
    "phase4_pcamix.py",      # PCAmix
    "phase4_phate.py",       # PHATE
    "fine_tune_tsne.py",     # TSNE
    "fine_tuning_umap.py",   # UMAP
]


def run_script(script: str) -> tuple[str, int]:
    """Execute the script and return its exit code."""
    cmd = ["python", script]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.stdout:
        logging.debug("%s output:\n%s", script, proc.stdout)
    if proc.stderr:
        logging.debug("%s errors:\n%s", script, proc.stderr)
    return script, proc.returncode


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    available = [s for s in SCRIPTS if Path(s).is_file()]
    if not available:
        logging.error("No fine tuning scripts found")
        return

    with ThreadPoolExecutor(max_workers=len(available)) as executor:
        futures = {executor.submit(run_script, s): s for s in available}
        for future in as_completed(futures):
            script, code = future.result()
            if code == 0:
                logging.info("%s completed successfully", script)
            else:
                logging.warning("%s exited with code %d", script, code)


if __name__ == "__main__":
    main()
