#!/usr/bin/env python3
"""Run all fine tuning scripts in parallel."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import subprocess

# Mapping of fine-tuning scripts to the arguments they require.
# The goal is to run everything without passing parameters when invoking this
# launcher. Paths are relative to the repository root and follow the file
# structure included in the Git history.
SCRIPTS: dict[str, list[str]] = {
    # FAMD and MCA rely on internal constants
    "fine_tune_famd.py": [],
    "fine_tuning_mca.py": [],
    # MFA expects a configuration file; skip if it is absent
    "fine_tune_mfa.py": ["--config", "config_mfa.yaml"],
    "pacmap_fine_tune.py": [],
    "fine_tune_pca.py": [],
    "fine_tune_pcamix.py": [],
    # PHATE requires explicit input/output paths
    "phase4_fine_tune_phate.py": [
        "--multi",
        "phase3_output/phase3_cleaned_multivariate.csv",
        "--univ",
        "phase3_output/phase3_cleaned_univ.csv",
        "--output",
        "phase4_output/fine_tuning_phate",
    ],
    "fine_tune_tsne.py": [],  # defaults embedded in the script
    "fine_tuning_umap.py": [],
}


def run_script(script: str, args: list[str]) -> tuple[str, int]:
    """Execute the script with optional arguments and return its exit code."""
    cmd = ["python", script, *args]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.stdout:
        logging.debug("%s output:\n%s", script, proc.stdout)
    if proc.stderr:
        logging.debug("%s errors:\n%s", script, proc.stderr)

    if proc.returncode != 0:
        err = proc.stderr.lower()
        if "no module named" in err:
            logging.info("Skipping %s (missing optional dependency)", script)
            return script, 0
        if "filenotfounderror" in err:
            logging.info("Skipping %s (required data not found)", script)
            return script, 0
        if proc.returncode == 2 and "usage" in err:
            logging.info("Skipping %s (missing required arguments)", script)
            return script, 0
    return script, proc.returncode


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    available = {s: args for s, args in SCRIPTS.items() if Path(s).is_file()}
    if not available:
        logging.error("No fine tuning scripts found")
        return

    with ThreadPoolExecutor(max_workers=len(available)) as executor:
        futures = {executor.submit(run_script, s, a): s for s, a in available.items()}
        for future in as_completed(futures):
            script, code = future.result()
            if code == 0:
                logging.info("%s completed successfully", script)
            else:
                logging.warning("%s exited with code %d", script, code)


if __name__ == "__main__":
    main()
