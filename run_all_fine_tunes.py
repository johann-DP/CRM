#!/usr/bin/env python3
"""Run all fine tuning scripts in parallel."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import subprocess

# Mapping of fine-tuning scripts to their default arguments.
# These defaults ensure the launcher can be executed without passing
# any command-line parameters. Paths mirror those used in the Phase 4
# notebooks and utilities.
SCRIPTS: dict[str, list[str]] = {
    # FAMD and MCA rely on internal constants
    "fine_tune_famd.py": [],
    "fine_tuning_mca.py": [],
    # MFA requires a configuration file; skip if absent
    "fine_tune_mfa.py": ["--config", "config_mfa.yaml"],
    "pacmap_fine_tune.py": [],
    "fine_tune_pca.py": [],
    "fine_tune_pcamix.py": [],
    # PHATE explicitly needs the cleaned data paths
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

    available: dict[str, list[str]] = {}
    for script, args in SCRIPTS.items():
        if not Path(script).is_file():
            continue
        if script == "fine_tune_mfa.py" and not Path("config_mfa.yaml").is_file():
            logging.info("Skipping %s (missing config_mfa.yaml)", script)
            continue
        available[script] = args

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
