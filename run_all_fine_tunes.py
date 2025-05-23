#!/usr/bin/env python3
"""Run all fine tuning scripts in parallel."""

from __future__ import annotations

import logging
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import subprocess

# List of fine tuning scripts to execute for each algorithm
# Order follows the project's main Phase 4 scripts
SCRIPTS = [
    "fine_tune_famd.py",         # FAMD
    "fine_tuning_mca.py",        # MCA
    "fine_tune_mfa.py",          # MFA
    "pacmap_fine_tune.py",       # PaCMAP
    "fine_tune_pca.py",          # PCA
    "fine_tune_pcamix.py",       # PCAmix
    "phase4_fine_tune_phate.py", # PHATE
    "fine_tune_tsne.py",         # TSNE
    "fine_tuning_umap.py",       # UMAP
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run all fine tuning scripts")
    parser.add_argument("--phase3", required=True, help="Phase3 multivariate CSV")
    parser.add_argument("--phase1", help="Phase1 categorical CSV")
    parser.add_argument("--phase2", help="Phase2 categorical CSV")
    parser.add_argument("--univ", help="Phase3 univariate CSV (for PHATE)")
    parser.add_argument("--output", required=True, help="Base output directory")
    parser.add_argument("--config_mfa", help="Config file for MFA")
    return parser.parse_args()


def run_script(cmd: list[str]) -> tuple[str, int]:
    """Execute the command and return its exit code."""
    script = cmd[1]
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
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    base = Path(args.output)
    cmds: list[list[str]] = []

    if Path("fine_tune_famd.py").is_file():
        cmds.append(["python", "fine_tune_famd.py", "--input", args.phase3, "--output", str(base / "fine_tuning_famd")])

    if Path("fine_tuning_mca.py").is_file() and args.phase1 and args.phase2:
        cmds.append([
            "python", "fine_tuning_mca.py",
            "--phase1", args.phase1,
            "--phase2", args.phase2,
            "--phase3", args.phase3,
            "--output", str(base / "fine_tuning_mca"),
        ])

    if Path("fine_tune_mfa.py").is_file():
        mfa_cmd = ["python", "fine_tune_mfa.py", "--input", args.phase3, "--output", str(base / "fine_tuning_mfa")]
        if args.config_mfa:
            mfa_cmd = ["python", "fine_tune_mfa.py", "--config", args.config_mfa, "--input", args.phase3, "--output", str(base / "fine_tuning_mfa")]
        cmds.append(mfa_cmd)

    if Path("pacmap_fine_tune.py").is_file():
        cmds.append(["python", "pacmap_fine_tune.py", "--input", args.phase3, "--output", str(base / "fine_tuning_pacmap")])

    if Path("fine_tune_pca.py").is_file():
        cmds.append(["python", "fine_tune_pca.py", "--input", args.phase3, "--output", str(base / "fine_tuning_pca")])

    if Path("fine_tune_pcamix.py").is_file():
        cmds.append(["python", "fine_tune_pcamix.py", "--input", args.phase3, "--output", str(base / "fine_tuning_pcamix")])

    if Path("phase4_fine_tune_phate.py").is_file():
        phate_cmd = ["python", "phase4_fine_tune_phate.py", "--multi", args.phase3, "--output", str(base / "fine_tuning_phate")]
        if args.univ:
            phate_cmd.extend(["--univ", args.univ])
        cmds.append(phate_cmd)

    if Path("fine_tune_tsne.py").is_file():
        cmds.append(["python", "fine_tune_tsne.py", "--input", args.phase3, "--output", str(base / "fine_tuning_tsne")])

    if Path("fine_tuning_umap.py").is_file():
        cmds.append(["python", "fine_tuning_umap.py", "--input", args.phase3, "--output", str(base / "fine_tuning_umap")])

    if not cmds:
        logging.error("No fine tuning scripts found")
        return

    with ThreadPoolExecutor(max_workers=len(cmds)) as executor:
        futures = {executor.submit(run_script, c): c[1] for c in cmds}
        for future in as_completed(futures):
            script, code = future.result()
            if code == 0:
                logging.info("%s completed successfully", script)
            else:
                logging.warning("%s exited with code %d", script, code)


if __name__ == "__main__":
    main()
