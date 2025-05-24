"""Launch all phase 4 fine tuning scripts in parallel.

Each script is executed with predefined paths. If the output directory
already contains files it is skipped to avoid reprocessing.
An ``OMP_NUM_THREADS`` value of ``1`` is set for each process so that
multiple jobs can fully utilise the available CPU cores.
"""

from __future__ import annotations

import os
import sys
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

BASE_DIR = Path(r"D:\DATAPREDICT\DATAPREDICT 2024\Missions\Digora")
PHASE1_CSV = BASE_DIR / "phase1_output" / "export_phase1_cleaned.csv"
PHASE2_CSV = BASE_DIR / "phase2_output" / "phase2_business_variables.csv"
PHASE3_MULTI = BASE_DIR / "phase3_output" / "phase3_cleaned_multivariate.csv"
PHASE3_UNIV = BASE_DIR / "phase3_output" / "phase3_cleaned_univ.csv"
PHASE4_DIR = BASE_DIR / "phase4_output"

class Job:
    def __init__(self, script: str, args: list[Path | str], out_dir: Path) -> None:
        self.script = script
        self.args = args
        self.out_dir = out_dir

    def already_done(self) -> bool:
        return self.out_dir.exists() and any(self.out_dir.iterdir())


JOBS: list[Job] = [
    Job(
        "fine_tune_famd.py",
        ["--input", PHASE3_MULTI, "--output", PHASE4_DIR / "fine_tune_famd"],
        PHASE4_DIR / "fine_tune_famd",
    ),
    Job(
        "fine_tuning_mca.py",
        [
            "--phase1",
            PHASE1_CSV,
            "--phase2",
            PHASE2_CSV,
            "--phase3",
            PHASE3_MULTI,
            "--output",
            PHASE4_DIR / "fine_tune_mca",
        ],
        PHASE4_DIR / "fine_tune_mca",
    ),
    Job(
        "fine_tune_mfa.py",
        ["--input", PHASE3_MULTI, "--output", PHASE4_DIR / "fine_tune_mfa"],
        PHASE4_DIR / "fine_tune_mfa",
    ),
    Job(
        "pacmap_fine_tune.py",
        ["--input", PHASE3_MULTI, "--output", PHASE4_DIR / "fine_tune_pacmap"],
        PHASE4_DIR / "fine_tune_pacmap",
    ),
    Job(
        "fine_tune_pca.py",
        ["--input", PHASE3_MULTI, "--output", PHASE4_DIR / "fine_tune_pca"],
        PHASE4_DIR / "fine_tune_pca",
    ),
    Job(
        "fine_tune_pcamix.py",
        ["--input", PHASE3_MULTI, "--output", PHASE4_DIR / "fine_tune_pcamix"],
        PHASE4_DIR / "fine_tune_pcamix",
    ),
    Job(
        "phase4_fine_tune_phate.py",
        [
            "--multi",
            PHASE3_MULTI,
            "--univ",
            PHASE3_UNIV,
            "--output",
            PHASE4_DIR / "fine_tune_phate",
        ],
        PHASE4_DIR / "fine_tune_phate",
    ),
    Job(
        "fine_tune_tsne.py",
        ["--input", PHASE3_MULTI, "--output", PHASE4_DIR / "fine_tune_tsne"],
        PHASE4_DIR / "fine_tune_tsne",
    ),
    Job(
        "fine_tuning_umap.py",
        ["--input", PHASE3_MULTI, "--output", PHASE4_DIR / "fine_tune_umap"],
        PHASE4_DIR / "fine_tune_umap",
    ),
]


def run_job(job: Job) -> tuple[str, int]:
    if job.already_done():
        return job.script, 0

    cmd = [sys.executable, str(Path(__file__).parent / job.script)]
    cmd += [str(a) for a in job.args]
    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", "1")
    proc = subprocess.run(cmd, env=env)
    return job.script, proc.returncode


def main() -> None:
    max_workers = min(os.cpu_count() or 1, len(JOBS))
    failed: list[str] = []
    with ThreadPoolExecutor(max_workers=max_workers) as exe:
        futures = {exe.submit(run_job, job): job for job in JOBS}
        for fut in as_completed(futures):
            script, ret = fut.result()
            if ret != 0:
                failed.append(script)

    if failed:
        print("Some scripts failed:", ", ".join(failed))
    else:
        print("All fine-tuning scripts completed or were skipped")


if __name__ == "__main__":
    main()
