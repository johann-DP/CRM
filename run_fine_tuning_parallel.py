import sys
import os
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE_DIR = Path(r"D:\DATAPREDICT\DATAPREDICT 2024\Missions\Digora")
PHASE1_CSV = BASE_DIR / "phase1_output" / "export_phase1_cleaned.csv"
PHASE2_CSV = BASE_DIR / "phase2_output" / "phase2_business_variables.csv"
PHASE3_MULTI = BASE_DIR / "phase3_output" / "phase3_cleaned_multivariate.csv"
PHASE3_UNIV = BASE_DIR / "phase3_output" / "phase3_cleaned_univ.csv"
PHASE4_DIR = BASE_DIR / "phase4_output"

SCRIPTS = [
    (
        "fine_tune_famd.py",
        PHASE4_DIR / "fine_tune_famd",
        ["--input", PHASE3_MULTI, "--output", PHASE4_DIR / "fine_tune_famd"],
    ),
    (
        "fine_tuning_mca.py",
        PHASE4_DIR / "fine_tune_mca",
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
    ),
    (
        "fine_tune_mfa.py",
        PHASE4_DIR / "fine_tune_mfa",
        ["--input", PHASE3_MULTI, "--output", PHASE4_DIR / "fine_tune_mfa"],
    ),
    (
        "pacmap_fine_tune.py",
        PHASE4_DIR / "fine_tune_pacmap",
        ["--input", PHASE3_MULTI, "--output", PHASE4_DIR / "fine_tune_pacmap"],
    ),
    (
        "fine_tune_pca.py",
        PHASE4_DIR / "fine_tune_pca",
        ["--input", PHASE3_MULTI, "--output", PHASE4_DIR / "fine_tune_pca"],
    ),
    (
        "fine_tune_pcamix.py",
        PHASE4_DIR / "fine_tune_pcamix",
        ["--input", PHASE3_MULTI, "--output", PHASE4_DIR / "fine_tune_pcamix"],
    ),
    (
        "phase4_fine_tune_phate.py",
        PHASE4_DIR / "fine_tune_phate",
        [
            "--multi",
            PHASE3_MULTI,
            "--univ",
            PHASE3_UNIV,
            "--output",
            PHASE4_DIR / "fine_tune_phate",
        ],
    ),
    (
        "fine_tune_tsne.py",
        PHASE4_DIR / "fine_tune_tsne",
        ["--input", PHASE3_MULTI, "--output", PHASE4_DIR / "fine_tune_tsne"],
    ),
    (
        "fine_tuning_umap.py",
        PHASE4_DIR / "fine_tune_umap",
        ["--input", PHASE3_MULTI, "--output", PHASE4_DIR / "fine_tune_umap"],
    ),
]


def run_script(script: str, args: list[Path | str]) -> int:
    """Run a fine-tuning script and return its exit code."""
    cmd = [sys.executable, str(Path(__file__).parent / script)]
    cmd += [str(a) for a in args]
    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", "1")
    result = subprocess.run(cmd, env=env)
    return result.returncode


def main() -> None:
    max_workers = min(len(SCRIPTS), os.cpu_count() or 1)
    failed: list[str] = []
    with ThreadPoolExecutor(max_workers=max_workers) as exe:
        future_to_script = {}
        for script, out_dir, args in SCRIPTS:
            if out_dir.exists() and any(out_dir.iterdir()):
                print(f"Skipping {script}: output already exists")
                continue
            out_dir.mkdir(parents=True, exist_ok=True)
            future = exe.submit(run_script, script, args)
            future_to_script[future] = script

        for future in as_completed(future_to_script):
            script = future_to_script[future]
            try:
                ret = future.result()
            except Exception:
                ret = 1
            if ret != 0:
                failed.append(script)

    if failed:
        print("Some scripts failed:", ", ".join(failed))
    else:
        print("All fine-tuning scripts completed successfully")


if __name__ == "__main__":
    main()
