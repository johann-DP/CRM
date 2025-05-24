import sys
import subprocess
from pathlib import Path

BASE_DIR = Path(r"D:\DATAPREDICT\DATAPREDICT 2024\Missions\Digora")
PHASE1_CSV = BASE_DIR / "phase1_output" / "export_phase1_cleaned.csv"
PHASE2_CSV = BASE_DIR / "phase2_output" / "phase2_business_variables.csv"
PHASE3_MULTI = BASE_DIR / "phase3_output" / "phase3_cleaned_multivariate.csv"
PHASE3_UNIV = BASE_DIR / "phase3_output" / "phase3_cleaned_univ.csv"
PHASE4_DIR = BASE_DIR / "phase4_output"

SCRIPTS = [
    (
        "fine_tune_famd.py",
        ["--input", PHASE3_MULTI, "--output", PHASE4_DIR / "fine_tune_famd"],
    ),
    (
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
    ),
    (
        "fine_tune_mfa.py",
        ["--input", PHASE3_MULTI, "--output", PHASE4_DIR / "fine_tune_mfa"],
    ),
    (
        "pacmap_fine_tune.py",
        ["--input", PHASE3_MULTI, "--output", PHASE4_DIR / "fine_tune_pacmap"],
    ),
    (
        "fine_tune_pca.py",
        ["--input", PHASE3_MULTI, "--output", PHASE4_DIR / "fine_tune_pca"],
    ),
    (
        "fine_tune_pcamix.py",
        ["--input", PHASE3_MULTI, "--output", PHASE4_DIR / "fine_tune_pcamix"],
    ),
    (
        "phase4_fine_tune_phate.py",
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
        ["--input", PHASE3_MULTI, "--output", PHASE4_DIR / "fine_tune_tsne"],
    ),
    (
        "fine_tuning_umap.py",
        ["--input", PHASE3_MULTI, "--output", PHASE4_DIR / "fine_tune_umap"],
    ),
]


def run_script(script: str, args: list[Path | str]) -> subprocess.Popen:
    cmd = [sys.executable, str(Path(__file__).parent / script)]
    cmd += [str(a) for a in args]
    return subprocess.Popen(cmd)


def main() -> None:
    processes: list[tuple[str, subprocess.Popen]] = []
    for script, args in SCRIPTS:
        proc = run_script(script, args)
        processes.append((script, proc))

    failed = []
    for script, proc in processes:
        ret = proc.wait()
        if ret != 0:
            failed.append(script)

    if failed:
        print("Some scripts failed:", ", ".join(failed))
    else:
        print("All fine-tuning scripts completed successfully")


if __name__ == "__main__":
    main()
