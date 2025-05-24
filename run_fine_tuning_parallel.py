import sys
import os
import sys
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


def _has_results(out_dir: Path) -> bool:
    """Return True if the given directory exists and contains any file."""
    return out_dir.exists() and any(out_dir.iterdir())


def run_script(script: str, args: list[Path | str]) -> int:
    """Run a fine‑tuning script and return its exit code."""
    cmd = [sys.executable, str(Path(__file__).parent / script)]
    cmd += [str(a) for a in args]
    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", "1")
    result = subprocess.run(cmd, env=env)
    return result.returncode


def main() -> None:
    to_run: list[tuple[str, list[Path | str]]] = []
    for script, args in SCRIPTS:
        if "--output" in args:
            out_idx = args.index("--output") + 1
            out = Path(args[out_idx])
            if _has_results(out):
                print(f"Skipping {script}: output already exists")
                continue
        to_run.append((script, args))

    max_workers = min(len(to_run), os.cpu_count() or 1)
    failed: list[str] = []
    with ThreadPoolExecutor(max_workers=max_workers) as exe:
        futures = {exe.submit(run_script, s, a): s for s, a in to_run}
        for fut in as_completed(futures):
            script = futures[fut]
            try:
                ret = fut.result()
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
