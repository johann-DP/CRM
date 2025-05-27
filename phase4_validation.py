import argparse
import hashlib
import importlib
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def run(cmd, cwd: Path) -> subprocess.CompletedProcess:
    """Run a command and return the completed process."""
    logging.info("Running command: %s", " ".join(cmd))
    return subprocess.run(cmd, cwd=cwd, text=True, capture_output=True)


def assert_returncode_zero(proc: subprocess.CompletedProcess, msg: str) -> None:
    if proc.returncode != 0:
        raise AssertionError(f"{msg}\nSTDOUT: {proc.stdout}\nSTDERR: {proc.stderr}")


def check_importability() -> None:
    try:
        importlib.import_module("phase4v3")
    except Exception as exc:
        raise AssertionError(f"Import phase4v3 failed: {exc}") from exc


def check_functions() -> None:
    mod = importlib.import_module("phase4v3")
    required = [
        "load_datasets",
        "prepare_data",
        "select_variables",
        "run_pca",
        "run_famd",
        "run_umap",
        "generate_figures",
        "evaluate_methods",
        "export_report_to_pdf",
    ]
    missing = [f for f in required if not callable(getattr(mod, f, None))]
    if missing:
        raise AssertionError(f"Fonctions manquantes/non callables: {', '.join(missing)}")


def check_lint(repo: Path) -> None:
    proc = run([sys.executable, "-m", "py_compile", "phase4v3.py"], cwd=repo)
    assert_returncode_zero(proc, "py_compile failed")
    proc = run(["flake8", "phase4v3.py", "--ignore=E203,W503"], cwd=repo)
    assert_returncode_zero(proc, "flake8 failed")


def check_pytest(repo: Path) -> None:
    proc = run([sys.executable, "-m", "pytest", "-q"], cwd=repo)
    assert_returncode_zero(proc, "pytest failed")


def make_dataset(tmpdir: Path) -> dict[str, Path]:
    rng = np.random.default_rng(0)
    n = 120
    clusters = rng.integers(0, 4, size=n)
    df = pd.DataFrame(
        {
            "num1": clusters + rng.normal(scale=0.1, size=n),
            "num2": clusters * 2 + rng.normal(scale=0.1, size=n),
            "num3": clusters * 3 + rng.normal(scale=0.1, size=n),
            "cat1": np.where(clusters % 3 == 0, "A", np.where(clusters % 3 == 1, "B", "C")),
            "cat2": np.where(clusters < 2, "D", "E"),
            "cat3": np.where(clusters % 2 == 0, "X", "Y"),
            "Date": pd.date_range("2023-01-01", periods=n),
            "Statut commercial": np.where(rng.random(n) < 0.8, "Gagné", "Perdu"),
        }
    )
    raw = df.copy()
    raw_path = tmpdir / "raw.csv"
    raw.to_csv(raw_path, index=False)

    df_out = raw.copy()
    df_out["flag_multivariate"] = False
    outliers = df_out.sample(3, random_state=1)
    df_out.loc[outliers.index, ["num1", "num2", "num3"]] = 1000
    df_out.loc[outliers.index, "flag_multivariate"] = True
    out_path = tmpdir / "with_outliers.csv"
    df_out.to_csv(out_path, index=False)

    clean = df_out.loc[~df_out["flag_multivariate"]].copy()
    clean = clean[clean["Statut commercial"] != "Perdu"].copy()
    clean_path = tmpdir / "clean.csv"
    clean.to_csv(clean_path, index=False)

    return {"raw": raw_path, "outliers": out_path, "clean": clean_path}


def make_config(paths: dict[str, Path], tmpdir: Path) -> Path:
    cfg = {
        "input_file": str(paths["raw"]),
        "phase1_file": str(paths["outliers"]),
        "phase2_file": str(paths["clean"]),
        "output_dir": str(tmpdir / "out"),
        "exclude_lost": True,
        "compare_versions": True,
        "run_temporal_tests": True,
        "output_pdf": str(tmpdir / "report.pdf"),
        "methods": ["pca", "umap"],
    }
    path = tmpdir / "mini_config.yaml"
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg, fh)
    return path


def run_pipeline(repo: Path, config_path: Path) -> subprocess.CompletedProcess:
    return run([sys.executable, "phase4v3.py", "--config", str(config_path)], cwd=repo)


def md5sum(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_output(tmpdir: Path) -> None:
    out_dir = tmpdir / "out"
    pdf = tmpdir / "report.pdf"
    assert pdf.exists() and os.path.getsize(pdf) > 50_000, "PDF manquant ou trop léger"
    figs = {
        "correlation": next(out_dir.glob("*correlation.png"), None),
        "umap_scatter": next(out_dir.glob("umap_scatter_2d.png"), None),
        "heatmap": out_dir / "methods_heatmap.png",
    }
    for key, path in figs.items():
        assert path and path.exists(), f"Figure {key} manquante"

    comp = out_dir / "comparison_metrics.csv"
    assert comp.exists(), "comparison_metrics.csv manquant"
    df_cmp = pd.read_csv(comp)
    cols = {"variance_cumulee_%", "silhouette", "trustworthiness"}
    assert cols <= set(df_cmp.columns), "Colonnes attendues manquantes dans comparison_metrics"

    metrics = pd.read_csv(out_dir / "metrics.csv")
    assert (metrics["silhouette"].between(-1, 1)).all(), "silhouette hors intervalle"
    assert (metrics["trustworthiness"] >= 0.8).any(), "Aucune méthode avec trustworthiness >= 0.8"


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate phase4v3 pipeline")
    parser.add_argument("--tmpdir", required=True)
    parser.add_argument("--keep", action="store_true")
    args = parser.parse_args()

    tmpdir = Path(args.tmpdir).resolve()
    if tmpdir.exists():
        shutil.rmtree(tmpdir)
    tmpdir.mkdir(parents=True)

    repo = Path(__file__).resolve().parent
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    try:
        check_importability()
        check_functions()
        check_lint(repo)
        check_pytest(repo)
        paths = make_dataset(tmpdir)
        config = make_config(paths, tmpdir)
        first = run_pipeline(repo, config)
        assert_returncode_zero(first, "Pipeline execution failed")
        validate_output(tmpdir)
        pdf_md5 = md5sum(tmpdir / "report.pdf")
        second = run_pipeline(repo, config)
        assert_returncode_zero(second, "Pipeline re-run failed")
        assert md5sum(tmpdir / "report.pdf") == pdf_md5, "PDF non reproductible"
    finally:
        if not args.keep:
            shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    main()
