# !/usr/bin/env python3
"""Phase 4 Version 3 utilities.

Bloc 1: Chargement et structuration des jeux de donnees.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import yaml

from data_preparation import prepare_data
from variable_selection import select_variables
from factor_methods import run_pca, run_mca, run_famd, run_mfa
from nonlinear_methods import run_all_nonlinear
from evaluate_methods import evaluate_methods, plot_methods_heatmap
from dataset_comparison import handle_missing_values
from visualization import generate_figures


# ---------------------------------------------------------------------------
# Utilitaires internes
# ---------------------------------------------------------------------------

def _read_dataset(path: Path) -> pd.DataFrame:
    """Read a CSV or Excel file into a ``DataFrame`` with basic type handling."""
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        df = pd.read_excel(path)

    # Clean column names
    df.columns = [str(c).strip() for c in df.columns]

    # Parse dates
    for col in df.columns:
        if "date" in col.lower():
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # Parse numerical amounts
    for col in df.select_dtypes(include="object"):
        if any(k in col.lower() for k in ["montant", "recette", "budget", "total"]):
            series = df[col].astype(str).str.replace("\xa0", "", regex=False)
            series = series.str.replace(" ", "", regex=False)
            series = series.str.replace(",", ".", regex=False)
            df[col] = pd.to_numeric(series, errors="coerce")
    return df


def _load_data_dictionary(path: Optional[Path]) -> Dict[str, str]:
    """Load column rename mapping from a data dictionary Excel file."""
    if path is None or not path.exists():
        return {}
    try:
        df = pd.read_excel(path)
    except Exception as exc:  # pragma: no cover - optional dependency
        logging.getLogger(__name__).warning("Could not read data dictionary: %s", exc)
        return {}

    cols = {c.lower(): c for c in df.columns}
    src = next((cols[c] for c in ["original", "colonne", "column"] if c in cols), None)
    dst = next((cols[c] for c in ["clean", "standard", "renamed"] if c in cols), None)
    if src is None or dst is None:
        return {}
    mapping = dict(zip(df[src].astype(str), df[dst].astype(str)))
    return mapping


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_datasets(config: Dict[str, str]) -> Dict[str, pd.DataFrame]:
    """Load raw and cleaned datasets for phase 4.

    Parameters
    ----------
    config : dict
        Configuration dictionary with at least ``input_file``. Optional keys are
        ``phase1_file``, ``phase2_file``, ``phase3_file`` (or ``phase3_multi_file`` and
        ``phase3_univ_file``) and ``data_dictionary``.

    Returns
    -------
    dict
        Mapping of dataset name to ``DataFrame``. Keys include ``raw`` and the
        phases present in ``config``.
    """
    logger = logging.getLogger(__name__)
    if "input_file" not in config:
        raise ValueError("'input_file' missing from config")

    datasets: Dict[str, pd.DataFrame] = {}

    raw_path = Path(config["input_file"])
    datasets["raw"] = _read_dataset(raw_path)
    logger.info("Raw dataset loaded from %s", raw_path)

    mapping = _load_data_dictionary(Path(config.get("data_dictionary", "")))

    def _apply_mapping(df: pd.DataFrame) -> pd.DataFrame:
        if mapping:
            df = df.rename(columns={c: mapping.get(c, c) for c in df.columns})
        return df

    for key, cfg_key in [
        ("phase1", "phase1_file"),
        ("phase2", "phase2_file"),
        ("phase3", "phase3_file"),
        ("phase3_multi", "phase3_multi_file"),
        ("phase3_univ", "phase3_univ_file"),
    ]:
        path_str = config.get(cfg_key)
        if not path_str:
            continue
        path = Path(path_str)
        if not path.exists():
            logger.warning("Dataset %s not found: %s", key, path)
            continue
        df = _read_dataset(path) if path.suffix.lower() != ".csv" else pd.read_csv(path)
        datasets[key] = _apply_mapping(df)
        logger.info("Loaded %s dataset from %s", key, path)

    # Basic coherence check: try to align column names with raw data when possible
    ref_cols = set(datasets["raw"].columns)
    for name, df in list(datasets.items()):
        extra = set(df.columns) - ref_cols
        if extra:
            logger.debug("%s has %d additional columns", name, len(extra))
    return datasets


# ---------------------------------------------------------------------------
# Pipeline helpers
# ---------------------------------------------------------------------------

def _setup_logging(output_dir: Path, level: str = "INFO") -> logging.Logger:
    """Configure ``logging`` to write to ``phase4.log`` in ``output_dir``."""
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger()
    logger.setLevel(level)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    file_handler = logging.FileHandler(output_dir / "phase4.log", encoding="utf-8")
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(fmt)
    logger.addHandler(stream_handler)
    return logger


def _run_single_dataset(
    df: pd.DataFrame,
    output_dir: Path,
    *,
    exclude_lost: bool = True,
    min_modalite_freq: int = 5,
    random_state: int = 0,
) -> Dict[str, Any]:
    """Execute the analysis pipeline on a single dataframe."""
    import numpy as np
    import matplotlib.pyplot as plt

    df_prep = prepare_data(df, exclude_lost=exclude_lost)
    df_active, quant_vars, qual_vars = select_variables(
        df_prep, min_modalite_freq=min_modalite_freq
    )
    df_active = handle_missing_values(df_active, quant_vars, qual_vars)

    factor_results: Dict[str, Any] = {}
    if quant_vars:
        factor_results["pca"] = run_pca(
            df_active, quant_vars, optimize=True, random_state=random_state
        )
    if qual_vars:
        factor_results["mca"] = run_mca(
            df_active, qual_vars, optimize=True, random_state=random_state
        )
    if quant_vars and qual_vars:
        try:
            factor_results["famd"] = run_famd(
                df_active,
                quant_vars,
                qual_vars,
                optimize=True,
                random_state=random_state,
            )
        except ValueError as exc:
            logging.getLogger(__name__).warning("FAMD skipped: %s", exc)

    groups = []
    if quant_vars:
        groups.append(quant_vars)
    if qual_vars:
        groups.append(qual_vars)
    if len(groups) > 1:
        factor_results["mfa"] = run_mfa(
            df_active, groups, optimize=True, random_state=random_state
        )

    nonlin_results = run_all_nonlinear(df_active)
    valid_nonlin = {
        k: v
        for k, v in nonlin_results.items()
        if isinstance(v.get("embeddings"), pd.DataFrame) and not v["embeddings"].empty
    }

    if not factor_results and not valid_nonlin:
        logging.getLogger(__name__).warning("No variables selected; skipping analysis")
        empty = pd.DataFrame()
        empty.to_csv(output_dir / "metrics.csv")
        return {"metrics": empty, "figures": {}}

    metrics = evaluate_methods(
        {**factor_results, **valid_nonlin},
        df_active,
        quant_vars,
        qual_vars,
        n_clusters=3 if len(df_active) > 3 else 2,
    )

    plot_methods_heatmap(metrics, output_dir)
    metrics.to_csv(output_dir / "metrics.csv")

    figures = generate_figures(factor_results, nonlin_results, df_active, quant_vars, qual_vars)
    for name, fig in figures.items():
        fig.savefig(output_dir / f"{name}.png")
        plt.close(fig)

    return {"metrics": metrics, "figures": figures}


def run_pipeline(config: Dict[str, Any]) -> Dict[str, Any]:
    """Run the complete phase 4 pipeline based on ``config``."""
    random_state = int(config.get("random_state", 0))
    output_dir = Path(config.get("output_dir", "phase4_output"))
    _setup_logging(output_dir)

    import numpy as np

    np.random.seed(random_state)

    datasets = load_datasets(config)
    data_key = config.get("dataset", "raw")
    if data_key not in datasets:
        raise KeyError(f"dataset '{data_key}' not found in config")

    logging.info("Running pipeline on dataset '%s'", data_key)
    result = _run_single_dataset(
        datasets[data_key],
        output_dir,
        exclude_lost=bool(config.get("exclude_lost", True)),
        min_modalite_freq=int(config.get("min_modalite_freq", 5)),
        random_state=random_state,
    )
    logging.info("Analysis complete")
    return result


def _load_config(path: Path) -> Dict[str, Any]:
    """Load a YAML or JSON configuration file."""
    with open(path, "r", encoding="utf-8") as fh:
        if path.suffix.lower() in {".yaml", ".yml"}:
            return yaml.safe_load(fh)
        return json.load(fh)


def main(argv: Optional[list[str]] = None) -> None:
    """Entry point for command line execution."""
    parser = argparse.ArgumentParser(description="Phase 4 analysis pipeline")
    parser.add_argument("--config", required=True, help="Path to config YAML/JSON")
    args = parser.parse_args(argv)

    cfg = _load_config(Path(args.config))
    run_pipeline(cfg)


if __name__ == "__main__":  # pragma: no cover - manual execution
    main()

