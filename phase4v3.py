#!/usr/bin/env python3
"""Bloc 1 – Chargement et structuration des jeux de données.

This module defines a helper ``load_datasets`` used in phase 4.
It loads the raw Excel/CSV export along with the cleaned datasets
produced during phases 1–3, applying minimal type conversions and
column harmonisation.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

from sklearn.preprocessing import StandardScaler

import pandas as pd
import numpy as np

# scripts de fine-tuning fournis (Bloc 1-bis)
try:
    from fine_tune_famd import run_famd as tune_famd
except ImportError:
    logging.getLogger(__name__).warning("Module fine_tune_famd introuvable")
    tune_famd = None
try:
    from fine_tune_pca import run_pca as tune_pca
except ImportError:
    logging.getLogger(__name__).warning("Module fine_tune_pca introuvable")
    tune_pca = None
try:
    from fine_tuning_mca import run_mca as tune_mca
except ImportError:
    logging.getLogger(__name__).warning("Module fine_tuning_mca introuvable")
    tune_mca = None
try:
    from fine_tune_mfa import run_mfa as tune_mfa
except ImportError:
    logging.getLogger(__name__).warning("Module fine_tune_mfa introuvable")
    tune_mfa = None
try:
    from fine_tuning_umap import run_umap as tune_umap
except ImportError:
    logging.getLogger(__name__).warning("Module fine_tuning_umap introuvable")
    tune_umap = None
try:
    from phase4_fine_tune_phate import run_phate as tune_phate
except ImportError:
    logging.getLogger(__name__).warning("Module phase4_fine_tune_phate introuvable")
    tune_phate = None
try:
    from pacmap_fine_tune import run_pacmap as tune_pacmap
except ImportError:
    logging.getLogger(__name__).warning("Module pacmap_fine_tune introuvable")
    tune_pacmap = None


def _read_generic(path: Path) -> pd.DataFrame:
    """Read a CSV or Excel file depending on its suffix."""
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    return pd.read_excel(path, engine="openpyxl")


def _coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    """Attempt basic type conversions (dates and numerics)."""
    df = df.copy()
    for col in df.columns:
        low = col.lower()
        if "date" in low:
            df[col] = pd.to_datetime(df[col], errors="ignore")
        elif df[col].dtype == object:
            cleaned = pd.to_numeric(df[col].str.replace(",", ".", regex=False),
                                    errors="ignore")
            if cleaned.notna().any():
                df[col] = cleaned
    return df


def _apply_dictionary(df: pd.DataFrame, mapping: Dict[str, str]) -> pd.DataFrame:
    if not mapping:
        return df
    rename = {c: mapping.get(c, c) for c in df.columns}
    return df.rename(columns=rename)


def load_datasets(config: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
    """Load raw and cleaned datasets from various phases.

    Parameters
    ----------
    config : dict
        Configuration dictionary containing at least ``input_file``.
        Optional keys ``phase1_file``, ``phase2_file`` and ``phase3_file`` can
        point to the cleaned datasets. ``data_dictionary`` may specify an
        Excel file mapping original to harmonised column names.

    Returns
    -------
    dict
        ``{"raw": df_raw, "phase1": df_p1, ...}`` with only available phases.
    """
    logger = logging.getLogger(__name__)

    datasets: Dict[str, pd.DataFrame] = {}

    # --- dictionnaire de données optionnel ---------------------------------
    mapping: Dict[str, str] = {}
    ddict = config.get("data_dictionary")
    if ddict:
        try:
            ddf = pd.read_excel(ddict)
            cols = {c.lower(): c for c in ddf.columns}
            old_col = next((cols[c] for c in ["original", "ancien", "variable"] if c in cols), None)
            new_col = next((cols[c] for c in ["new", "standard", "nouveau"] if c in cols), None)
            if old_col and new_col:
                mapping = dict(zip(ddf[old_col].astype(str), ddf[new_col].astype(str)))
        except Exception as exc:
            logger.warning("Impossible de lire le dictionnaire de données %s: %s", ddict, exc)

    # --- raw dataset -------------------------------------------------------
    raw_path = Path(config["input_file"])
    logger.info("Chargement du fichier brut: %s", raw_path)
    df_raw = _coerce_types(_read_generic(raw_path))
    df_raw = _apply_dictionary(df_raw, mapping)
    datasets["raw"] = df_raw

    # --- cleaned datasets --------------------------------------------------
    phase_paths = {
        "phase1": config.get("phase1_file"),
        "phase2": config.get("phase2_file"),
        "phase3": config.get("phase3_file"),
    }
    for name, path_str in phase_paths.items():
        if not path_str:
            continue
        path = Path(path_str)
        logger.info("Chargement %s : %s", name, path)
        df = _coerce_types(_read_generic(path))
        df = _apply_dictionary(df, mapping)
        datasets[name] = df

    # --- simple cohérence colonnes ---------------------------------------
    ref_cols = set(df_raw.columns)
    for name, df in datasets.items():
        miss = ref_cols - set(df.columns)
        extra = set(df.columns) - ref_cols
        if miss or extra:
            logger.warning("Colonnes incohérentes pour %s (manquantes=%s, en_trop=%s)",
                           name, sorted(miss), sorted(extra))
    return datasets


def select_variables(
    df: pd.DataFrame,
    *,
    data_dict: Optional[pd.DataFrame] = None,
    min_modalite_freq: int = 5,
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """Identify quantitative and qualitative variables for dimensional analyses.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame nettoyé issu de :func:`prepare_data`.
    data_dict : Optional[pandas.DataFrame], optional
        Dictionnaire précisant les variables actives (colonne ``keep`` booléenne).
    min_modalite_freq : int, default=5
        Seuil sous lequel les modalités rares sont regroupées en ``Autre``.

    Returns
    -------
    tuple
        ``(df_active, quant_vars, qual_vars)`` avec le DataFrame restreint aux
        variables retenues.
    """
    logger = logging.getLogger(__name__)
    df = df.copy()

    # ----- 1. Exclusion des colonnes non informatives -----------------------
    exclude: set[str] = set()
    if data_dict is not None:
        cols = {c.lower(): c for c in data_dict.columns}
        name_col = next((cols[c] for c in ["variable", "column", "colonne"] if c in cols), None)
        keep_col = next((cols[c] for c in ["keep", "active", "actif"] if c in cols), None)
        if name_col and keep_col:
            excl = data_dict.loc[~data_dict[keep_col].astype(bool), name_col].astype(str)
            exclude.update(excl.tolist())

    keywords = ["id", "code", "ident", "uuid", "titre", "comment", "desc", "texte"]
    for col in df.columns:
        low = col.lower()
        if any(k in low for k in keywords):
            exclude.add(col)
        elif df[col].nunique(dropna=False) <= 1:
            exclude.add(col)
        elif df[col].isna().mean() > 0.9:
            exclude.add(col)
        elif (
            df[col].dtype == object
            and df[col].str.len().mean() > 50
            and df[col].nunique() > 20
        ):
            exclude.add(col)

    if exclude:
        logger.info("Exclusion de %s colonnes non pertinentes", len(exclude))
        df = df.drop(columns=[c for c in exclude if c in df.columns])

    # ----- 2. Séparation quanti/quali --------------------------------------
    quant_vars = list(df.select_dtypes(include=["number"]).columns)
    qual_vars = [c for c in df.columns if c not in quant_vars]

    for col in quant_vars:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        if (df[col] > 0).all() and abs(df[col].skew()) > 3:
            df[col] = np.log10(df[col] + 1)

    # ----- 3. Traitement des qualitatives ----------------------------------
    final_qual: List[str] = []
    for col in qual_vars:
        df[col] = df[col].astype("category")
        counts = df[col].value_counts(dropna=False)
        rares = counts[counts < min_modalite_freq].index
        if len(rares):
            df[col] = df[col].cat.add_categories("Autre")
            df[col] = df[col].where(~df[col].isin(rares), "Autre").astype("category")
        if df[col].nunique() > 1:
            final_qual.append(col)

    qual_vars = final_qual
    quant_vars = [c for c in quant_vars if df[c].var(skipna=True) not in (0, float("nan"))]

    selected = quant_vars + qual_vars
    df_active = df[selected].copy()

    logger.info("%s variables quantitatives conservées", len(quant_vars))
    logger.info("%s variables qualitatives conservées", len(qual_vars))

    return df_active, quant_vars, qual_vars



# ---------------------------------------------------------------------------
# Integration of subsequent blocks
# ---------------------------------------------------------------------------
from block4_factor_methods import run_all_factor_methods
from nonlinear_methods import run_all_nonlin
from block6_visualization import generate_figures
from block7_evaluation import evaluate_methods, plot_methods_heatmap
from block9_unsupervised_cv import unsupervised_cv_and_temporal_tests
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from datetime import datetime


def export_report_to_pdf(figures: Dict[str, plt.Figure],
                         tables: Dict[str, Any],
                         output_path: str | Path) -> None:
    """Compile figures and tables into a single PDF report."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    def _df_to_fig(df: pd.DataFrame, title: str) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(8.27, 11.69))
        ax.axis("off")
        ax.set_title(title)
        tbl = ax.table(cellText=df.values,
                       colLabels=df.columns,
                       rowLabels=df.index,
                       loc="center")
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8)
        return fig

    with PdfPages(out) as pdf:
        # Title page
        fig, ax = plt.subplots(figsize=(8.27, 11.69))
        ax.axis("off")
        ax.text(0.5, 0.6, "Rapport Phase 4", ha="center", va="center", fontsize=16)
        ax.text(0.5, 0.4, datetime.now().strftime("%d/%m/%Y"),
                ha="center", va="center", fontsize=12)
        pdf.savefig(fig)
        plt.close(fig)

        # Figures
        for name, fig in figures.items():
            if fig is None:
                continue
            fig.suptitle(name.replace("_", " "))
            pdf.savefig(fig)
            plt.close(fig)

        # Tables and text
        for name, obj in tables.items():
            if isinstance(obj, plt.Figure):
                pdf.savefig(obj)
                plt.close(obj)
            elif isinstance(obj, pd.DataFrame):
                f = _df_to_fig(obj.round(2), name.replace("_", " "))
                pdf.savefig(f)
                plt.close(f)
            else:
                f, ax = plt.subplots(figsize=(8.27, 11.69))
                ax.axis("off")
                ax.set_title(name.replace("_", " "))
                ax.text(0.5, 0.5, str(obj), ha="center", va="center")
                pdf.savefig(f)
                plt.close(f)

        # Closing page
        fig, ax = plt.subplots(figsize=(8.27, 11.69))
        ax.axis("off")
        ax.text(0.5, 0.5, "Fin du rapport", ha="center", va="center")
        pdf.savefig(fig)
        plt.close(fig)


def main() -> None:
    """Minimal orchestrator for phase 4 using helper functions."""
    import argparse
    import json
    try:
        import yaml  # type: ignore
    except Exception:  # pragma: no cover - optional dependency
        yaml = None

    parser = argparse.ArgumentParser(description="Run phase 4 pipeline")
    parser.add_argument("--config", required=True, help="YAML or JSON configuration")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        raise FileNotFoundError(cfg_path)
    with open(cfg_path, "r", encoding="utf-8") as fh:
        if cfg_path.suffix.lower() == ".json":
            config = json.load(fh)
        else:
            if yaml is None:
                raise RuntimeError("PyYAML is required for YAML configuration")
            config = yaml.safe_load(fh)

    datasets = load_datasets(config)
    df = datasets.get("phase3") or datasets.get("phase2") or datasets.get("phase1") or datasets["raw"]

    df_active, quant_vars, qual_vars = select_variables(df)

    factor_results = run_all_factor_methods(df_active, quant_vars, qual_vars)
    nonlin_results = run_all_nonlin(df_active)

    cv_temporal = unsupervised_cv_and_temporal_tests(df_active, quant_vars, qual_vars)

    figs = generate_figures(factor_results, nonlin_results, df_active, quant_vars, qual_vars)

    metrics = evaluate_methods({**factor_results, **nonlin_results},
                              df_active, quant_vars, qual_vars)

    out_dir = Path(config.get("output_dir", "phase4_output"))
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics.to_csv(out_dir / "methods_comparison.csv")
    plot_methods_heatmap(metrics, out_dir)

    for name, fig in figs.items():
        fig.savefig(out_dir / f"{name}.png")

    with open(out_dir / "cv_temporal_results.json", "w", encoding="utf-8") as fh:
        json.dump(cv_temporal, fh, ensure_ascii=False, indent=2)

    tables = {
        "Comparaison des methodes": metrics,
        "Validation croisee / Temporal": pd.DataFrame.from_dict(cv_temporal, orient="index"),
    }

    output_pdf = Path(config.get("output_pdf", out_dir / "phase4_report.pdf"))
    export_report_to_pdf(figs, tables, output_pdf)
    print(f"Figures saved in {out_dir}")
    print(f"PDF generated at {output_pdf}")


if __name__ == "__main__":
    main()
