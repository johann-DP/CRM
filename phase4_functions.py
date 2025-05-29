"""Aggregated helper functions for phase 4 pipeline."""

from __future__ import annotations

# ---------------------------------------------------------------------------
# dataset_loader.py
# ---------------------------------------------------------------------------
# -*- coding: utf-8 -*-
"""Utilities for loading CRM datasets.

This module contains a :func:`load_datasets` function extracted from
``phase4v3.py`` so that the old monolithic script can be removed.
The API is kept identical for backward compatibility.
"""

from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import os
# limite OpenBLAS à 24 threads (ou moins)
os.environ["OPENBLAS_NUM_THREADS"] = "24"

import pandas as pd

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def _read_dataset(path: Path) -> pd.DataFrame:
    """Read a CSV or Excel file with basic type handling."""
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        df = pd.read_excel(path)
    df.columns = [str(c).strip() for c in df.columns]
    for col in df.columns:
        if "date" in col.lower():
            df[col] = pd.to_datetime(df[col], errors="coerce")
    for col in df.select_dtypes(include="object"):
        if any(k in col.lower() for k in ["montant", "recette", "budget", "total"]):
            series = df[col].astype(str).str.replace("\xa0", "", regex=False)
            series = series.str.replace(" ", "", regex=False)
            series = series.str.replace(",", ".", regex=False)
            df[col] = pd.to_numeric(series, errors="coerce")
    return df


def _load_data_dictionary(path: Optional[Path]) -> Dict[str, str]:
    """Load column rename mapping from an Excel data dictionary."""
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


def load_datasets(
    config: Mapping[str, Any], *, ignore_schema: bool = False
) -> Dict[str, pd.DataFrame]:
    """Load raw and processed datasets according to ``config``.

    Parameters
    ----------
    config:
        Mapping of configuration options. At minimum ``input_file`` must be
        provided.
    ignore_schema:
        If ``True`` the column comparison between the raw dataset and the other
        datasets is relaxed: missing columns are added with ``NA`` values and
        extra columns are dropped instead of raising a :class:`ValueError`.
    """
    logger = logging.getLogger(__name__)

    if not isinstance(config, Mapping):
        raise TypeError("config must be a mapping")
    if "input_file" not in config:
        raise ValueError("'input_file' missing from config")

    mapping = _load_data_dictionary(Path(config.get("data_dictionary", "")))

    def _apply_mapping(df: pd.DataFrame) -> pd.DataFrame:
        if mapping:
            df = df.rename(columns={c: mapping.get(c, c) for c in df.columns})
        return df

    datasets: Dict[str, pd.DataFrame] = {}
    raw_path = Path(config["input_file"])
    datasets["raw"] = _read_dataset(raw_path)
    logger.info(
        "Dataset brut chargé depuis %s [%d lignes, %d colonnes]",
        raw_path,
        datasets["raw"].shape[0],
        datasets["raw"].shape[1],
    )

    datasets["raw"] = _apply_mapping(datasets["raw"])

    for key, cfg_key in [
        ("cleaned_1", "input_file_cleaned_1"),
        ("phase2", "phase2_file"),
        ("cleaned_3_all", "input_file_cleaned_3_all"),
        ("cleaned_3_multi", "input_file_cleaned_3_multi"),
        ("cleaned_3_univ", "input_file_cleaned_3_univ"),
    ]:
        path_str = config.get(cfg_key)
        if not path_str:
            continue
        path = Path(path_str)
        if not path.exists():
            logger.warning("Jeu de données %s introuvable : %s", key, path)
            continue
        df = _read_dataset(path)
        datasets[key] = _apply_mapping(df)
        logger.info(
            "Jeu de données %s chargé depuis %s [%d lignes, %d colonnes]",
            key,
            path,
            df.shape[0],
            df.shape[1],
        )
    ref_cols = list(datasets["raw"].columns)
    ref_set = set(ref_cols)
    for name, df in list(datasets.items()):
        cols = set(df.columns)
        missing = ref_set - cols
        extra = cols - ref_set
        if missing or extra:
            if not ignore_schema:
                raise ValueError(
                    f"{name} columns mismatch: missing {missing or None}, extra {extra or None}"
                )
            if missing:
                for col in missing:
                    df[col] = pd.NA
            if extra:
                df = df.drop(columns=list(extra))
        # reorder columns so all datasets share the same order
        datasets[name] = df[ref_cols]
    return datasets


# ---------------------------------------------------------------------------
# data_preparation.py
# ---------------------------------------------------------------------------
"""Data preparation utilities for Phase 4.

This module provides a self-contained ``prepare_data`` function used to
clean and standardise the CRM datasets before dimensionality reduction. It
re-implements the relevant logic previously found in ``phase4v2.py`` and
the fine tuning scripts so that these legacy files can be removed.
"""


import logging

import pandas as pd
from pathlib import Path

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def prepare_data(
    df: pd.DataFrame,
    *,
    exclude_lost: bool = True,
    flagged_ids_path: str | Path | None = None,
) -> pd.DataFrame:
    """Return a cleaned and standardised copy of ``df``.

    Parameters
    ----------
    df : pd.DataFrame
        Raw or already partially cleaned CRM dataset.
    exclude_lost : bool, default ``True``
        If ``True``, rows marked as lost or cancelled opportunities are
        removed from the returned DataFrame.
    flagged_ids_path : str or Path, optional
        Optional CSV file containing an identifier column named ``Code`` to
        remove additional rows flagged as outliers during phase 3. The file is
        ignored if not found.

    Returns
    -------
    pd.DataFrame
        The cleaned DataFrame with numerical columns scaled to zero mean and
        unit variance.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")

    df_clean = df.copy()

    # ------------------------------------------------------------------
    # 1) Dates: parse and drop obvious out-of-range values
    # ------------------------------------------------------------------
    date_cols = [c for c in df_clean.columns if "date" in c.lower()]
    min_date = pd.Timestamp("1990-01-01")
    max_date = pd.Timestamp("2050-12-31")
    for col in date_cols:
        df_clean[col] = pd.to_datetime(df_clean[col], errors="coerce")
        mask = df_clean[col].lt(min_date) | df_clean[col].gt(max_date)
        if mask.any():
            logger.warning("%d invalid dates replaced by NaT in '%s'", mask.sum(), col)
            df_clean.loc[mask, col] = pd.NaT

    # ------------------------------------------------------------------
    # 2) Monetary amounts: numeric conversion and negative values
    # ------------------------------------------------------------------
    amount_cols = [
        "Total recette actualisé",
        "Total recette réalisé",
        "Total recette produit",
        "Budget client estimé",
    ]
    for col in amount_cols:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")
            neg = df_clean[col] < 0
            if neg.any():
                logger.warning("%d negative values set to NaN in '%s'", neg.sum(), col)
                df_clean.loc[neg, col] = np.nan

    # ------------------------------------------------------------------
    # 3) Remove duplicate opportunity identifiers if present
    # ------------------------------------------------------------------
    if "Code" in df_clean.columns:
        before = len(df_clean)
        df_clean = df_clean.drop_duplicates(subset=["Code"])
        if len(df_clean) != before:
            logger.info("%d lignes dupliquees supprimees", before - len(df_clean))

    # ------------------------------------------------------------------
    # 4) Optional external list of outliers to exclude
    # ------------------------------------------------------------------
    flagged_file = (
        Path(flagged_ids_path)
        if flagged_ids_path is not None
        else Path(__file__).with_name("dataset_phase3_flagged.csv")
    )
    if flagged_file.is_file() and "Code" in df_clean.columns:
        try:
            flagged_df = pd.read_csv(flagged_file)
        except Exception as exc:  # pragma: no cover - unexpected I/O error
            logger.warning("Could not read %s: %s", flagged_file, exc)
        else:
            if "Code" in flagged_df.columns:
                flagged_ids = set(flagged_df["Code"])
                mask_flagged = df_clean["Code"].isin(flagged_ids)
                if mask_flagged.any():
                    logger.info(
                        "%d valeurs aberrantes supprimees via %s",
                        int(mask_flagged.sum()),
                        flagged_file.name,
                    )
                    df_clean = df_clean.loc[~mask_flagged]

    # ------------------------------------------------------------------
    # 5) Derived indicators used in later analyses
    # ------------------------------------------------------------------
    if {"Date de début actualisée", "Date de fin réelle"} <= set(df_clean.columns):
        df_clean["duree_projet_jours"] = (
            df_clean["Date de fin réelle"] - df_clean["Date de début actualisée"]
        ).dt.days
    if {"Total recette réalisé", "Budget client estimé"} <= set(df_clean.columns):
        denom = df_clean["Budget client estimé"].replace(0, np.nan)
        df_clean["taux_realisation"] = df_clean["Total recette réalisé"] / denom
        df_clean["taux_realisation"] = df_clean["taux_realisation"].replace(
            [np.inf, -np.inf], np.nan
        )
    if {"Total recette réalisé", "Charge prévisionnelle projet"} <= set(
        df_clean.columns
    ):
        df_clean["marge_estimee"] = (
            df_clean["Total recette réalisé"] - df_clean["Charge prévisionnelle projet"]
        )

    # ------------------------------------------------------------------
    # 6) Simple missing value handling
    # ------------------------------------------------------------------
    impute_cols: list[str] = [c for c in amount_cols if c in df_clean.columns]
    if "taux_realisation" in df_clean.columns:
        impute_cols.append("taux_realisation")
    for col in impute_cols:
        median = df_clean[col].median()
        df_clean[col] = df_clean[col].fillna(median)
    for col in df_clean.select_dtypes(include="object"):
        df_clean[col] = df_clean[col].fillna("Non renseigné").astype("category")

    # ------------------------------------------------------------------
    # 7) Filter multivariate outliers flagged during phase 3
    # ------------------------------------------------------------------
    if "flag_multivariate" in df_clean.columns:
        out = df_clean["flag_multivariate"].astype(bool)
        if out.any():
            logger.info(
                "%d valeurs aberrantes supprimees via flag_multivariate", int(out.sum())
            )
            df_clean = df_clean.loc[~out]

    # ------------------------------------------------------------------
    # 8) Exclude lost or cancelled opportunities if requested
    # ------------------------------------------------------------------
    if exclude_lost and "Statut commercial" in df_clean.columns:
        lost_mask = (
            df_clean["Statut commercial"]
            .astype(str)
            .str.contains("perdu|annul|aband", case=False, na=False)
        )
        if lost_mask.any():
            logger.info("%d lost opportunities removed", int(lost_mask.sum()))
            df_clean = df_clean.loc[~lost_mask]
    if exclude_lost and "Motif_non_conformité" in df_clean.columns:
        mask_nc = df_clean["Motif_non_conformité"].notna() & df_clean[
            "Motif_non_conformité"
        ].astype(str).str.strip().ne("")
        if mask_nc.any():
            logger.info("%d non conformities removed", int(mask_nc.sum()))
            df_clean = df_clean.loc[~mask_nc]

    # ------------------------------------------------------------------
    # 9) Standardise numerical columns
    # ------------------------------------------------------------------
    num_cols = df_clean.select_dtypes(include=[np.number]).columns
    num_cols = [c for c in num_cols if c != "Code"]
    if len(num_cols) > 0:
        scaler = StandardScaler()
        df_clean[num_cols] = scaler.fit_transform(df_clean[num_cols])

    return df_clean


"""Utility functions for selecting active variables for CRM analyses."""

from typing import List, Tuple

import pandas as pd


def select_variables(
    df: pd.DataFrame, min_modalite_freq: int = 5
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """Return the dataframe restricted to relevant variables.

    Parameters
    ----------
    df:
        Cleaned dataframe coming from ``prepare_data``.
    min_modalite_freq:
        Minimum frequency below which categorical levels are grouped in
        ``"Autre"``.

    Returns
    -------
    tuple
        ``(df_active, quantitative_vars, qualitative_vars)`` where
        ``df_active`` contains the scaled numeric columns and categorical
        columns cast to ``category``.
    """
    logger = logging.getLogger(__name__)

    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")

    df = df.copy()

    # Columns explicitly ignored based on the data dictionary / prior phases
    exclude = {
        "Code",
        "ID",
        "Id",
        "Identifiant",
        "Client",
        "Contact principal",
        "Titre",
        "texte",
        "commentaire",
        "Commentaires",
    }

    # Drop constant columns and those in the exclusion list
    n_unique = df.nunique(dropna=False)
    constant_cols = n_unique[n_unique <= 1].index.tolist()
    drop_cols = set(constant_cols) | {c for c in df.columns if c in exclude}

    # Remove datetime columns
    drop_cols.update([c for c in df.select_dtypes(include="datetime").columns])

    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    quantitative_vars: List[str] = []
    qualitative_vars: List[str] = []

    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            series = pd.to_numeric(df[col], errors="coerce")
            if series.var(skipna=True) == 0 or series.isna().all():
                logger.warning("Variable quantitative '%s' exclue", col)
                continue
            df[col] = series.astype(float)
            quantitative_vars.append(col)
        else:
            series = df[col].astype("category")
            unique_ratio = series.nunique(dropna=False) / len(series)
            if unique_ratio > 0.8:
                logger.warning("Variable textuelle '%s' exclue", col)
                continue
            counts = series.value_counts(dropna=False)
            if len(series) < min_modalite_freq:
                threshold = 0  # no grouping on tiny samples
            else:
                threshold = min_modalite_freq
            rares = counts[counts < threshold].index
            if len(rares) > 0:
                logger.info(
                    "%d modalités rares dans '%s' regroupées en 'Autre'",
                    len(rares),
                    col,
                )
                if "Autre" not in series.cat.categories:
                    series = series.cat.add_categories(["Autre"])
                series = series.apply(lambda x: "Autre" if x in rares else x).astype(
                    "category"
                )
            if series.nunique(dropna=False) <= 1:
                logger.warning("Variable qualitative '%s' exclue", col)
                continue
            df[col] = series
            qualitative_vars.append(col)

    df_active = df[quantitative_vars + qualitative_vars].copy()

    if quantitative_vars:
        scaler = StandardScaler()
        df_active[quantitative_vars] = scaler.fit_transform(
            df_active[quantitative_vars]
        )

    for col in qualitative_vars:
        df_active[col] = df_active[col].astype("category")

    logger.info("DataFrame actif avec %d variables", len(df_active.columns))
    return df_active, quantitative_vars, qualitative_vars


"""Utility functions for factorial analyses (PCA, MCA, FAMD, MFA).

This module implements standalone wrappers around ``scikit-learn`` and
``prince`` to run the main factorial analysis methods used in the project.
The functions do not depend on other local modules so they can be reused
"""

# ---------------------------------------------------------------------------
# dataset_comparison.py
# ---------------------------------------------------------------------------
# -*- coding: utf-8 -*-
"""Dataset comparison utilities for CRM analyses.

This module implements a `compare_datasets_versions` function applying the
complete dimensionality reduction pipeline on multiple dataset versions. It
re-uses the standalone helper functions provided in this repository (data
preparation, variable selection, factor methods, non-linear methods and
metrics evaluation) and does **not** depend on legacy scripts such as
``phase4v2.py`` or ``fine_tune_*``.
"""


import logging
from typing import Any, Dict, List, Optional, Mapping
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# Missing value handling (copied from phase4v2.py)
# ---------------------------------------------------------------------------


def handle_missing_values(
    df: pd.DataFrame, quant_vars: List[str], qual_vars: List[str]
) -> pd.DataFrame:
    """Impute and drop remaining NA values if needed.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame to process.
    quant_vars : list of str
        Names of quantitative variables.
    qual_vars : list of str
        Names of qualitative variables.

    Returns
    -------
    pandas.DataFrame
        DataFrame with missing values handled.
    """
    logger = logging.getLogger(__name__)
    na_count = int(df.isna().sum().sum())
    if na_count > 0:
        logger.info("Imputation des %d valeurs manquantes restantes", na_count)
        if quant_vars:
            df[quant_vars] = df[quant_vars].fillna(df[quant_vars].median())
        for col in qual_vars:
            if (
                df[col].dtype.name == "category"
                and "Non renseigné" not in df[col].cat.categories
            ):
                df[col] = df[col].cat.add_categories("Non renseigné")
            df[col] = df[col].fillna("Non renseigné").astype("category")
        remaining = int(df.isna().sum().sum())
        if remaining > 0:
            logger.warning(
                "%d NA subsistent après imputation → suppression des lignes concernées",
                remaining,
            )
            df.dropna(inplace=True)
        # After dropping rows, remove categories that may no longer be present
        for col in qual_vars:
            if df[col].dtype.name == "category":
                df[col] = df[col].cat.remove_unused_categories()
    else:
        logger.info("Aucune valeur manquante détectée après sanity_check")

    # Ensure no stray unused categories remain even if no imputation occurred
    for col in qual_vars:
        if df[col].dtype.name == "category":
            df[col] = df[col].cat.remove_unused_categories()

    if df.isna().any().any():
        logger.error("Des NA demeurent dans df après traitement")
    else:
        logger.info("DataFrame sans NA prêt pour FAMD")
    return df


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compare_datasets_versions(
    datasets: Dict[str, pd.DataFrame],
    *,
    exclude_lost: bool = True,
    min_modalite_freq: int = 5,
    output_dir: Optional[str | Path] = None,
    exclude_lost_map: Optional[Mapping[str, bool]] = None,
) -> Dict[str, Any]:
    """Compare dimensionality reduction results between dataset versions.

    Parameters
    ----------
    datasets : dict
        Mapping of version name to raw ``DataFrame``.
    exclude_lost : bool, default ``True``
        Whether to remove lost/cancelled opportunities during preparation.
    exclude_lost_map : Mapping[str, bool], optional
        Mapping overriding ``exclude_lost`` for specific dataset versions.
    min_modalite_freq : int, default ``5``
        Frequency threshold passed to :func:`variable_selection.select_variables`.
    output_dir : str or Path, optional
        Base directory where figures will be saved. A subdirectory per dataset
        version is created when provided.

    Returns
    -------
    dict
        Dictionary with two keys:
        ``"metrics"`` containing the concatenated metrics table and
        ``"details"`` mapping each version name to its individual results
        (metrics, figures and intermediate objects).
    """
    if not isinstance(datasets, dict):
        raise TypeError("datasets must be a dictionary")

    results_by_version: Dict[str, Any] = {}
    metrics_frames: List[pd.DataFrame] = []
    base_dir = Path(output_dir) if output_dir is not None else None

    for name, df in datasets.items():
        logger.info("Processing dataset version '%s'", name)
        excl = (
            exclude_lost_map.get(name, exclude_lost)
            if exclude_lost_map
            else exclude_lost
        )
        df_prep = prepare_data(df, exclude_lost=excl)
        df_active, quant_vars, qual_vars = select_variables(
            df_prep, min_modalite_freq=min_modalite_freq
        )
        df_active = handle_missing_values(df_active, quant_vars, qual_vars)

        # Factorial methods
        factor_results: Dict[str, Any] = {}
        if quant_vars:
            factor_results["pca"] = run_pca(df_active, quant_vars, optimize=True)
        if qual_vars:
            factor_results["mca"] = run_mca(df_active, qual_vars, optimize=True)
        if quant_vars and qual_vars:
            try:
                factor_results["famd"] = run_famd(
                    df_active, quant_vars, qual_vars, optimize=True
                )
            except ValueError as exc:
                logger.warning("FAMD skipped: %s", exc)
        groups = []
        if quant_vars:
            groups.append(quant_vars)
        if qual_vars:
            groups.append(qual_vars)
        if len(groups) > 1:
            factor_results["mfa"] = run_mfa(df_active, groups, optimize=True)

        # Non-linear methods
        nonlin_results = run_all_nonlinear(df_active)

        # Metrics and figures
        cleaned_nonlin = {
            k: v
            for k, v in nonlin_results.items()
            if "embeddings" in v
            and isinstance(v["embeddings"], pd.DataFrame)
            and not v["embeddings"].empty
        }
        all_results = {**factor_results, **cleaned_nonlin}
        k_max = min(15, max(2, len(df_active) - 1))
        metrics = evaluate_methods(
            all_results,
            df_active,
            quant_vars,
            qual_vars,
            k_range=range(2, k_max + 1),
        )
        metrics["dataset_version"] = name
        try:
            fig_dir = base_dir / name if base_dir is not None else None
            figures = generate_figures(
                factor_results,
                nonlin_results,
                df_active,
                quant_vars,
                qual_vars,
                output_dir=fig_dir,
            )
        except Exception as exc:  # pragma: no cover - visualization failure
            logger.warning("Figure generation failed: %s", exc)
            figures = {}

        results_by_version[name] = {
            "metrics": metrics,
            "figures": figures,
            "factor_results": factor_results,
            "nonlinear_results": nonlin_results,
            "quant_vars": quant_vars,
            "qual_vars": qual_vars,
            "df_active": df_active,
        }
        metrics_frames.append(metrics)

    combined = (
        pd.concat(metrics_frames).reset_index().rename(columns={"index": "method"})
    )
    return {"metrics": combined, "details": results_by_version}


if __name__ == "__main__":  # pragma: no cover - manual testing helper
    import pprint

    logging.basicConfig(level=logging.INFO)
    # Example usage with dummy data
    df = pd.DataFrame(
        {
            "Code": [1, 2, 3],
            "Date de début actualisée": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "Date de fin réelle": ["2024-01-05", "2024-01-06", "2024-01-07"],
            "Total recette réalisé": [1000, 2000, 1500],
            "Budget client estimé": [1100, 2100, 1600],
            "Charge prévisionnelle projet": [800, 1800, 1300],
            "Statut commercial": ["Gagné", "Perdu", "Gagné"],
            "Type opportunité": ["T1", "T2", "T1"],
        }
    )
    datasets = {"v1": df, "v2": df.drop(1)}
    out = compare_datasets_versions(datasets, output_dir=Path("figures"))
    pprint.pprint(out["metrics"].head())

# ---------------------------------------------------------------------------
# factor_methods.py
# ---------------------------------------------------------------------------
# -*- coding: utf-8 -*-
"""Utility functions for factorial analyses (PCA, MCA, FAMD, MFA).

This module implements standalone wrappers around ``scikit-learn`` and
``prince`` to run the main factorial analysis methods used in the project.
The functions do not depend on other local modules so they can be reused
independently of ``phase4v2.py`` or the fine-tuning scripts.
"""

from typing import Dict, List, Optional, Sequence, Mapping, Union

from pandas.api.types import is_object_dtype, is_categorical_dtype

import numpy as np
import pandas as pd
import prince


def _get_explained_inertia(model: object) -> List[float]:
    """Return the explained inertia ratio for a fitted model."""
    inertia = getattr(model, "explained_inertia_", None)
    if inertia is not None:
        return list(np.asarray(inertia, dtype=float))

    eigenvalues = getattr(model, "eigenvalues_", None)
    if eigenvalues is None:
        return []
    total = float(np.sum(eigenvalues))
    if total == 0:
        return []
    return list(np.asarray(eigenvalues, dtype=float) / total)


def _select_n_components(eigenvalues: np.ndarray, threshold: float = 0.8) -> int:
    """Select a number of components using Kaiser and inertia criteria."""
    ev = np.asarray(eigenvalues, dtype=float)
    if ev.sum() <= 1.0:
        ev = ev * len(ev)

    n_kaiser = max(1, int(np.sum(ev >= 1)))
    ratios = ev / ev.sum()
    cum = np.cumsum(ratios)
    n_inertia = int(np.searchsorted(cum, threshold) + 1)
    return max(n_kaiser, n_inertia)


def run_pca(
    df_active: pd.DataFrame,
    quant_vars: List[str],
    n_components: Optional[int] = None,
    *,
    optimize: bool = False,
    variance_threshold: float = 0.8,
    whiten: Optional[bool] = None,
    svd_solver: Optional[str] = None,
) -> Dict[str, object]:
    """Run a Principal Component Analysis on quantitative variables.

    Parameters
    ----------
    df_active : pandas.DataFrame
        DataFrame containing the active observations.
    quant_vars : list of str
        Names of the quantitative columns to use.
    n_components : int, optional
        Number of components to keep. If ``None`` and ``optimize`` is ``True``
        the value is determined automatically with a variance threshold.
    optimize : bool, default ``False``
        Activate automatic selection of ``n_components`` when ``n_components`` is
        not provided.
    variance_threshold : float, default ``0.8``
        Cumulative explained variance ratio threshold when ``optimize`` is true.
    whiten : bool, optional
        If provided, sets the ``whiten`` parameter of :class:`~sklearn.decomposition.PCA`.
    svd_solver : str, optional
        If provided, sets the ``svd_solver`` parameter of :class:`~sklearn.decomposition.PCA`.

    Returns
    -------
    dict
        ``{"model", "inertia", "embeddings", "loadings", "runtime_s"}``
    """
    start = time.perf_counter()
    logger = logging.getLogger(__name__)
    stds = df_active[quant_vars].std()
    zero_std = stds[stds == 0].index.tolist()
    if zero_std:
        logger.warning("Variables constantes exclues du scaling : %s", zero_std)
        quant_vars = [c for c in quant_vars if c not in zero_std]

    X = StandardScaler().fit_transform(df_active[quant_vars])
    max_dim = min(X.shape)

    if optimize and n_components is None:
        tmp = PCA(n_components=max_dim).fit(X)
        n_components = _select_n_components(
            tmp.explained_variance_, threshold=variance_threshold
        )
        logger.info("PCA: selected %d components automatically", n_components)

    n_components = n_components or max_dim
    kwargs = {}
    if whiten is not None:
        kwargs["whiten"] = whiten
    if svd_solver is not None:
        kwargs["svd_solver"] = svd_solver
    pca = PCA(n_components=n_components, **kwargs)
    emb = pca.fit_transform(X)

    inertia = pd.Series(
        pca.explained_variance_ratio_,
        index=[f"F{i+1}" for i in range(pca.n_components_)],
    )
    embeddings = pd.DataFrame(
        emb,
        index=df_active.index,
        columns=[f"F{i+1}" for i in range(pca.n_components_)],
    )
    loadings = pd.DataFrame(
        pca.components_.T,
        index=quant_vars,
        columns=[f"F{i+1}" for i in range(pca.n_components_)],
    )

    runtime = time.perf_counter() - start
    result = {
        "model": pca,
        "inertia": inertia,
        "embeddings": embeddings,
        "loadings": loadings,
        "runtime_s": runtime,
    }
    # aliases for compatibility with other naming conventions
    result["explained_variance_ratio"] = inertia
    result["coords"] = embeddings
    result["runtime"] = runtime
    return result


def run_mca(
    df_active: pd.DataFrame,
    qual_vars: List[str],
    n_components: Optional[int] = None,
    *,
    optimize: bool = False,
    variance_threshold: float = 0.8,
    normalize: bool = True,
    n_iter: int = 3,
    max_components: int = 10,
) -> Dict[str, object]:
    """Run Multiple Correspondence Analysis on qualitative variables.

    Parameters
    ----------
    df_active : pandas.DataFrame
        Input data with qualitative variables.
    qual_vars : list of str
        Names of the qualitative columns to use.
    n_components : int, optional
        Number of dimensions to compute. If ``None`` and ``optimize`` is
        ``True`` the value is selected automatically.
    optimize : bool, default ``False``
        Activate automatic selection of ``n_components`` when not provided.
    variance_threshold : float, default ``0.8``
        Cumulative inertia threshold when ``optimize`` is enabled.
    normalize : bool, default ``True``
        If ``True`` applies the Benzecri correction (``correction='benzecri'``).
    n_iter : int, default ``3``
        Number of iterations for the underlying algorithm.
    """
    start = time.perf_counter()

    df_cat = df_active[qual_vars].astype("category")

    max_dim = sum(df_cat[c].nunique() - 1 for c in df_cat.columns)
    max_limit = min(max_dim, max_components)

    mca = prince.MCA(
        n_components=max_limit,
        n_iter=n_iter,
        correction="benzecri" if normalize else None,
    ).fit(df_cat)

    if optimize and n_components is None:
        ev = getattr(mca, "eigenvalues_", None)
        if ev is None:
            ev = np.asarray(mca.explained_inertia_) * len(mca.explained_inertia_)
        components = [i + 1 for i, v in enumerate(ev) if v > 1.0]
        if components:
            n_components = min(max(components), max_limit)
        else:
            n_components = min(len(ev), max_limit)
        logger.info("MCA: selected %d components automatically", n_components)
    else:
        n_components = n_components or max_limit

    if n_components != mca.n_components:
        mca = prince.MCA(
            n_components=n_components,
            n_iter=n_iter,
            correction="benzecri" if normalize else None,
        ).fit(df_cat)

    inertia = pd.Series(
        _get_explained_inertia(mca), index=[f"F{i+1}" for i in range(mca.n_components)]
    )
    embeddings = mca.row_coordinates(df_cat)
    embeddings.index = df_active.index
    col_coords = mca.column_coordinates(df_cat)

    runtime = time.perf_counter() - start
    result = {
        "model": mca,
        "inertia": inertia,
        "embeddings": embeddings,
        "column_coords": col_coords,
        "runtime_s": runtime,
    }
    result["explained_inertia"] = inertia
    result["coords"] = embeddings
    result["runtime"] = runtime
    return result


def run_famd(
    df_active: pd.DataFrame,
    quant_vars: List[str],
    qual_vars: List[str],
    n_components: Optional[int] = None,
    *,
    optimize: bool = False,
    variance_threshold: float = 0.8,
    n_components_rule: Optional[str] = None,
) -> Dict[str, object]:
    """Run Factor Analysis of Mixed Data (FAMD).

    Parameters
    ----------
    n_components_rule : str, optional
        Placeholder for compatibility with configuration files. Currently
        ignored.
    """
    start = time.perf_counter()
    logger = logging.getLogger(__name__)

    if n_components_rule is not None:
        logger.info(
            "FAMD n_components_rule=%s ignored (not implemented)",
            n_components_rule,
        )

    scaler = StandardScaler()
    stds = df_active[quant_vars].std()
    zero_std = stds[stds == 0].index.tolist()
    if zero_std:
        logger.warning("Variables constantes exclues du scaling : %s", zero_std)
        quant_vars = [c for c in quant_vars if c not in zero_std]
    X_quanti = scaler.fit_transform(df_active[quant_vars]) if quant_vars else np.empty((len(df_active), 0))
    df_quanti = pd.DataFrame(X_quanti, index=df_active.index, columns=quant_vars)
    df_mix = pd.concat([df_quanti, df_active[qual_vars].astype("category")], axis=1)

    if df_mix.isnull().any().any():
        raise ValueError("Input contains NaN values")

    if optimize and n_components is None:
        max_dim = df_mix.shape[1]
        tmp = prince.FAMD(n_components=max_dim, n_iter=3).fit(df_mix)
        eig = getattr(tmp, "eigenvalues_", None)
        if eig is None:
            eig = np.asarray(_get_explained_inertia(tmp)) * max_dim
        n_components = _select_n_components(eig, threshold=variance_threshold)
        logger.info("FAMD: selected %d components automatically", n_components)

    n_components = n_components or df_mix.shape[1]
    famd = prince.FAMD(n_components=n_components, n_iter=3)
    famd = famd.fit(df_mix)

    inertia = pd.Series(
        _get_explained_inertia(famd),
        index=[f"F{i+1}" for i in range(famd.n_components)],
    )
    embeddings = famd.row_coordinates(df_mix)
    embeddings.index = df_active.index
    if hasattr(famd, "column_coordinates"):
        col_coords = famd.column_coordinates(df_mix)
    elif hasattr(famd, "column_coordinates_"):
        col_coords = famd.column_coordinates_
    else:
        col_coords = pd.DataFrame()

    if hasattr(famd, "column_contributions"):
        contrib = famd.column_contributions(df_mix)
    elif hasattr(famd, "column_contributions_"):
        contrib = famd.column_contributions_
    else:
        contrib = (col_coords**2).div((col_coords**2).sum(axis=0), axis=1) * 100

    runtime = time.perf_counter() - start
    result = {
        "model": famd,
        "inertia": inertia,
        "embeddings": embeddings,
        "column_coords": col_coords,
        "contributions": contrib,
        "runtime_s": runtime,
    }
    result["explained_inertia"] = inertia
    result["coords"] = embeddings
    result["runtime"] = runtime
    return result


def run_mfa(
    df_active: pd.DataFrame,
    groups: Union[Mapping[str, Sequence[str]], Sequence[Sequence[str]]] | None = None,
    n_components: Optional[int] = None,
    *,
    segment_col: Optional[str] = None,
    optimize: bool = False,
    variance_threshold: float = 0.8,
    normalize: bool = True,
    weights: Optional[Union[Mapping[str, float], Sequence[float]]] = None,
    n_iter: int = 3,
) -> Dict[str, object]:
    """Run Multiple Factor Analysis.

    The ``groups`` argument defines the variable blocks used by MFA. Each
    element of ``groups`` must be a list of column names present in
    ``df_active``. When invoking :func:`run_pipeline`, these groups can be
    provided in the configuration file under ``mfa: {groups: ...}``.
    ``weights`` allows adjusting the relative importance of each group by
    multiplying its (optionally normalised) columns by the specified factor.
    """
    start = time.perf_counter()

    if segment_col:
        df_active = df_active.copy()
        df_active["__segment__"] = df_active[segment_col]
        groups = [[c for c in df_active.columns if c != "__segment__"], ["__segment__"]]

    if groups is None:
        groups = [df_active.columns.tolist()]

    if isinstance(groups, Mapping):
        group_names = list(groups.keys())
        group_list = list(groups.values())
    else:
        group_list = list(groups)
        group_names = [f"G{i}" for i in range(1, len(group_list) + 1)]

    # one-hot encode qualitative variables that appear in groups
    qual_cols = []
    for group in group_list:
        for col in group:
            if col in df_active.columns and (
                is_object_dtype(df_active[col]) or is_categorical_dtype(df_active[col])
            ):
                qual_cols.append(col)
    # remove duplicates while preserving order
    seen = set()
    qual_cols = [c for c in qual_cols if not (c in seen or seen.add(c))]
    if qual_cols:
        enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        encoded = enc.fit_transform(df_active[qual_cols])
        df_dummies = pd.DataFrame(
            encoded, index=df_active.index, columns=enc.get_feature_names_out(qual_cols)
        )
    else:
        df_dummies = pd.DataFrame(index=df_active.index)

    df_num = df_active.drop(columns=qual_cols)
    df_all = pd.concat([df_num, df_dummies], axis=1)

    groups_dict: Dict[str, List[str]] = {}
    used_cols: List[str] = []
    for name, g in zip(group_names, group_list):
        cols: List[str] = []
        for v in g:
            if v in df_all.columns:
                cols.append(v)
            else:
                # qualitative variables have been expanded
                cols.extend([c for c in df_all.columns if c.startswith(f"{v}_")])
        if cols:
            groups_dict[name] = cols
            used_cols.extend(cols)

    remaining = [c for c in df_all.columns if c not in used_cols]
    if remaining:
        groups_dict[f"G{len(groups_dict)+1}"] = remaining
        used_cols.extend(remaining)
    df_all = df_all[used_cols]

    weights_map: Dict[str, float] = {}
    if weights is not None:
        if isinstance(weights, Mapping):
            weights_map = {str(k): float(v) for k, v in weights.items()}
        else:
            weight_list = list(weights)
            if len(weight_list) != len(group_names):
                logger.warning(
                    "MFA weights length mismatch: expected %d, got %d",
                    len(group_names),
                    len(weight_list),
                )
            for name, w in zip(group_names, weight_list):
                weights_map[name] = float(w)

    if normalize:
        scaler = StandardScaler()
        for name, cols in list(groups_dict.items()):
            if not cols:
                continue
            stds = df_all[cols].std()
            zero_std = stds[stds == 0].index.tolist()
            if zero_std:
                logger.warning("Variables constantes exclues du scaling : %s", zero_std)
                cols = [c for c in cols if c not in zero_std]
                df_all.drop(columns=zero_std, inplace=True)
            if not cols:
                del groups_dict[name]
                continue
            df_all[cols] = scaler.fit_transform(df_all[cols])
            groups_dict[name] = cols
            w = weights_map.get(name)
            if w is not None and w != 1.0:
                df_all[cols] = df_all[cols] * w
    else:
        for name, cols in list(groups_dict.items()):
            if not cols:
                del groups_dict[name]
                continue
            w = weights_map.get(name)
            if w is not None and w != 1.0:
                df_all[cols] = df_all[cols] * w

    if optimize and n_components is None:
        max_dim = df_all.shape[1]
        tmp = prince.MFA(n_components=max_dim, n_iter=n_iter)
        tmp = tmp.fit(df_all, groups=groups_dict)
        eig = getattr(tmp, "eigenvalues_", None)
        if eig is None:
            eig = (tmp.percentage_of_variance_ / 100) * max_dim
        n_components = _select_n_components(
            np.asarray(eig), threshold=variance_threshold
        )
        max_limit = min(max_dim, 10)
        n_components = min(n_components, max_limit)
        logger.info("MFA: selected %d components automatically", n_components)

    max_limit = min(df_all.shape[1], 10)
    n_components = n_components or max_limit
    mfa = prince.MFA(n_components=n_components, n_iter=n_iter)
    mfa = mfa.fit(df_all, groups=groups_dict)
    inertia_values = np.asarray(mfa.percentage_of_variance_, dtype=float) / 100
    inertia_values = (
        inertia_values / inertia_values.sum() if inertia_values.sum() > 0 else inertia_values
    )
    mfa.explained_inertia_ = inertia_values
    embeddings = mfa.row_coordinates(df_all)
    embeddings.index = df_active.index

    inertia = pd.Series(
        mfa.explained_inertia_,
        index=[f"F{i+1}" for i in range(len(mfa.explained_inertia_))],
    )

    runtime = time.perf_counter() - start
    result = {
        "model": mfa,
        "inertia": inertia,
        "embeddings": embeddings,
        "runtime_s": runtime,
    }
    result["explained_inertia"] = inertia
    result["coords"] = embeddings
    result["runtime"] = runtime
    return result


# ---------------------------------------------------------------------------
# nonlinear_methods.py
# ---------------------------------------------------------------------------
"""Non-linear dimensionality reduction utilities.

This module re-implements the UMAP, PHATE and PaCMAP wrappers that were
previously scattered across several scripts. The functions are self-contained
and do not depend on ``phase4v2.py`` or the fine-tuning scripts so that those
files can be removed without breaking the pipeline.
"""


import logging
import time
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

# Optional dependencies are imported lazily to avoid costly import time
# (e.g. PaCMAP triggers numba compilation on import).
try:  # pragma: no cover - optional dependency may not be present
    import umap  # type: ignore
except Exception:
    umap = None

# ``phate`` and ``pacmap`` are set to ``None`` and will only be imported when
# the corresponding functions are called.  This prevents slow start-up during
# test collection when those heavy libraries are available.
phate = None  # type: ignore
pacmap = None  # type: ignore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _encode_mixed(df: pd.DataFrame) -> np.ndarray:
    """Return a numeric matrix from ``df`` with scaling and one-hot encoding."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(exclude="number").columns.tolist()

    if numeric_cols:
        stds = df[numeric_cols].std()
        zero_std = stds[stds == 0].index.tolist()
        if zero_std:
            logger.warning("Variables constantes exclues du scaling : %s", zero_std)
            numeric_cols = [c for c in numeric_cols if c not in zero_std]

    X_num = (
        StandardScaler().fit_transform(df[numeric_cols])
        if numeric_cols
        else np.empty((len(df), 0))
    )

    if cat_cols:
        try:
            enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        except TypeError:  # pragma: no cover - older scikit-learn
            enc = OneHotEncoder(handle_unknown="ignore")
        X_cat = enc.fit_transform(df[cat_cols])
    else:
        X_cat = np.empty((len(df), 0))

    if X_num.size and X_cat.size:
        X = np.hstack([X_num, X_cat])
    elif X_num.size:
        X = X_num
    else:
        X = X_cat

    # ensure no NaN values
    if np.isnan(X).any():  # pragma: no cover - should not happen
        X = np.nan_to_num(X)

    return X


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_umap(
    df_active: pd.DataFrame,
    n_components: int = 2,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    *,
    metric: str | None = "euclidean",
    n_jobs: int = -1,
) -> Dict[str, Any]:
    """Run UMAP on ``df_active`` and return model and embeddings.

    The mixed-type dataframe is converted to a purely numeric matrix using
    :func:`_encode_mixed` (standardising numeric columns and one-hot encoding
    categoricals) before fitting UMAP.
    """
    if umap is None:  # pragma: no cover - optional dependency may be absent
        logger.warning("UMAP is not installed; skipping")
        return {
            "model": None,
            "embeddings": pd.DataFrame(index=df_active.index),
            "params": {},
        }

    start = time.perf_counter()
    X = _encode_mixed(df_active)

    if metric is None:
        metric = "euclidean"

    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        n_jobs=n_jobs,
    )
    embedding = reducer.fit_transform(X)
    runtime = time.perf_counter() - start

    cols = [f"U{i + 1}" for i in range(n_components)]
    emb_df = pd.DataFrame(embedding, index=df_active.index, columns=cols)

    params = {
        "n_components": n_components,
        "n_neighbors": n_neighbors,
        "min_dist": min_dist,
        "metric": metric,
    }
    return {
        "model": reducer,
        "embeddings": emb_df,
        "params": params,
        "runtime_s": runtime,
    }


def run_phate(
    df_active: pd.DataFrame,
    n_components: int = 2,
    k: int = 15,
    a: int = 40,
    *,
    t: str | int = "auto",
    knn: int | None = None,
    decay: int | None = None,
) -> Dict[str, Any]:
    """Run PHATE on ``df_active``.

    The dataframe is encoded numerically via :func:`_encode_mixed` before
    fitting PHATE.  The ``knn`` keyword is accepted as an alias for ``k`` and
    ``decay`` can be used as an alias for ``a``. Returns an empty result if the
    library is unavailable.
    """
    global phate
    if phate is None:
        try:  # pragma: no cover - lazy optional import
            import phate as _phate  # type: ignore

            phate = _phate
        except Exception:
            logger.warning("PHATE is not installed; skipping")
            return {
                "model": None,
                "embeddings": pd.DataFrame(index=df_active.index),
                "params": {},
            }

    if knn is not None:
        if isinstance(knn, str):
            try:
                k = int(knn)
            except ValueError:  # invalid string like "auto"
                logger.warning("Invalid PHATE knn value '%s'; using default", knn)
        else:
            k = knn
    if decay is not None:
        a = decay

    start = time.perf_counter()
    X = _encode_mixed(df_active)

    op = phate.PHATE(
        n_components=n_components,
        knn=k,
        decay=a,
        t=t,
        n_jobs=-1,
        verbose=False,
    )
    embedding = op.fit_transform(X)
    runtime = time.perf_counter() - start

    cols = [f"P{i + 1}" for i in range(n_components)]
    emb_df = pd.DataFrame(embedding, index=df_active.index, columns=cols)

    params = {"n_components": n_components, "k": k, "a": a, "t": t}
    return {"model": op, "embeddings": emb_df, "params": params, "runtime_s": runtime}


def run_pacmap(
    df_active: pd.DataFrame,
    n_components: int = 2,
    n_neighbors: int = 10,
    *,
    MN_ratio: float = 0.5,
    FP_ratio: float = 2.0,
    num_iters: Tuple[int, int, int] = (10, 10, 10),
) -> Dict[str, Any]:
    """Run PaCMAP on ``df_active``.

    ``df_active`` is encoded to numeric form via :func:`_encode_mixed` prior to
    fitting. If PaCMAP is unavailable or fails, ``model`` is ``None`` and the
    returned embeddings are empty.
    """
    global pacmap
    if pacmap is None:
        try:  # pragma: no cover - lazy optional import
            import pacmap as _pacmap  # type: ignore

            pacmap = _pacmap
        except Exception:
            logger.warning("PaCMAP is not installed; skipping")
            return {
                "model": None,
                "embeddings": pd.DataFrame(index=df_active.index),
                "params": {},
            }

    start = time.perf_counter()
    X = _encode_mixed(df_active)

    try:
        params = dict(
            n_components=n_components,
            n_neighbors=n_neighbors,
            MN_ratio=MN_ratio,
            FP_ratio=FP_ratio,
            num_iters=num_iters,
            verbose=False,
            apply_pca=True,
        )
        model = pacmap.PaCMAP(**params)
        embedding = model.fit_transform(X)
    except Exception as exc:  # pragma: no cover - rare runtime error
        logger.warning("PaCMAP failed: %s", exc)
        return {
            "model": None,
            "embeddings": pd.DataFrame(index=df_active.index),
            "params": {},
        }

    runtime = time.perf_counter() - start
    cols = [f"C{i + 1}" for i in range(n_components)]
    emb_df = pd.DataFrame(embedding, index=df_active.index, columns=cols)
    params.pop("verbose")
    params.pop("apply_pca")
    return {
        "model": model,
        "embeddings": emb_df,
        "params": params,
        "runtime_s": runtime,
    }


def run_all_nonlinear(df_active: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """Execute all non-linear methods on ``df_active`` and collect the results."""
    results: Dict[str, Dict[str, Any]] = {}

    try:
        results["umap"] = run_umap(df_active)
    except Exception as exc:  # pragma: no cover - unexpected runtime failure
        logger.warning("UMAP failed: %s", exc)
        results["umap"] = {"error": str(exc)}

    try:
        results["phate"] = run_phate(df_active)
    except Exception as exc:  # pragma: no cover - unexpected runtime failure
        logger.warning("PHATE failed: %s", exc)
        results["phate"] = {"error": str(exc)}

    if pacmap is not None:
        try:
            results["pacmap"] = run_pacmap(df_active)
        except Exception as exc:  # pragma: no cover - unexpected runtime failure
            logger.warning("PaCMAP failed: %s", exc)
            results["pacmap"] = {"error": str(exc)}

    return results


# ---------------------------------------------------------------------------
# evaluate_methods.py
# ---------------------------------------------------------------------------
"""Metrics to compare dimensionality reduction methods.
"""

from pathlib import Path
from typing import Any, Dict, Sequence, Iterable, Tuple, Optional
from joblib import Parallel, delayed

import numpy as np
import pandas as pd
from sklearn.manifold import trustworthiness
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture


def tune_kmeans_clusters(
    X: np.ndarray, k_range: Iterable[int] = range(2, 16)
) -> Tuple[np.ndarray, int]:
    """Return K-Means labels using the best silhouette over ``k_range``."""
    best_score = -1.0
    best_labels: Optional[np.ndarray] = None
    best_k = 2
    X = np.asarray(X)
    for k in k_range:
        if k >= len(X) or k < 2:
            continue
        labels = KMeans(n_clusters=k).fit_predict(X)
        if len(np.unique(labels)) < 2:
            score = -1.0
        else:
            score = silhouette_score(X, labels)
        if score > best_score:
            best_score = score
            best_labels = labels
            best_k = k
    if best_labels is None:
        k = max(2, min(len(X), 2))
        best_labels = KMeans(n_clusters=k).fit_predict(X)
        best_k = k
    return best_labels, best_k


def tune_agglomerative_clusters(
    X: np.ndarray, k_range: Iterable[int] = range(2, 16)
) -> Tuple[np.ndarray, int]:
    """Return Agglomerative clustering labels using the best silhouette."""
    best_score = -1.0
    best_labels: Optional[np.ndarray] = None
    best_k = 2
    X = np.asarray(X)
    for k in k_range:
        if k >= len(X) or k < 2:
            continue
        labels = AgglomerativeClustering(n_clusters=k).fit_predict(X)
        if len(np.unique(labels)) < 2:
            score = -1.0
        else:
            score = silhouette_score(X, labels)
        if score > best_score:
            best_score = score
            best_labels = labels
            best_k = k
    if best_labels is None:
        k = max(2, min(len(X), 2))
        best_labels = AgglomerativeClustering(n_clusters=k).fit_predict(X)
        best_k = k
    return best_labels, best_k


def tune_dbscan_clusters(
    X: np.ndarray,
    eps_values: Iterable[float] = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8),
    min_samples: int = 5,
) -> Tuple[np.ndarray, float]:
    """Return DBSCAN labels using the best silhouette over ``eps_values``."""
    best_score = -1.0
    best_labels: Optional[np.ndarray] = None
    best_eps = 0.5
    X = np.asarray(X)
    for eps in eps_values:
        labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        if n_clusters < 2:
            score = -1.0
        else:
            score = silhouette_score(X, labels)
        if score > best_score:
            best_score = score
            best_labels = labels
            best_eps = eps
    if best_labels is None:
        best_labels = DBSCAN(eps=best_eps, min_samples=min_samples).fit_predict(X)
    return best_labels, best_eps


def tune_gmm_clusters(
    X: np.ndarray, k_range: Iterable[int] = range(2, 16)
) -> Tuple[np.ndarray, int]:
    """Return Gaussian Mixture labels using the best silhouette."""
    best_score = -1.0
    best_labels: Optional[np.ndarray] = None
    best_k = 2
    X = np.asarray(X)
    for k in k_range:
        if k >= len(X) or k < 2:
            continue
        gmm = GaussianMixture(n_components=k, covariance_type="full")
        labels = gmm.fit_predict(X)
        if len(np.unique(labels)) < 2:
            score = -1.0
        else:
            score = silhouette_score(X, labels)
        if score > best_score:
            best_score = score
            best_labels = labels
            best_k = k
    if best_labels is None:
        k = max(2, min(len(X), 2))
        best_labels = GaussianMixture(n_components=k).fit_predict(X)
        best_k = k
    return best_labels, best_k


def auto_cluster_labels(
    X: np.ndarray, k_range: Iterable[int] = range(2, 11)
) -> Tuple[np.ndarray, int, str]:
    """Return automatic cluster labels using HDBSCAN when available.

    If the optional :mod:`hdbscan` package is installed and produces at least
    two distinct clusters, it is used. Otherwise the function falls back to a
    K-Means clustering with the number of clusters tuned via silhouette score
    over ``k_range``.
    """

    try:  # pragma: no cover - optional dependency
        import hdbscan  # type: ignore

        clusterer = hdbscan.HDBSCAN()
        labels = clusterer.fit_predict(X)
        if len(np.unique(labels)) > 1:
            return labels, len(np.unique(labels)), "hdbscan"
    except Exception:  # pragma: no cover - missing library
        labels = None

    km_labels, km_k = tune_kmeans_clusters(X, k_range)
    return km_labels, km_k, "kmeans"


def dunn_index(X: np.ndarray, labels: np.ndarray) -> float:
    """Compute the Dunn index of a clustering.

    Parameters
    ----------
    X:
        Coordinates of the points.
    labels:
        Cluster labels for each point.

    Returns
    -------
    float
        Dunn index (higher is better). ``NaN`` if undefined.
    """
    from scipy.spatial.distance import pdist, squareform

    if len(np.unique(labels)) < 2:
        return float("nan")

    dist = squareform(pdist(X))
    unique = np.unique(labels)

    intra_diam = []
    min_inter = np.inf

    for i, ci in enumerate(unique):
        idx_i = np.where(labels == ci)[0]
        if len(idx_i) > 1:
            intra = dist[np.ix_(idx_i, idx_i)].max()
        else:
            intra = 0.0
        intra_diam.append(intra)

        for cj in unique[i + 1 :]:
            idx_j = np.where(labels == cj)[0]
            inter = dist[np.ix_(idx_i, idx_j)].min()
            if inter < min_inter:
                min_inter = inter

    max_intra = max(intra_diam)
    if max_intra == 0:
        return float("nan")
    return float(min_inter / max_intra)


def evaluate_methods(
    results_dict: Dict[str, Dict[str, Any]],
    df_active: pd.DataFrame,
    quant_vars: Sequence[str],
    qual_vars: Sequence[str],
    *,
    k_range: Iterable[int] = range(2, 16),
) -> pd.DataFrame:
    """Compute comparison metrics for each dimensionality reduction method.

    Parameters
    ----------
    results_dict:
        Mapping of method name to a dictionary containing at least ``embeddings``
        as a DataFrame. Optionally ``inertia`` (list or Series) and
        ``runtime_s`` or ``runtime``.
    df_active:
        Original high dimensional dataframe.
    quant_vars:
        Names of quantitative variables in ``df_active``.
    qual_vars:
        Names of qualitative variables in ``df_active``.
    k_range:
        Range of ``k`` values to test when tuning the K-Means clustering.

    Returns
    -------
    pandas.DataFrame
        Metrics table indexed by method name.
    """
    rows = []
    n_features = len(quant_vars) + len(qual_vars)

    logger = logging.getLogger(__name__)

    def _process(item: tuple[str, Dict[str, Any]]) -> tuple[str, np.ndarray, str, Dict[str, Any]]:
        method, info = item

        inertias = info.get("inertia")
        if inertias is None:
            inertias = []
        if isinstance(inertias, pd.Series):
            inertias = inertias.tolist()
        inertias = list(inertias)

        if inertias and inertias[0] > 0.5:
            logger.warning(
                "Attention : l'axe F1 de %s explique %.1f%% de la variance",
                method.upper(),
                inertias[0] * 100,
            )

        contrib = info.get("contributions")
        if isinstance(contrib, pd.DataFrame) and "F1" in contrib:
            top = float(contrib["F1"].max())
            if top > 50:
                dom_var = contrib["F1"].idxmax()
                logger.warning(
                    "Attention : la variable %s domine l’axe F1 de la %s avec %.1f%% de contribution, risque de projection biaisée.",
                    dom_var,
                    method.upper(),
                    top,
                )

        kaiser = int(sum(1 for eig in np.array(inertias) * n_features if eig > 1))
        cum_inertia = float(sum(inertias) * 100) if inertias else np.nan

        X_low = info["embeddings"].values
        labels, best_k, algo = auto_cluster_labels(X_low, k_range)
        info["cluster_labels"] = labels
        info["cluster_k"] = best_k
        info["cluster_algo"] = algo
        if len(labels) <= best_k or len(set(labels)) < 2:
            sil = float("nan")
            dunn = float("nan")
        else:
            sil = float(silhouette_score(X_low, labels))
            dunn = dunn_index(X_low, labels)

        try:
            enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        except TypeError:  # pragma: no cover - older scikit-learn
            enc = OneHotEncoder(handle_unknown="ignore")
        X_num = StandardScaler().fit_transform(
            df_active.loc[info["embeddings"].index, quant_vars]
        )
        X_cat = (
            enc.fit_transform(df_active.loc[info["embeddings"].index, qual_vars])
            if qual_vars
            else np.empty((len(X_low), 0))
        )
        X_high = np.hstack([X_num, X_cat])
        k_nn = min(10, max(1, len(X_high) // 2))
        if k_nn >= len(X_high) / 2:
            T = float("nan")
            C = float("nan")
        else:
            T = float(trustworthiness(X_high, X_low, n_neighbors=k_nn))
            C = float(trustworthiness(X_low, X_high, n_neighbors=k_nn))

        runtime = (
            info.get("runtime_seconds") or info.get("runtime_s") or info.get("runtime")
        )

        row = {
            "method": method,
            "variance_cumulee_%": cum_inertia,
            "nb_axes_kaiser": kaiser,
            "silhouette": sil,
            "dunn_index": dunn,
            "trustworthiness": T,
            "continuity": C,
            "runtime_seconds": runtime,
            "cluster_k": best_k,
            "cluster_algo": algo,
        }
        return method, labels, row

    parallel_res = Parallel(n_jobs=-1)(delayed(_process)(it) for it in results_dict.items())
    rows = []
    for method, labels, row in parallel_res:
        results_dict[method]["cluster_labels"] = labels
        results_dict[method]["cluster_k"] = row["cluster_k"]
        results_dict[method]["cluster_algo"] = row["cluster_algo"]
        rows.append(row)
    df_metrics = pd.DataFrame(rows).set_index("method")
    return df_metrics


def plot_methods_heatmap(df_metrics: pd.DataFrame, output_path: str | Path) -> None:
    """Plot a normalized heatmap of ``df_metrics``.

    Parameters
    ----------
    df_metrics:
        DataFrame as returned by :func:`evaluate_methods`.
    output_path:
        Directory where ``methods_heatmap.png`` will be saved.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    output = Path(output_path)
    output.mkdir(parents=True, exist_ok=True)

    df_norm = df_metrics.select_dtypes(include=[np.number]).copy()
    for col in df_norm.columns:
        if not pd.api.types.is_numeric_dtype(df_norm[col]):
            df_norm[col] = 0.0
            continue
        cmin, cmax = df_norm[col].min(), df_norm[col].max()
        if pd.isna(cmin) or cmax == cmin:
            df_norm[col] = 0.0
        else:
            df_norm[col] = (df_norm[col] - cmin) / (cmax - cmin)

    # keep the original numeric values for annotations before normalization
    annot = df_metrics[df_norm.columns].copy()
    if "variance_cumulee_%" in annot:
        annot["variance_cumulee_%"] = (
            annot["variance_cumulee_%"].round().astype("Int64")
        )
    if "nb_axes_kaiser" in annot:
        annot["nb_axes_kaiser"] = annot["nb_axes_kaiser"].astype("Int64")
    for col in annot.columns:
        if col not in {"variance_cumulee_%", "nb_axes_kaiser"} and pd.api.types.is_numeric_dtype(annot[col]):
            annot[col] = annot[col].map(lambda x: f"{x:.2f}" if pd.notna(x) else "")

    fig, ax = plt.subplots(figsize=(12, 6), dpi=200)
    sns.heatmap(
        df_norm,
        annot=annot,
        fmt="",
        cmap="coolwarm",
        vmin=0,
        vmax=1,
        ax=ax,
    )
    ax.set_title("Comparaison des méthodes")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    fig.savefig(output / "methods_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# visualization.py
# ---------------------------------------------------------------------------
"""Comparative visualization utilities for dimensionality reduction results."""


import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from matplotlib.lines import Line2D
import pandas as pd
import seaborn as sns
import numpy as np
from typing import Dict, Any, List, Optional, Sequence
from sklearn.cluster import KMeans


def plot_correlation_circle(
    factor_model: Any,
    quant_vars: Sequence[str],
    output_path: str | Path,
    *,
    coords: Optional[pd.DataFrame] = None,
) -> Path:
    """Generate and save a correlation circle for ``factor_model``.

    Parameters
    ----------
    factor_model:
        Fitted model exposing ``components_`` or ``column_correlations_``.
    quant_vars:
        Names of quantitative variables to include.
    output_path:
        Destination path for the created figure.
    coords : pandas.DataFrame, optional
        Pre-computed coordinates for the variables. When provided, the
        ``factor_model`` is only used to extract the explained variance for the
        title and may lack ``components_`` or ``column_correlations_``.
    """

    if coords is not None:
        coords = coords.loc[[v for v in quant_vars if v in coords.index], ["F1", "F2"]]
    elif hasattr(factor_model, "column_correlations_"):
        coords = pd.DataFrame(
            factor_model.column_correlations_,
            columns=["F1", "F2"],
        )
        coords = coords.loc[[v for v in quant_vars if v in coords.index]]
    elif hasattr(factor_model, "components_"):
        comps = np.asarray(factor_model.components_, dtype=float)
        names = getattr(factor_model, "feature_names_in_", list(quant_vars))
        eig = getattr(factor_model, "explained_variance_", None)
        if eig is not None:
            load = comps[:2].T * np.sqrt(eig[:2])
        else:
            load = comps[:2].T
        coords = pd.DataFrame(load, index=names, columns=["F1", "F2"])
        coords = coords.loc[[v for v in quant_vars if v in coords.index]]
    elif hasattr(factor_model, "column_coordinates_"):
        tmp = pd.DataFrame(factor_model.column_coordinates_, copy=True)
        tmp.columns = ["F" + str(i + 1) for i in range(tmp.shape[1])]
        coords = tmp.loc[[v for v in quant_vars if v in tmp.index], ["F1", "F2"]]
    else:  # pragma: no cover - unexpected model type
        raise AttributeError("factor_model lacks components")

    norms = np.sqrt(np.square(coords["F1"]) + np.square(coords["F2"]))
    scale = float(norms.max()) if len(norms) else 1.0

    # Use a single reference circle centred at the origin.  The radius is fixed
    # to 1 so that the correlation circle is not cluttered with additional
    # cos² circles.
    fig, ax = plt.subplots(figsize=(6, 6), dpi=200)

    if not any(isinstance(p, plt.Circle) and np.isclose(p.radius, 1.0) for p in ax.patches):
        circle = plt.Circle((0, 0), 1.0, color="grey", fill=False, linestyle="dashed")
        ax.add_patch(circle)
    ax.axhline(0, color="grey", lw=0.5)
    ax.axvline(0, color="grey", lw=0.5)

    palette = sns.color_palette("husl", len(coords))
    handles: list[Line2D] = []
    for var, color, norm in zip(coords.index, palette, norms):
        x, y = coords.loc[var, ["F1", "F2"]]
        alpha = 0.4 + 0.6 * (norm / scale) if scale else 1.0
        ax.arrow(
            0,
            0,
            x,
            y,
            head_width=0.02 * scale,
            length_includes_head=True,
            width=0.001 * scale,
            linewidth=0.5,
            color=color,
            alpha=alpha,
        )
        handles.append(Line2D([0], [0], color=color, lw=1.0, label=str(var)))

    ax.legend(
        handles=handles,
        loc="upper right",
        bbox_to_anchor=(1.3, 1.0),
        frameon=False,
        fontsize="small",
    )
    limit = max(scale, 1.0) * 1.1
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_xlabel("F1")
    ax.set_ylabel("F2")


    method_name = factor_model.__class__.__name__.upper()
    if hasattr(factor_model, "explained_variance_ratio_"):
        inertia = np.asarray(
            getattr(factor_model, "explained_variance_ratio_"), dtype=float
        )
    else:
        inertia = np.asarray(_get_explained_inertia(factor_model), dtype=float)
    var2 = float(np.sum(inertia[:2]) * 100) if inertia.size else 0.0
    ax.set_title(
        f"Cercle des corrélations – {method_name} (F1+F2 = {var2:.1f} % de variance)"
    )
    ax.set_aspect("equal")
    fig.tight_layout()

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300)
    plt.close(fig)
    return output


def _choose_color_var(df: pd.DataFrame, qual_vars: List[str]) -> Optional[str]:
    """Return a qualitative variable available in ``df`` to colour scatter plots."""
    preferred = [
        "Statut production",
        "Statut commercial",
        "Type opportunité",
    ]
    for col in preferred:
        if col in df.columns:
            return col
    for col in qual_vars:
        if col in df.columns:
            return col
    return None


def plot_scatter_2d(
    emb_df: pd.DataFrame, df_active: pd.DataFrame, color_var: Optional[str], title: str
) -> plt.Figure:
    """Return a 2D scatter plot figure coloured by ``color_var``."""
    fig, ax = plt.subplots(figsize=(12, 6), dpi=200)
    if color_var is None or color_var not in df_active.columns:
        ax.scatter(
            emb_df.iloc[:, 0],
            emb_df.iloc[:, 1],
            s=10,
            alpha=0.6,
            color="tab:blue",
        )
    else:
        cats = df_active.loc[emb_df.index, color_var].astype("category")
        palette = sns.color_palette("tab10", len(cats.cat.categories))
        for cat, color in zip(cats.cat.categories, palette):
            mask = cats == cat
            ax.scatter(
                emb_df.loc[mask, emb_df.columns[0]],
                emb_df.loc[mask, emb_df.columns[1]],
                s=10,
                alpha=0.6,
                color=color,
                label=str(cat),
            )
        if str(color_var).lower().startswith("sous-"):
            ax.legend(
                title=color_var,
                bbox_to_anchor=(0.5, -0.15),
                loc="upper center",
                ncol=3,
            )
        else:
            ax.legend(title=color_var, bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.set_xlabel(emb_df.columns[0])
    ax.set_ylabel(emb_df.columns[1])
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_scatter_3d(
    emb_df: pd.DataFrame, df_active: pd.DataFrame, color_var: Optional[str], title: str
) -> plt.Figure:
    """Return a 3D scatter plot figure coloured by ``color_var``."""
    fig = plt.figure(figsize=(12, 6), dpi=200)
    ax = fig.add_subplot(111, projection="3d")
    if color_var is None or color_var not in df_active.columns:
        ax.scatter(
            emb_df.iloc[:, 0],
            emb_df.iloc[:, 1],
            emb_df.iloc[:, 2],
            s=10,
            alpha=0.6,
            color="tab:blue",
        )
    else:
        cats = df_active.loc[emb_df.index, color_var].astype("category")
        palette = sns.color_palette("tab10", len(cats.cat.categories))
        for cat, color in zip(cats.cat.categories, palette):
            mask = cats == cat
            ax.scatter(
                emb_df.loc[mask, emb_df.columns[0]],
                emb_df.loc[mask, emb_df.columns[1]],
                emb_df.loc[mask, emb_df.columns[2]],
                s=10,
                alpha=0.6,
                color=color,
                label=str(cat),
            )
        if str(color_var).lower().startswith("sous-"):
            ax.legend(
                title=color_var,
                bbox_to_anchor=(0.5, -0.1),
                loc="upper center",
                ncol=3,
            )
        else:
            ax.legend(title=color_var, bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.set_xlabel(emb_df.columns[0])
    ax.set_ylabel(emb_df.columns[1])
    ax.set_zlabel(emb_df.columns[2])
    ax.set_title(title)
    ax.view_init(elev=20, azim=60)
    fig.tight_layout()
    return fig


def plot_cluster_scatter_3d(
    emb_df: pd.DataFrame, labels: np.ndarray, title: str
) -> plt.Figure:
    """Return a 3D scatter plot coloured by cluster labels."""
    fig = plt.figure(figsize=(12, 6), dpi=200)
    ax = fig.add_subplot(111, projection="3d")
    unique = np.unique(labels)
    try:
        cmap = matplotlib.colormaps.get_cmap("tab10")
    except AttributeError:  # Matplotlib < 3.6
        cmap = matplotlib.cm.get_cmap("tab10")
    n_colors = cmap.N if hasattr(cmap, "N") else len(unique)
    centroids = []
    for i, lab in enumerate(unique):
        mask = labels == lab
        ax.scatter(
            emb_df.loc[mask, emb_df.columns[0]],
            emb_df.loc[mask, emb_df.columns[1]],
            emb_df.loc[mask, emb_df.columns[2]],
            s=10,
            alpha=0.6,
            color=cmap(i % n_colors),
            label=str(lab),
        )
        centroid = emb_df.loc[mask, emb_df.columns[:3]].mean().values
        centroids.append(centroid)
    if centroids:
        centroids = np.vstack(centroids)
        ax.scatter(
            centroids[:, 0],
            centroids[:, 1],
            centroids[:, 2],
            marker="x",
            s=60,
            color="black",
            zorder=3,
        )
    ax.legend(title="cluster", bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.set_xlabel(emb_df.columns[0])
    ax.set_ylabel(emb_df.columns[1])
    ax.set_zlabel(emb_df.columns[2])
    ax.set_title(title)
    ax.view_init(elev=20, azim=60)
    fig.tight_layout()
    return fig


def plot_cluster_scatter(
    emb_df: pd.DataFrame, labels: np.ndarray, title: str
) -> plt.Figure:
    """Return a 2D scatter plot coloured by cluster labels.

    Parameters
    ----------
    emb_df : pandas.DataFrame
        Embedding coordinates with at least two columns.
    labels : array-like
        Cluster labels for each observation.
    title : str
        Title of the figure.
    """
    fig, ax = plt.subplots(figsize=(12, 6), dpi=200)
    unique = np.unique(labels)
    try:
        cmap = matplotlib.colormaps.get_cmap("tab10")
    except AttributeError:  # Matplotlib < 3.6
        cmap = matplotlib.cm.get_cmap("tab10")
    n_colors = cmap.N if hasattr(cmap, "N") else len(unique)
    centroids = []
    for i, lab in enumerate(unique):
        mask = labels == lab
        ax.scatter(
            emb_df.loc[mask, emb_df.columns[0]],
            emb_df.loc[mask, emb_df.columns[1]],
            s=10,
            alpha=0.6,
            color=cmap(i % n_colors),
            label=str(lab),
        )
        centroid = emb_df.loc[mask, emb_df.columns[:2]].mean().values
        centroids.append(centroid)

    if centroids:
        centroids = np.vstack(centroids)
        ax.scatter(
            centroids[:, 0],
            centroids[:, 1],
            marker="x",
            s=60,
            color="black",
            zorder=3,
        )
    ax.legend(title="cluster", bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.set_xlabel(emb_df.columns[0])
    ax.set_ylabel(emb_df.columns[1])
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_cluster_scatter_3d(
    emb_df: pd.DataFrame, labels: np.ndarray, title: str
) -> plt.Figure:
    """Return a 3D scatter plot coloured by clusters."""
    fig = plt.figure(figsize=(12, 6), dpi=200)
    ax = fig.add_subplot(111, projection="3d")
    unique = np.unique(labels)
    try:
        cmap = matplotlib.colormaps.get_cmap("tab10")
    except AttributeError:  # pragma: no cover - older Matplotlib
        cmap = matplotlib.cm.get_cmap("tab10")
    n_colors = cmap.N if hasattr(cmap, "N") else len(unique)
    for i, lab in enumerate(unique):
        mask = labels == lab
        ax.scatter(
            emb_df.loc[mask, emb_df.columns[0]],
            emb_df.loc[mask, emb_df.columns[1]],
            emb_df.loc[mask, emb_df.columns[2]],
            s=10,
            alpha=0.6,
            color=cmap(i % n_colors),
            label=str(lab),
        )
    ax.legend(title="cluster", bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.set_xlabel(emb_df.columns[0])
    ax.set_ylabel(emb_df.columns[1])
    ax.set_zlabel(emb_df.columns[2])
    ax.set_title(title)
    ax.view_init(elev=20, azim=60)
    fig.tight_layout()
    return fig


def plot_cluster_distribution(labels: np.ndarray, title: str) -> plt.Figure:
    """Return a bar chart showing the count of points per cluster."""
    unique, counts = np.unique(labels, return_counts=True)
    fig, ax = plt.subplots(figsize=(6, 4), dpi=200)
    try:
        cmap = matplotlib.colormaps.get_cmap("tab10")
    except AttributeError:  # pragma: no cover - older Matplotlib
        cmap = matplotlib.cm.get_cmap("tab10")
    n_colors = cmap.N if hasattr(cmap, "N") else len(unique)
    colors = [cmap(i % n_colors) for i in range(len(unique))]
    positions = range(len(unique))
    ax.bar(positions, counts, color=colors, edgecolor="black")
    ax.set_xticks(list(positions))
    ax.set_xticklabels([str(u) for u in unique])
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Effectif")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_cluster_grid(
    emb_df: pd.DataFrame,
    km_labels: np.ndarray,
    ag_labels: np.ndarray,
    db_labels: np.ndarray,
    method: str,
    km_k: int,
    ag_k: int,
    db_eps: float,
) -> plt.Figure:
    """Return a 2x2 grid comparing clustering algorithms."""

    fig, axes = plt.subplots(2, 2, figsize=(12, 12), dpi=200)
    axes = axes.ravel()

    # Baseline
    axes[0].scatter(
        emb_df.iloc[:, 0],
        emb_df.iloc[:, 1],
        s=10,
        alpha=0.6,
        color="tab:blue",
    )
    axes[0].set_title(f"{method.upper()} \u2013 No Clustering")

    def _plot(ax: plt.Axes, labels: np.ndarray, title: str) -> None:
        unique = np.unique(labels)
        try:
            cmap = matplotlib.colormaps.get_cmap("tab10")
        except AttributeError:  # pragma: no cover - older Matplotlib
            cmap = matplotlib.cm.get_cmap("tab10")
        n_colors = cmap.N if hasattr(cmap, "N") else len(unique)
        for i, lab in enumerate(unique):
            mask = labels == lab
            color = "lightgray" if lab == -1 else cmap(i % n_colors)
            ax.scatter(
                emb_df.loc[mask, emb_df.columns[0]],
                emb_df.loc[mask, emb_df.columns[1]],
                s=10,
                alpha=0.6,
                color=color,
                label=str(lab),
            )
        ax.set_xlabel(emb_df.columns[0])
        ax.set_ylabel(emb_df.columns[1])
        ax.set_title(title)

    _plot(
        axes[1],
        km_labels,
        f"{method.upper()} \u2013 K-Means (k={km_k})",
    )
    _plot(
        axes[2],
        ag_labels,
        f"{method.upper()} \u2013 Agglomerative (n={ag_k})",
    )
    _plot(
        axes[3],
        db_labels,
        f"{method.upper()} \u2013 DBSCAN (ε={db_eps:g})",
    )

    for ax in axes:
        ax.legend(title="cluster", bbox_to_anchor=(1.05, 1), loc="upper left")

    fig.tight_layout()
    return fig


def cluster_segment_table(
    labels: Sequence[int] | pd.Series,
    segments: Sequence[str] | pd.Series,
) -> pd.DataFrame:
    """Return a cross-tabulation of segments per cluster."""
    if len(labels) != len(segments):
        raise ValueError("labels and segments must have same length")
    ser_labels = pd.Series(labels, name="cluster")
    ser_segments = pd.Series(segments, name="segment")
    return pd.crosstab(ser_labels, ser_segments)


def plot_cluster_segment_heatmap(
    table: pd.DataFrame, title: str
) -> plt.Figure:
    """Return a heatmap visualising ``table`` counts."""
    import seaborn as sns

    fig, ax = plt.subplots(figsize=(8, 4), dpi=200)
    sns.heatmap(table, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Segment")
    ax.set_ylabel("Cluster")
    fig.tight_layout()
    return fig


def _extract_quant_coords(coords: pd.DataFrame, quant_vars: List[str]) -> pd.DataFrame:
    """Extract F1/F2 coordinates for quantitative variables if available."""
    cols = [c for c in ["F1", "F2"] if c in coords.columns]
    if len(cols) < 2:
        # fall back to the first available columns
        extra = [c for c in coords.columns if c not in cols][: 2 - len(cols)]
        cols.extend(extra)
    if len(cols) < 2:
        return pd.DataFrame(columns=["F1", "F2"])
    subset = coords.loc[[v for v in quant_vars if v in coords.index], cols]
    subset = subset.rename(columns={cols[0]: "F1", cols[1]: "F2"})
    return subset


def _corr_from_embeddings(
    emb: pd.DataFrame, df_active: pd.DataFrame, quant_vars: List[str]
) -> pd.DataFrame:
    """Return correlations of quantitative variables with the first two dims."""
    if emb.shape[1] < 2:
        return pd.DataFrame(columns=["F1", "F2"])
    data = {}
    f1 = emb.iloc[:, 0]
    f2 = emb.iloc[:, 1]
    for var in quant_vars:
        if var in df_active.columns:
            series = df_active.loc[emb.index, var]
            data[var] = [series.corr(f1), series.corr(f2)]
    if not data:
        return pd.DataFrame(columns=["F1", "F2"])
    return pd.DataFrame(data, index=["F1", "F2"]).T


def plot_scree(
    explained_variance: Sequence[float] | pd.Series,
    method_name: str,
    output_path: str | Path,
) -> Path:
    """Generate and save a scree plot for ``method_name``.

    The function writes the image to ``output_path`` and returns that path.
    ``explained_variance`` can be a sequence of eigenvalues or variance ratios.
    """

    values = np.asarray(explained_variance, dtype=float)
    if values.max() > 1.0:
        ratios = values / values.sum()
    else:
        ratios = values

    axes = np.arange(1, len(ratios) + 1)
    fig, ax = plt.subplots(figsize=(12, 6), dpi=200)
    ax.bar(
        axes,
        ratios * 100,
        color=sns.color_palette("deep")[0],
        edgecolor="black",
    )
    cum = np.cumsum(ratios)
    ax.plot(axes, cum * 100, "-o", color="#C04000")

    if values.max() > 1.0:
        ax.axhline(1, color="red", ls="--", lw=0.8, label="Kaiser")

    # The 80% cumulative inertia marker is shown as a horizontal line at 80 on
    # the percentage scale (or ``0.8`` if the axis uses fractions).  Because the
    # y-axis here is expressed in percent, the line is drawn at ``y=80``.  For
    # MFA this replaces a former vertical marker placed at the component index
    # where cumulative inertia reached 80%.
    ax.axhline(80, color="green", ls="--", lw=0.8, label="80% cumul")

    ax.set_xlabel("Composante")
    ax.set_ylabel("% Variance expliquée")
    ax.set_title(f"Éboulis des variances – {method_name}")
    ax.set_xticks(list(axes))
    ax.legend(loc="upper right")
    fig.tight_layout()

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300)
    plt.close(fig)
    return output


def plot_famd_contributions(contrib: pd.DataFrame, n: int = 10) -> plt.Figure:
    """Return a bar plot of variable contributions to F1 and F2.

    ``contrib`` is typically obtained from ``prince.FAMD.column_contributions``
    and may have integer component labels.  The function is tolerant to the
    number of components provided and only uses the first two.  When only one
    component is present a zero-filled second component is added to avoid index
    errors.
    """

    # Normalise column names to ``F1``/``F2`` ------------------------------
    if not {"F1", "F2"}.issubset(contrib.columns):
        cols = list(contrib.columns[:2])
        rename = {}
        if cols:
            rename[cols[0]] = "F1"
        if len(cols) > 1:
            rename[cols[1]] = "F2"
        contrib = contrib.rename(columns=rename)
    if "F2" not in contrib.columns:
        contrib["F2"] = 0.0

    # Aggregate contributions by variable --------------------------------
    grouped: dict[str, pd.Series] = {}
    for idx in contrib.index:
        var = idx.split("__", 1)[0]
        grouped.setdefault(var, pd.Series(dtype=float))
        grouped[var] = grouped[var].add(
            contrib.loc[idx, ["F1", "F2"]], fill_value=0
        )
    df = pd.DataFrame(grouped).T.fillna(0)

    # Order by total contribution and keep top ``n`` ---------------------
    sort_index = df.sum(axis=1).sort_values(ascending=False).index
    df = df.loc[sort_index].iloc[:n]

    fig, ax = plt.subplots(figsize=(12, 6), dpi=200)
    df[["F1", "F2"]].plot(kind="bar", ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_ylabel("% Contribution")
    ax.set_title("Contributions des variables – FAMD (F1 et F2)")
    ax.legend(title="Axe")
    fig.tight_layout()
    return fig


def plot_embedding(
    coords_df: pd.DataFrame,
    color_by: Optional[Sequence[Any]] = None,
    title: str = "",
    output_path: str | Path = "",
) -> Path:
    """Generate and save a 2D scatter plot from ``coords_df``.

    Parameters
    ----------
    coords_df:
        DataFrame with two columns representing the embedding.
    color_by:
        Optional sequence of labels used to colour the points.
    title:
        Title of the figure.
    output_path:
        Destination path for the saved image.
    """

    fig, ax = plt.subplots(figsize=(12, 6), dpi=200)
    if color_by is None:
        ax.scatter(coords_df.iloc[:, 0], coords_df.iloc[:, 1], s=10, alpha=0.6)
    else:
        labels = pd.Series(list(color_by), index=coords_df.index)
        if labels.dtype.kind in {"O", "b"} or str(labels.dtype).startswith("category"):
            cats = labels.astype("category")
            palette = sns.color_palette("tab10", len(cats.cat.categories))
            for cat, color in zip(cats.cat.categories, palette):
                mask = cats == cat
                ax.scatter(
                    coords_df.loc[mask, coords_df.columns[0]],
                    coords_df.loc[mask, coords_df.columns[1]],
                    s=10,
                    alpha=0.6,
                    color=color,
                    label=str(cat),
                )
            ax.legend(
                title=getattr(labels, "name", ""),
                bbox_to_anchor=(1.05, 1),
                loc="upper left",
            )
        else:
            sc = ax.scatter(
                coords_df.iloc[:, 0],
                coords_df.iloc[:, 1],
                c=labels,
                cmap="viridis",
                s=10,
                alpha=0.6,
            )
            fig.colorbar(sc, ax=ax)

    ax.set_xlabel(coords_df.columns[0])
    ax.set_ylabel(coords_df.columns[1])
    ax.set_title(title)
    fig.tight_layout()

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300)
    plt.close(fig)
    return output


def generate_figures(
    factor_results: Dict[str, Dict[str, Any]],
    nonlin_results: Dict[str, Dict[str, Any]],
    df_active: pd.DataFrame,
    quant_vars: List[str],
    qual_vars: List[str],
    output_dir: Optional[Path] = None,
    *,
    cluster_k: int | None = None,
    segment_col: str | None = None,
) -> Dict[str, plt.Figure]:
    """Generate and optionally save comparative visualization figures.

    Parameters
    ----------
    output_dir : Path or None, optional
        Directory where figures will be saved.
    cluster_k : int or None, optional
        If ``None``, the number of clusters is tuned automatically up to a
        maximum of 10. Otherwise ``cluster_k`` is treated as the upper bound for
        the search range.
    segment_col : str or None, optional
        Name of the column in ``df_active`` containing business segments.
        When provided, a heatmap comparing clusters to segments is generated for
        each method.
    """
    color_var = None
    figures: Dict[str, plt.Figure] = {}
    out = Path(output_dir) if output_dir is not None else None
    segments = (
        df_active[segment_col]
        if segment_col is not None and segment_col in df_active.columns
        else None
    )

    def _save(fig: plt.Figure, method: str, name: str) -> None:
        if out is None:
            return
        sub = out / method.lower()
        sub.mkdir(parents=True, exist_ok=True)
        fig.savefig(sub / f"{name}.png", dpi=300)
        plt.close(fig)

    for method, res in factor_results.items():
        emb = res.get("embeddings")
        if isinstance(emb, pd.DataFrame) and emb.shape[1] >= 2:
            title = f"Projection des affaires – {method.upper()}"
            fig = plot_scatter_2d(emb.iloc[:, :2], df_active, color_var, title)
            figures[f"{method}_scatter_2d"] = fig
            _save(fig, method, f"{method}_scatter_2d")
            max_k = cluster_k if cluster_k is not None else min(15, len(emb) - 1)
            km_labels, km_k = tune_kmeans_clusters(emb.iloc[:, :2].values, range(2, max_k + 1))
            ag_labels, ag_k = tune_agglomerative_clusters(emb.iloc[:, :2].values, range(2, max_k + 1))
            db_labels, db_eps = tune_dbscan_clusters(emb.iloc[:, :2].values)

            grid_fig = plot_cluster_grid(
                emb.iloc[:, :2],
                km_labels,
                ag_labels,
                db_labels,
                method,
                km_k,
                ag_k,
                db_eps,
            )
            figures[f"{method}_cluster_comparison"] = grid_fig
            _save(grid_fig, method, f"{method}_cluster_comparison")

            labels = km_labels
            dist_fig = plot_cluster_distribution(
                labels,
                f"Répartition des segments – {method.upper()} (K-Means)",
            )
            figures[f"{method}_cluster_dist"] = dist_fig
            _save(dist_fig, method, f"{method}_cluster_dist_kmeans")
            if segments is not None:
                table = cluster_segment_table(labels, segments.loc[emb.index])
                heat = plot_cluster_segment_heatmap(
                    table,
                    f"Segments vs clusters – {method.upper()} (K-Means)",
                )
                figures[f"{method}_cluster_segments"] = heat
                _save(heat, method, f"{method}_cluster_segments_kmeans")
            if emb.shape[1] >= 3:
                fig3d = plot_scatter_3d(
                    emb.iloc[:, :3],
                    df_active,
                    color_var,
                    f"Projection 3D – {method.upper()}",
                )
                figures[f"{method}_scatter_3d"] = fig3d
                _save(fig3d, method, f"{method}_scatter_3d")
                cfig3d = plot_cluster_scatter_3d(
                    emb.iloc[:, :3],
                    km_labels,
                    f"Projection 3D – {method.upper()} (K-Means, k={km_k})",
                )
                figures[f"{method}_clusters_3d"] = cfig3d
                _save(cfig3d, method, f"{method}_clusters_kmeans_k{km_k}_3d")
        coords = res.get("loadings")
        if coords is None:
            coords = res.get("column_coords")
        if isinstance(coords, pd.DataFrame):
            qcoords = _extract_quant_coords(coords, quant_vars)
            if qcoords.empty and isinstance(emb, pd.DataFrame):
                qcoords = _corr_from_embeddings(emb, df_active, quant_vars)
        elif isinstance(emb, pd.DataFrame):
            qcoords = _corr_from_embeddings(emb, df_active, quant_vars)
        else:
            qcoords = pd.DataFrame()
        if not qcoords.empty and "model" in res:
            dest = (
                Path(output_dir) / method.lower() / f"{method}_correlation.png"
                if output_dir
                else Path(f"{method}_correlation.png")
            )
            path = plot_correlation_circle(
                res["model"], quant_vars, dest, coords=qcoords
            )
            figures[f"{method}_correlation"] = path
        inertia = res.get("inertia")
        if isinstance(inertia, pd.Series) and not inertia.empty:
            dest = (
                Path(output_dir) / method.lower() / f"{method}_scree.png"
                if output_dir
                else Path(f"{method}_scree.png")
            )
            path = plot_scree(inertia, method.upper(), dest)
            figures[f"{method}_scree"] = path
        if method == "famd":
            contrib = res.get("contributions")
            if isinstance(contrib, pd.DataFrame) and not contrib.empty:
                fig_contrib = plot_famd_contributions(contrib)
                figures[f"{method}_contributions"] = fig_contrib
                _save(fig_contrib, method, f"{method}_contributions")

    for method, res in nonlin_results.items():
        emb = res.get("embeddings")
        if isinstance(emb, pd.DataFrame) and emb.shape[1] >= 2:
            title = f"Projection des affaires – {method.upper()}"
            fig = plot_scatter_2d(emb.iloc[:, :2], df_active, color_var, title)
            figures[f"{method}_scatter_2d"] = fig
            _save(fig, method, f"{method}_scatter_2d")
            max_k = cluster_k if cluster_k is not None else min(15, len(emb) - 1)
            km_labels, km_k = tune_kmeans_clusters(emb.iloc[:, :2].values, range(2, max_k + 1))
            ag_labels, ag_k = tune_agglomerative_clusters(emb.iloc[:, :2].values, range(2, max_k + 1))
            db_labels, db_eps = tune_dbscan_clusters(emb.iloc[:, :2].values)

            grid_fig = plot_cluster_grid(
                emb.iloc[:, :2],
                km_labels,
                ag_labels,
                db_labels,
                method,
                km_k,
                ag_k,
                db_eps,
            )
            labels = km_labels
            title = (
                f"Projection {method.upper()} – coloration par clusters (K-Means, k={km_k})"
            )
            save_name = f"{method}_clusters_kmeans_k{km_k}"
            cfig = plot_cluster_scatter(emb.iloc[:, :2], labels, title)
            figures[save_name] = cfig
            _save(cfig, method, save_name)
            if segments is not None:
                table = cluster_segment_table(labels, segments.loc[emb.index])
                heat = plot_cluster_segment_heatmap(
                    table,
                    f"Segments vs clusters – {method.upper()} (K-Means)",
                )
                figures[f"{method}_cluster_segments"] = heat
                _save(heat, method, f"{method}_cluster_segments_kmeans")
            if emb.shape[1] >= 3:
                fig3d = plot_scatter_3d(
                    emb.iloc[:, :3],
                    df_active,
                    color_var,
                    f"Projection 3D – {method.upper()}",
                )
                figures[f"{method}_scatter_3d"] = fig3d
                _save(fig3d, method, f"{method}_scatter_3d")
                cfig3d = plot_cluster_scatter_3d(
                    emb.iloc[:, :3],
                    km_labels,
                    f"Projection 3D – {method.upper()} (K-Means, k={km_k})",
                )
                figures[f"{method}_clusters_3d"] = cfig3d
                _save(cfig3d, method, f"{method}_clusters_kmeans_k{km_k}_3d")

    return figures


# ---------------------------------------------------------------------------
# unsupervised_cv.py
# ---------------------------------------------------------------------------
"""Unsupervised cross-validation and temporal robustness tests."""

from typing import Sequence, Tuple, Optional, Dict

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from scipy.spatial.distance import pdist

__all__ = ["unsupervised_cv_and_temporal_tests"]


def _find_date_column(df: pd.DataFrame) -> Optional[str]:
    """Return the first column name containing 'date' (case-insensitive)."""
    for col in df.columns:
        if "date" in col.lower():
            return col
    return None


def _fit_preprocess(
    df: pd.DataFrame, quant_vars: Sequence[str], qual_vars: Sequence[str]
) -> Tuple[np.ndarray, Optional[StandardScaler], Optional[OneHotEncoder]]:
    """Scale numeric columns and one-hot encode categoricals."""
    scaler: Optional[StandardScaler] = None
    encoder: Optional[OneHotEncoder] = None

    X_num = np.empty((len(df), 0))
    if quant_vars:
        scaler = StandardScaler()
        X_num = scaler.fit_transform(df[quant_vars])

    X_cat = np.empty((len(df), 0))
    if qual_vars:
        try:
            encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        except TypeError:  # pragma: no cover - older scikit-learn
            encoder = OneHotEncoder(handle_unknown="ignore")
        X_cat = encoder.fit_transform(df[qual_vars])

    if X_num.size and X_cat.size:
        X = np.hstack([X_num, X_cat])
    elif X_num.size:
        X = X_num
    else:
        X = X_cat

    return X, scaler, encoder


def _transform(
    df: pd.DataFrame,
    quant_vars: Sequence[str],
    qual_vars: Sequence[str],
    scaler: Optional[StandardScaler],
    encoder: Optional[OneHotEncoder],
) -> np.ndarray:
    """Apply preprocessing fitted on the training data."""
    X_num = np.empty((len(df), 0))
    if quant_vars and scaler is not None:
        X_num = scaler.transform(df[quant_vars])

    X_cat = np.empty((len(df), 0))
    if qual_vars and encoder is not None:
        X_cat = encoder.transform(df[qual_vars])

    if X_num.size and X_cat.size:
        return np.hstack([X_num, X_cat])
    if X_num.size:
        return X_num
    return X_cat


def _axis_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Return the mean absolute cosine similarity between corresponding axes."""
    sims = []
    for v1, v2 in zip(a, b):
        num = np.abs(np.dot(v1, v2))
        denom = np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-12
        sims.append(num / denom)
    return float(np.mean(sims)) if sims else float("nan")


def _distance_discrepancy(X1: np.ndarray, X2: np.ndarray) -> float:
    """Return the relative Frobenius norm between distance matrices."""
    d1 = pdist(X1)
    d2 = pdist(X2)
    norm = np.linalg.norm(d2) + 1e-12
    return float(np.linalg.norm(d1 - d2) / norm)


def unsupervised_cv_and_temporal_tests(
    df_active: pd.DataFrame,
    quant_vars: Sequence[str],
    qual_vars: Sequence[str],
    *,
    n_splits: int = 5,
) -> Dict[str, Dict[str, float]]:
    """Assess stability of PCA/UMAP with cross-validation and temporal splits."""

    logger = logging.getLogger(__name__)

    if not isinstance(df_active, pd.DataFrame):
        raise TypeError("df_active must be a DataFrame")

    kf: Optional[KFold]
    if n_splits < 2:
        logger.warning("n_splits < 2; skipping cross-validation")
        kf = None
    else:
        kf = KFold(n_splits=n_splits, shuffle=True)

    pca_axis_scores: list[float] = []
    pca_dist_scores: list[float] = []
    pca_var_ratio: list[float] = []
    umap_dist_scores: list[float] = []

    try:
        import umap  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        logger.warning("UMAP unavailable: %s", exc)
        umap = None  # type: ignore

    if kf is not None:
        def _process_split(train_idx: np.ndarray, test_idx: np.ndarray) -> tuple[float, float, float, float]:
            df_train = df_active.iloc[train_idx]
            df_test = df_active.iloc[test_idx]

            X_train, scaler, encoder = _fit_preprocess(df_train, quant_vars, qual_vars)
            X_test = _transform(df_test, quant_vars, qual_vars, scaler, encoder)

            n_comp = min(2, X_train.shape[1]) or 1
            pca_train = PCA(n_components=n_comp).fit(X_train)
            emb_proj = pca_train.transform(X_test)

            pca_test = PCA(n_components=n_comp)
            emb_test = pca_test.fit_transform(X_test)

            axis_score = _axis_similarity(pca_train.components_, pca_test.components_)
            dist_score = _distance_discrepancy(emb_proj, emb_test)
            var_ratio = float(pca_train.explained_variance_ratio_[0]) if pca_train.explained_variance_ratio_.size else float("nan")

            umap_score = float("nan")
            if umap is not None:
                reducer_train = umap.UMAP(n_components=2, n_jobs=-1)
                reducer_train.fit(X_train)
                emb_umap_proj = reducer_train.transform(X_test)
                reducer_test = umap.UMAP(n_components=2, n_jobs=-1)
                emb_umap_test = reducer_test.fit_transform(X_test)
                umap_score = _distance_discrepancy(emb_umap_proj, emb_umap_test)

            return axis_score, dist_score, var_ratio, umap_score

        results = Parallel(n_jobs=-1)(
            delayed(_process_split)(tr, te) for tr, te in kf.split(df_active)
        )
        for axis, dist, var, um in results:
            pca_axis_scores.append(axis)
            pca_dist_scores.append(dist)
            if not np.isnan(var):
                pca_var_ratio.append(var)
            if not np.isnan(um):
                umap_dist_scores.append(um)

    cv_stability = {
        "pca_axis_corr_mean": (
            float(np.nanmean(pca_axis_scores)) if pca_axis_scores else float("nan")
        ),
        "pca_axis_corr_std": (
            float(np.nanstd(pca_axis_scores)) if pca_axis_scores else float("nan")
        ),
        "pca_var_first_axis_mean": (
            float(np.nanmean(pca_var_ratio)) if pca_var_ratio else float("nan")
        ),
        "pca_var_first_axis_std": (
            float(np.nanstd(pca_var_ratio)) if pca_var_ratio else float("nan")
        ),
        "pca_distance_mse_mean": (
            float(np.mean(pca_dist_scores)) if pca_dist_scores else float("nan")
        ),
        "pca_distance_mse_std": (
            float(np.std(pca_dist_scores)) if pca_dist_scores else float("nan")
        ),
        "umap_distance_mse_mean": (
            float(np.mean(umap_dist_scores)) if umap_dist_scores else float("nan")
        ),
        "umap_distance_mse_std": (
            float(np.std(umap_dist_scores)) if umap_dist_scores else float("nan")
        ),
    }

    # Temporal robustness -------------------------------------------------
    date_col = _find_date_column(df_active)
    temporal_shift: Optional[Dict[str, float]] = None
    if date_col:
        df_sorted = df_active.sort_values(date_col)
        split_point = len(df_sorted) // 2
        df_old = df_sorted.iloc[:split_point]
        df_new = df_sorted.iloc[split_point:]

        X_old, scaler, encoder = _fit_preprocess(df_old, quant_vars, qual_vars)
        X_new = _transform(df_new, quant_vars, qual_vars, scaler, encoder)

        n_comp = min(2, X_old.shape[1]) or 1
        pca_old = PCA(n_components=n_comp).fit(X_old)
        emb_proj = pca_old.transform(X_new)

        pca_new = PCA(n_components=n_comp)
        emb_new = pca_new.fit_transform(X_new)

        axis_corr = _axis_similarity(pca_old.components_, pca_new.components_)
        dist_diff = _distance_discrepancy(emb_proj, emb_new)
        mean_shift = float(
            np.linalg.norm(
                emb_proj.mean(axis=0) - pca_old.transform(X_old).mean(axis=0)
            )
        )

        umap_dist = float("nan")
        if umap is not None:
            reducer_old = umap.UMAP(n_components=2, n_jobs=-1)
            reducer_old.fit(X_old)
            emb_proj_umap = reducer_old.transform(X_new)
            reducer_new = umap.UMAP(n_components=2, n_jobs=-1)
            emb_new_umap = reducer_new.fit_transform(X_new)
            umap_dist = _distance_discrepancy(emb_proj_umap, emb_new_umap)

        temporal_shift = {
            "pca_axis_corr": axis_corr,
            "pca_distance_mse": dist_diff,
            "pca_mean_shift": mean_shift,
            "umap_distance_mse": umap_dist,
        }

    return {"cv_stability": cv_stability, "temporal_shift": temporal_shift}


# ---------------------------------------------------------------------------
# pdf_report.py
# ---------------------------------------------------------------------------
"""Utilities to build a consolidated PDF report of phase 4 results."""


import datetime
import logging
import os
import tempfile
from contextlib import suppress
from pathlib import Path
from typing import Mapping, Union

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd


def _table_to_figure(df: pd.DataFrame, title: str) -> plt.Figure:
    """Return a Matplotlib figure displaying ``df`` as a table.

    Parameters
    ----------
    df : pandas.DataFrame
        Table to render.
    title : str
        Title of the figure.
    """
    # height grows with number of rows
    fig_height = 0.4 * len(df) + 1.5
    fig, ax = plt.subplots(figsize=(8.0, fig_height), dpi=200)
    ax.axis("off")
    ax.set_title(title)

    table = ax.table(
        cellText=df.values,
        colLabels=list(df.columns),
        rowLabels=list(df.index),
        cellLoc="center",
        rowLoc="center",
        loc="center",
    )
    table.scale(1, 1.2)
    fig.tight_layout()
    return fig


def format_metrics_table(df: pd.DataFrame) -> pd.DataFrame:
    """Return ``df`` with values formatted as strings for display."""
    formatted = df.copy()
    for col in formatted.columns:
        series = formatted[col]
        if col == "variance_cumulee_%":
            formatted[col] = series.map(
                lambda x: f"{int(round(x))}" if pd.notna(x) else ""
            )
        elif col == "nb_axes_kaiser":
            formatted[col] = series.map(
                lambda x: f"{int(x)}" if pd.notna(x) else ""
            )
        elif pd.api.types.is_integer_dtype(series):
            formatted[col] = series.map(
                lambda x: f"{int(x)}" if pd.notna(x) else ""
            )
        elif pd.api.types.is_float_dtype(series):
            formatted[col] = series.map(
                lambda x: f"{x:.2f}" if pd.notna(x) else ""
            )
        else:
            formatted[col] = series.astype(str).replace("nan", "")
    return formatted


def export_report_to_pdf(
    figures: Mapping[str, Union[plt.Figure, str, Path]],
    tables: Mapping[str, Union[pd.DataFrame, str, Path]],
    output_path: str | Path,
) -> Path:
    """Create a structured PDF gathering all figures and tables from phase 4.

    The function tries to use :mod:`fpdf` for advanced layout. If ``fpdf`` is not
    available, it falls back to :class:`matplotlib.backends.backend_pdf.PdfPages`
    (used in earlier versions).

    Parameters
    ----------
    figures : mapping
        Mapping from figure name to either a Matplotlib :class:`~matplotlib.figure.Figure`
        or a path to an existing image file.
    tables : mapping
        Mapping from table name to a :class:`pandas.DataFrame` or a CSV file path.
    output_path : str or :class:`pathlib.Path`
        Destination path of the PDF file.
        Pages are added in portrait mode by default but switch to landscape when
        a table has many columns.

    Returns
    -------
    pathlib.Path
        Path to the generated PDF.
    """

    if not isinstance(output_path, (str, Path)):
        raise TypeError("output_path must be a path-like object")

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Exporting PDF report to %s", out)

    # Ensure previous figures do not accumulate and trigger warnings
    plt.close("all")

    try:
        from fpdf import FPDF  # type: ignore

        pdf = FPDF(format="A4", unit="mm")
        pdf.set_auto_page_break(auto=True, margin=10)

        def _add_title(text: str, size: int = 14) -> None:
            pdf.set_font("Helvetica", "B", size)
            pdf.cell(0, 10, txt=text, ln=1, align="C")

        # Title page
        pdf.add_page()
        _add_title("Rapport d'analyse Phase 4 – Résultats Dimensionnels", 16)
        pdf.set_font("Helvetica", size=12)
        today = datetime.datetime.now().strftime("%Y-%m-%d")
        pdf.cell(0, 10, f"Généré le {today}", ln=1, align="C")

        # Tables first (comparatif des méthodes, etc.)
        for name, table in tables.items():
            if isinstance(table, (str, Path)):
                try:
                    table = pd.read_csv(table)
                except Exception:
                    continue
            if not isinstance(table, pd.DataFrame):
                continue
            orientation = "L" if len(table.columns) > 6 else "P"
            pdf.add_page(orientation=orientation)
            _add_title(name)
            pdf.set_font("Courier", size=8)
            table_str = table.to_string()
            for line in table_str.splitlines():
                pdf.cell(0, 4, line, ln=1)

        # Figures
        tmp_paths: list[str] = []
        for name, figure in figures.items():
            if figure is None:
                continue
            pdf.add_page()
            _add_title(name)
            if isinstance(figure, (str, Path)):
                img_path = str(figure)
            else:
                tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                figure.savefig(tmp.name, dpi=200, bbox_inches="tight")
                plt.close(figure)
                img_path = tmp.name
                tmp_paths.append(tmp.name)
            pdf.image(img_path, w=180)

        pdf.output(str(out))

        for p in tmp_paths:
            with suppress(OSError):
                os.remove(p)

        plt.close("all")

    except Exception:  # pragma: no cover - fallback when FPDF not installed
        logger.info("FPDF not available, falling back to PdfPages")

        with PdfPages(out) as pdf_backend:
            fig, ax = plt.subplots(figsize=(8.27, 11.69), dpi=200)
            today = datetime.datetime.now().strftime("%Y-%m-%d")
            ax.text(
                0.5,
                0.6,
                "Rapport des analyses – Phase 4",
                fontsize=20,
                ha="center",
                va="center",
            )
            ax.text(
                0.5, 0.4, f"Généré le {today}", fontsize=12, ha="center", va="center"
            )
            ax.axis("off")
            pdf_backend.savefig(fig, dpi=300)
            plt.close(fig)

            for name, figure in figures.items():
                if figure is None:
                    continue
                if isinstance(figure, (str, Path)):
                    img = plt.imread(figure)
                    f, ax = plt.subplots()
                    ax.imshow(img)
                    ax.axis("off")
                    f.suptitle(name, fontsize=12)
                    pdf_backend.savefig(f, dpi=300)
                    plt.close(f)
                    continue
                if isinstance(fig, (str, Path)):
                    img = plt.imread(fig)
                    f, ax = plt.subplots()
                    ax.imshow(img)
                    ax.axis("off")
                    f.suptitle(name, fontsize=12)
                    pdf_backend.savefig(f, dpi=300)
                    plt.close(f)
                    continue
                try:
                    figure.suptitle(name, fontsize=12)
                    pdf_backend.savefig(figure, dpi=300)
                finally:
                    plt.close(figure)

            for name, table in tables.items():
                if isinstance(table, (str, Path)):
                    try:
                        table = pd.read_csv(table)
                    except Exception:
                        continue
                if not isinstance(table, pd.DataFrame):
                    continue
                fig = _table_to_figure(table, name)
                pdf_backend.savefig(fig, dpi=300)
                plt.close(fig)

        plt.close("all")

    return out


# ---------------------------------------------------------------------------
# best_params.py
# ---------------------------------------------------------------------------
import csv
import json
from pathlib import Path
from typing import Any, Dict


def _parse_value(value: str) -> Any:
    v = value.strip()
    if v.lower() in {"", "null"}:
        return None
    if v.lower() == "true":
        return True
    if v.lower() == "false":
        return False
    if (v.startswith('"') and v.endswith('"')) or (
        v.startswith("'") and v.endswith("'")
    ):
        return v[1:-1]
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        pass
    try:
        return json.loads(v)
    except Exception:
        return v


def load_best_params(
    csv_path: Path | str = Path(__file__).with_name("best_params.csv"),
) -> Dict[str, Dict[str, Any]]:
    csv_path = Path(csv_path)
    params: Dict[str, Dict[str, Any]] = {}
    if not csv_path.exists():
        return params
    with open(csv_path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            method = row["method"].strip().upper()
            param = row["param"].strip()
            value = _parse_value(row["value"])
            params.setdefault(method, {})[param] = value
    return params


BEST_PARAMS = load_best_params()
BEST_PARAMS.pop("PCAMIX", None)
