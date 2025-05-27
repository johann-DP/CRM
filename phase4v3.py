#!/usr/bin/env python3
"""Self-contained Phase 4 analysis pipeline.

This script combines all helper functions needed to reproduce the analyses
performed in previous notebooks and scripts.  It is organised in logical
sections so each function can easily be tested independently.  A YAML or JSON
configuration file drives the execution making the pipeline fully
reproducible.  Random seeds are fixed for deterministic behaviour.
"""

# ---------------------------------------------------------------------------
# Imports & Configuration
# ---------------------------------------------------------------------------
from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
import datetime
import inspect
import time

import numpy as np
import pandas as pd
import yaml

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import trustworthiness
from sklearn.metrics import silhouette_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import prince
import umap

try:  # optional dependencies
    import phate  # type: ignore
except Exception:  # pragma: no cover - optional dependency may not be present
    phate = None

try:
    import pacmap  # type: ignore
except Exception:  # pragma: no cover - optional dependency may not be present
    pacmap = None

from best_params import BEST_PARAMS


# ---------------------------------------------------------------------------
# Data Loading & Preparation
# ---------------------------------------------------------------------------

def _read_dataset(path: Path) -> pd.DataFrame:
    """Read a CSV or Excel file into a ``DataFrame`` with basic type handling."""
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


def load_datasets(config: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
    """Load raw and cleaned datasets specified in ``config``."""
    logger = logging.getLogger(__name__)
    if "input_file" not in config:
        raise ValueError("'input_file' missing from config")
    datasets: Dict[str, pd.DataFrame] = {}
    raw_path = Path(config["input_file"])
    datasets["raw"] = _read_dataset(raw_path)
    logger.info(
        "Raw dataset loaded from %s [%d rows, %d cols]",
        raw_path,
        datasets["raw"].shape[0],
        datasets["raw"].shape[1],
    )

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
        df = _read_dataset(path)
        datasets[key] = _apply_mapping(df)
        logger.info(
            "Loaded %s dataset from %s [%d rows, %d cols]",
            key,
            path,
            df.shape[0],
            df.shape[1],
        )

    ref_cols = set(datasets["raw"].columns)
    for name, df in list(datasets.items()):
        extra = set(df.columns) - ref_cols
        if extra:
            logger.debug("%s has %d additional columns", name, len(extra))
    return datasets


# ---------------------------------------------------------------------------
# Data preparation helpers
# ---------------------------------------------------------------------------

def prepare_data(df: pd.DataFrame, exclude_lost: bool = True) -> pd.DataFrame:
    """Clean and standardise ``df`` for analysis."""
    logger = logging.getLogger(__name__)
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    df_clean = df.copy()
    date_cols = [c for c in df_clean.columns if "date" in c.lower()]
    min_date = pd.Timestamp("1990-01-01")
    max_date = pd.Timestamp("2050-12-31")
    for col in date_cols:
        df_clean[col] = pd.to_datetime(df_clean[col], errors="coerce")
        mask = df_clean[col].lt(min_date) | df_clean[col].gt(max_date)
        if mask.any():
            logger.warning("%d invalid dates replaced by NaT in '%s'", mask.sum(), col)
            df_clean.loc[mask, col] = pd.NaT
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
    if "Code" in df_clean.columns:
        before = len(df_clean)
        df_clean = df_clean.drop_duplicates(subset=["Code"])
        if len(df_clean) != before:
            logger.info("%d duplicated rows dropped", before - len(df_clean))
    if {"Date de début actualisée", "Date de fin réelle"} <= set(df_clean.columns):
        df_clean["duree_projet_jours"] = (
            df_clean["Date de fin réelle"] - df_clean["Date de début actualisée"]
        ).dt.days
    if {"Total recette réalisé", "Budget client estimé"} <= set(df_clean.columns):
        denom = df_clean["Budget client estimé"].replace(0, np.nan)
        df_clean["taux_realisation"] = df_clean["Total recette réalisé"] / denom
        df_clean["taux_realisation"] = df_clean["taux_realisation"].replace([np.inf, -np.inf], np.nan)
    if {"Total recette réalisé", "Charge prévisionnelle projet"} <= set(df_clean.columns):
        df_clean["marge_estimee"] = df_clean["Total recette réalisé"] - df_clean["Charge prévisionnelle projet"]
    impute_cols: list[str] = [c for c in amount_cols if c in df_clean.columns]
    if "taux_realisation" in df_clean.columns:
        impute_cols.append("taux_realisation")
    for col in impute_cols:
        median = df_clean[col].median()
        df_clean[col] = df_clean[col].fillna(median)
    for col in df_clean.select_dtypes(include="object"):
        df_clean[col] = df_clean[col].fillna("Non renseigné").astype("category")
    if "flag_multivariate" in df_clean.columns:
        out = df_clean["flag_multivariate"].astype(bool)
        if out.any():
            logger.info("%d outliers removed via 'flag_multivariate'", int(out.sum()))
            df_clean = df_clean.loc[~out]
    if exclude_lost and "Statut commercial" in df_clean.columns:
        lost_mask = df_clean["Statut commercial"].astype(str).str.contains(
            "perdu|annul|aband", case=False, na=False
        )
        if lost_mask.any():
            logger.info("%d lost opportunities removed", int(lost_mask.sum()))
            df_clean = df_clean.loc[~lost_mask]
    if exclude_lost and "Motif_non_conformité" in df_clean.columns:
        mask_nc = df_clean["Motif_non_conformité"].notna() & df_clean["Motif_non_conformité"].astype(str).str.strip().ne("")
        if mask_nc.any():
            logger.info("%d non conformities removed", int(mask_nc.sum()))
            df_clean = df_clean.loc[~mask_nc]
    num_cols = df_clean.select_dtypes(include=[np.number]).columns
    num_cols = [c for c in num_cols if c != "Code"]
    if len(num_cols) > 0:
        scaler = StandardScaler()
        df_clean[num_cols] = scaler.fit_transform(df_clean[num_cols])
    return df_clean


def select_variables(df: pd.DataFrame, min_modalite_freq: int = 5) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """Return dataframe restricted to relevant variables."""
    logger = logging.getLogger(__name__)
    candidate_quant = [
        "Total recette actualisé",
        "Total recette réalisé",
        "Total recette produit",
        "Budget client estimé",
        "duree_projet_jours",
        "taux_realisation",
        "marge_estimee",
    ]
    candidate_qual = [
        "Statut commercial",
        "Statut production",
        "Type opportunité",
        "Catégorie",
        "Sous-catégorie",
        "Pilier",
        "Entité opérationnelle",
        "Présence partenaire",
    ]
    exclude = {"Code", "Client", "Contact principal", "Titre"}
    quant_vars: List[str] = []
    for col in candidate_quant:
        if col not in df.columns or col in exclude:
            continue
        series = pd.to_numeric(df[col], errors="coerce")
        if series.var(skipna=True) == 0 or series.isna().all():
            logger.warning("Variable quantitative '%s' exclue", col)
            continue
        df[col] = series.astype(float)
        quant_vars.append(col)
    qual_vars: List[str] = []
    for col in candidate_qual:
        if col not in df.columns or col in exclude:
            continue
        series = df[col].astype("category")
        counts = series.value_counts(dropna=False)
        rares = counts[counts < min_modalite_freq].index
        if len(rares) > 0:
            logger.info("%d modalités rares dans '%s' regroupées en 'Autre'", len(rares), col)
            if "Autre" not in series.cat.categories:
                series = series.cat.add_categories(["Autre"])
            series = series.apply(lambda x: "Autre" if x in rares else x).astype("category")
        if series.nunique(dropna=False) <= 1:
            logger.warning("Variable qualitative '%s' exclue", col)
            continue
        df[col] = series
        qual_vars.append(col)
    df_active = df[quant_vars + qual_vars].copy()
    if quant_vars:
        scaler = StandardScaler()
        df_active[quant_vars] = scaler.fit_transform(df_active[quant_vars])
    for col in qual_vars:
        df_active[col] = df_active[col].astype("category")
    logger.info("DataFrame actif avec %d variables", len(df_active.columns))
    return df_active, quant_vars, qual_vars


def handle_missing_values(df: pd.DataFrame, quant_vars: List[str], qual_vars: List[str]) -> pd.DataFrame:
    """Impute and drop remaining NA values if needed."""
    logger = logging.getLogger(__name__)
    na_count = int(df.isna().sum().sum())
    if na_count > 0:
        logger.info("Imputation des %d valeurs manquantes restantes", na_count)
        if quant_vars:
            df[quant_vars] = df[quant_vars].fillna(df[quant_vars].median())
        for col in qual_vars:
            if df[col].dtype.name == "category" and "Non renseigné" not in df[col].cat.categories:
                df[col] = df[col].cat.add_categories("Non renseigné")
            df[col] = df[col].fillna("Non renseigné").astype("category")
        remaining = int(df.isna().sum().sum())
        if remaining > 0:
            logger.warning("%d NA subsistent après imputation → suppression des lignes concernées", remaining)
            df.dropna(inplace=True)
        for col in qual_vars:
            if df[col].dtype.name == "category":
                df[col] = df[col].cat.remove_unused_categories()
    else:
        logger.info("Aucune valeur manquante détectée après sanity_check")
    for col in qual_vars:
        if df[col].dtype.name == "category":
            df[col] = df[col].cat.remove_unused_categories()
    if df.isna().any().any():
        logger.error("Des NA demeurent dans df après traitement")
    else:
        logger.info("DataFrame sans NA prêt pour FAMD")
    return df


# ---------------------------------------------------------------------------
# Dimensionality Reduction Methods
# ---------------------------------------------------------------------------

def _get_explained_inertia(model: object) -> List[float]:
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
    random_state: Optional[int] = None,
    whiten: Optional[bool] = None,
    svd_solver: Optional[str] = None,
) -> Dict[str, object]:
    """Run a Principal Component Analysis on quantitative variables."""
    start = time.perf_counter()
    logger = logging.getLogger(__name__)

    X = StandardScaler().fit_transform(df_active[quant_vars])
    max_dim = min(X.shape)

    if optimize and n_components is None:
        tmp = PCA(n_components=max_dim, random_state=random_state).fit(X)
        n_components = _select_n_components(tmp.explained_variance_, threshold=variance_threshold)
        logger.info("PCA: selected %d components automatically", n_components)

    n_components = n_components or max_dim
    kwargs = {}
    if whiten is not None:
        kwargs["whiten"] = whiten
    if svd_solver is not None:
        kwargs["svd_solver"] = svd_solver
    pca = PCA(n_components=n_components, random_state=random_state, **kwargs)
    emb = pca.fit_transform(X)

    inertia = pd.Series(pca.explained_variance_ratio_, index=[f"F{i+1}" for i in range(pca.n_components_)])
    embeddings = pd.DataFrame(emb, index=df_active.index, columns=[f"F{i+1}" for i in range(pca.n_components_)])
    loadings = pd.DataFrame(pca.components_.T, index=quant_vars, columns=[f"F{i+1}" for i in range(pca.n_components_)])

    runtime = time.perf_counter() - start
    return {
        "model": pca,
        "inertia": inertia,
        "embeddings": embeddings,
        "loadings": loadings,
        "runtime_s": runtime,
    }


def run_mca(
    df_active: pd.DataFrame,
    qual_vars: List[str],
    n_components: Optional[int] = None,
    *,
    optimize: bool = False,
    variance_threshold: float = 0.8,
    random_state: Optional[int] = None,
    normalize: bool = True,
    n_iter: int = 3,
) -> Dict[str, object]:
    """Run Multiple Correspondence Analysis."""
    start = time.perf_counter()
    logger = logging.getLogger(__name__)
    df_cat = df_active[qual_vars].astype("category")

    if optimize and n_components is None:
        max_dim = sum(df_cat[c].nunique() - 1 for c in df_cat.columns)
        tmp = prince.MCA(n_components=max_dim, random_state=random_state).fit(df_cat)
        eig = getattr(tmp, "eigenvalues_", None)
        if eig is None:
            eig = np.asarray(_get_explained_inertia(tmp)) * max_dim
        n_components = _select_n_components(eig, threshold=variance_threshold)
        logger.info("MCA: selected %d components automatically", n_components)

    n_components = n_components or 5
    corr = "benzecri" if normalize else None
    mca = prince.MCA(
        n_components=n_components,
        n_iter=n_iter,
        correction=corr,
        random_state=random_state,
    )
    mca = mca.fit(df_cat)

    inertia = pd.Series(_get_explained_inertia(mca), index=[f"F{i+1}" for i in range(mca.n_components)])
    embeddings = mca.row_coordinates(df_cat)
    embeddings.index = df_active.index
    col_coords = mca.column_coordinates(df_cat)

    runtime = time.perf_counter() - start
    return {
        "model": mca,
        "inertia": inertia,
        "embeddings": embeddings,
        "column_coords": col_coords,
        "runtime_s": runtime,
    }


def run_famd(
    df_active: pd.DataFrame,
    quant_vars: List[str],
    qual_vars: List[str],
    n_components: Optional[int] = None,
    *,
    optimize: bool = False,
    variance_threshold: float = 0.8,
    random_state: Optional[int] = None,
) -> Dict[str, object]:
    """Run Factor Analysis of Mixed Data (FAMD)."""
    start = time.perf_counter()
    logger = logging.getLogger(__name__)

    scaler = StandardScaler()
    X_quanti = scaler.fit_transform(df_active[quant_vars])
    df_quanti = pd.DataFrame(X_quanti, index=df_active.index, columns=quant_vars)
    df_mix = pd.concat([df_quanti, df_active[qual_vars].astype("category")], axis=1)

    if df_mix.isnull().any().any():
        raise ValueError("Input contains NaN values")

    if optimize and n_components is None:
        max_dim = df_mix.shape[1]
        tmp = prince.FAMD(n_components=max_dim, n_iter=3, random_state=random_state).fit(df_mix)
        eig = getattr(tmp, "eigenvalues_", None)
        if eig is None:
            eig = np.asarray(_get_explained_inertia(tmp)) * max_dim
        n_components = _select_n_components(eig, threshold=variance_threshold)
        logger.info("FAMD: selected %d components automatically", n_components)

    n_components = n_components or df_mix.shape[1]
    famd = prince.FAMD(n_components=n_components, n_iter=3, random_state=random_state)
    famd = famd.fit(df_mix)

    inertia = pd.Series(_get_explained_inertia(famd), index=[f"F{i+1}" for i in range(famd.n_components)])
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
        contrib = (col_coords ** 2).div((col_coords ** 2).sum(axis=0), axis=1) * 100
    runtime = time.perf_counter() - start
    return {
        "model": famd,
        "inertia": inertia,
        "embeddings": embeddings,
        "column_coords": col_coords,
        "contributions": contrib,
        "runtime_s": runtime,
    }


def run_mfa(
    df_active: pd.DataFrame,
    groups: List[List[str]],
    n_components: Optional[int] = None,
    *,
    optimize: bool = False,
    variance_threshold: float = 0.8,
    normalize: bool = True,
    random_state: Optional[int] = None,
    n_iter: int = 3,
) -> Dict[str, object]:
    """Run Multiple Factor Analysis."""
    start = time.perf_counter()
    logger = logging.getLogger(__name__)
    qual_cols = []
    for group in groups:
        for col in group:
            if col in df_active.columns and (
                df_active[col].dtype.name == "category" or df_active[col].dtype == object
            ):
                qual_cols.append(col)
    seen = set()
    qual_cols = [c for c in qual_cols if not (c in seen or seen.add(c))]
    if qual_cols:
        try:
            enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        except TypeError:  # pragma: no cover - older scikit-learn
            enc = OneHotEncoder(sparse=False, handle_unknown="ignore")
        encoded = enc.fit_transform(df_active[qual_cols])
        df_dummies = pd.DataFrame(encoded, index=df_active.index, columns=enc.get_feature_names_out(qual_cols))
    else:
        df_dummies = pd.DataFrame(index=df_active.index)
    df_num = df_active.drop(columns=qual_cols)
    df_all = pd.concat([df_num, df_dummies], axis=1)

    groups_dict: Dict[str, List[str]] = {}
    used_cols: List[str] = []
    for i, g in enumerate(groups, 1):
        cols: List[str] = []
        for v in g:
            if v in df_all.columns:
                cols.append(v)
            else:
                cols.extend([c for c in df_all.columns if c.startswith(f"{v}_")])
        if cols:
            name = f"G{i}"
            groups_dict[name] = cols
            used_cols.extend(cols)
    remaining = [c for c in df_all.columns if c not in used_cols]
    if remaining:
        groups_dict[f"G{len(groups_dict)+1}"] = remaining
        used_cols.extend(remaining)
    df_all = df_all[used_cols]

    if normalize:
        scaler = StandardScaler()
        for cols in groups_dict.values():
            if cols:
                df_all[cols] = scaler.fit_transform(df_all[cols])

    if optimize and n_components is None:
        max_dim = df_all.shape[1]
        tmp = prince.MFA(n_components=max_dim, n_iter=n_iter, random_state=random_state)
        tmp = tmp.fit(df_all, groups=groups_dict)
        eig = getattr(tmp, "eigenvalues_", None)
        if eig is None:
            eig = (tmp.percentage_of_variance_ / 100) * max_dim
        n_components = _select_n_components(np.asarray(eig), threshold=variance_threshold)
        logger.info("MFA: selected %d components automatically", n_components)

    n_components = n_components or 5
    mfa = prince.MFA(n_components=n_components, n_iter=n_iter, random_state=random_state)
    mfa = mfa.fit(df_all, groups=groups_dict)
    mfa.explained_inertia_ = mfa.percentage_of_variance_ / 100
    embeddings = mfa.row_coordinates(df_all)
    embeddings.index = df_active.index

    inertia = pd.Series(mfa.explained_inertia_, index=[f"F{i+1}" for i in range(len(mfa.explained_inertia_))])
    runtime = time.perf_counter() - start
    return {
        "model": mfa,
        "inertia": inertia,
        "embeddings": embeddings,
        "runtime_s": runtime,
    }


def _encode_mixed(df: pd.DataFrame) -> np.ndarray:
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(exclude="number").columns.tolist()
    X_num = StandardScaler().fit_transform(df[numeric_cols]) if numeric_cols else np.empty((len(df), 0))
    if cat_cols:
        try:
            enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        except TypeError:  # pragma: no cover - older scikit-learn
            enc = OneHotEncoder(sparse=False, handle_unknown="ignore")
        X_cat = enc.fit_transform(df[cat_cols])
    else:
        X_cat = np.empty((len(df), 0))
    if X_num.size and X_cat.size:
        return np.hstack([X_num, X_cat])
    if X_num.size:
        return X_num
    return X_cat


def run_umap(
    df_active: pd.DataFrame,
    n_components: int = 2,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    *,
    metric: str = "euclidean",
    random_state: Optional[int] = None,
) -> Dict[str, Any]:
    """Run UMAP on ``df_active`` and return model and embeddings."""
    start = time.perf_counter()
    X = _encode_mixed(df_active)
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
    )
    embedding = reducer.fit_transform(X)
    runtime = time.perf_counter() - start
    cols = [f"UMAP{i + 1}" for i in range(n_components)]
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
    random_state: Optional[int] = None,
) -> Dict[str, Any]:
    """Run PHATE on ``df_active`` (returns empty result if not installed)."""
    if phate is None:
        logging.getLogger(__name__).warning("PHATE is not installed; skipping")
        return {"model": None, "embeddings": pd.DataFrame(index=df_active.index), "params": {}}
    start = time.perf_counter()
    X = _encode_mixed(df_active)
    op = phate.PHATE(
        n_components=n_components,
        knn=k,
        decay=a,
        t=t,
        n_jobs=-1,
        random_state=random_state,
        verbose=False,
    )
    embedding = op.fit_transform(X)
    runtime = time.perf_counter() - start
    cols = [f"PHATE{i + 1}" for i in range(n_components)]
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
    random_state: Optional[int] = None,
) -> Dict[str, Any]:
    """Run PaCMAP on ``df_active`` (skipped if not installed or failure)."""
    if pacmap is None:
        logging.getLogger(__name__).warning("PaCMAP is not installed; skipping")
        return {"model": None, "embeddings": pd.DataFrame(index=df_active.index), "params": {}}
    start = time.perf_counter()
    X = _encode_mixed(df_active)
    try:
        params = dict(
            n_components=n_components,
            n_neighbors=n_neighbors,
            MN_ratio=MN_ratio,
            FP_ratio=FP_ratio,
            num_iters=num_iters,
            random_state=random_state,
            verbose=False,
            apply_pca=True,
        )
        model = pacmap.PaCMAP(**params)
        embedding = model.fit_transform(X)
    except Exception as exc:  # pragma: no cover - rare runtime error
        logging.getLogger(__name__).warning("PaCMAP failed: %s", exc)
        return {"model": None, "embeddings": pd.DataFrame(index=df_active.index), "params": {}}
    runtime = time.perf_counter() - start
    cols = [f"PACMAP{i + 1}" for i in range(n_components)]
    emb_df = pd.DataFrame(embedding, index=df_active.index, columns=cols)
    params.pop("verbose")
    params.pop("apply_pca")
    return {
        "model": model,
        "embeddings": emb_df,
        "params": params,
        "runtime_s": runtime,
    }


# ---------------------------------------------------------------------------
# Visualization & Evaluation
# ---------------------------------------------------------------------------

def plot_correlation_circle(coords: pd.DataFrame, title: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12, 6), dpi=200)
    circle = plt.Circle((0, 0), 1, color="grey", fill=False, linestyle="dashed")
    ax.add_patch(circle)
    ax.axhline(0, color="grey", lw=0.5)
    ax.axvline(0, color="grey", lw=0.5)
    for var in coords.index:
        x, y = coords.loc[var, ["F1", "F2"]]
        ax.arrow(0, 0, x, y, head_width=0.02, length_includes_head=True, color="black")
        offset_x = x * 1.15 + 0.03 * np.sign(x)
        offset_y = y * 1.15 + 0.03 * np.sign(y)
        ax.text(offset_x, offset_y, str(var), fontsize=8, ha="center", va="center")
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_xlabel("F1")
    ax.set_ylabel("F2")
    ax.set_title(title)
    ax.set_aspect("equal")
    fig.tight_layout()
    return fig


def _choose_color_var(df: pd.DataFrame, qual_vars: List[str]) -> Optional[str]:
    preferred = ["Statut production", "Statut commercial", "Type opportunité"]
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
    fig, ax = plt.subplots(figsize=(12, 6), dpi=200)
    if color_var is None or color_var not in df_active.columns:
        ax.scatter(emb_df.iloc[:, 0], emb_df.iloc[:, 1], s=10, alpha=0.7)
    else:
        cats = df_active.loc[emb_df.index, color_var].astype("category")
        palette = sns.color_palette("tab10", len(cats.cat.categories))
        for cat, color in zip(cats.cat.categories, palette):
            mask = cats == cat
            ax.scatter(
                emb_df.loc[mask, emb_df.columns[0]],
                emb_df.loc[mask, emb_df.columns[1]],
                s=10,
                alpha=0.7,
                color=color,
                label=str(cat),
            )
        ax.legend(title=color_var, bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.set_xlabel(emb_df.columns[0])
    ax.set_ylabel(emb_df.columns[1])
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_scatter_3d(
    emb_df: pd.DataFrame, df_active: pd.DataFrame, color_var: Optional[str], title: str
) -> plt.Figure:
    fig = plt.figure(figsize=(12, 6), dpi=200)
    ax = fig.add_subplot(111, projection="3d")
    if color_var is None or color_var not in df_active.columns:
        ax.scatter(emb_df.iloc[:, 0], emb_df.iloc[:, 1], emb_df.iloc[:, 2], s=10, alpha=0.7)
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
                alpha=0.7,
                color=color,
                label=str(cat),
            )
        ax.legend(title=color_var, bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.set_xlabel(emb_df.columns[0])
    ax.set_ylabel(emb_df.columns[1])
    ax.set_zlabel(emb_df.columns[2])
    ax.set_title(title)
    fig.tight_layout()
    return fig


def _extract_quant_coords(coords: pd.DataFrame, quant_vars: List[str]) -> pd.DataFrame:
    cols = [c for c in ["F1", "F2"] if c in coords.columns]
    if len(cols) < 2:
        extra = [c for c in coords.columns if c not in cols][: 2 - len(cols)]
        cols.extend(extra)
    if len(cols) < 2:
        return pd.DataFrame(columns=["F1", "F2"])
    subset = coords.loc[[v for v in quant_vars if v in coords.index], cols]
    subset = subset.rename(columns={cols[0]: "F1", cols[1]: "F2"})
    return subset


def generate_figures(
    factor_results: Dict[str, Dict[str, Any]],
    nonlin_results: Dict[str, Dict[str, Any]],
    df_active: pd.DataFrame,
    quant_vars: List[str],
    qual_vars: List[str],
) -> Dict[str, plt.Figure]:
    color_var = _choose_color_var(df_active, qual_vars)
    figures: Dict[str, plt.Figure] = {}
    first_3d_done = False
    for method, res in factor_results.items():
        emb = res.get("embeddings")
        if isinstance(emb, pd.DataFrame) and emb.shape[1] >= 2:
            fig = plot_scatter_2d(
                emb.iloc[:, :2],
                df_active,
                color_var,
                f"Projection des affaires – {method.upper()}",
            )
            figures[f"{method}_scatter_2d"] = fig
            if not first_3d_done and emb.shape[1] >= 3:
                figures[f"{method}_scatter_3d"] = plot_scatter_3d(
                    emb.iloc[:, :3],
                    df_active,
                    color_var,
                    f"Projection 3D – {method.upper()}",
                )
                first_3d_done = True
        coords = res.get("loadings")
        if coords is None:
            coords = res.get("column_coords")
        if isinstance(coords, pd.DataFrame):
            qcoords = _extract_quant_coords(coords, quant_vars)
            if not qcoords.empty:
                var_pc = res.get("inertia")
                pct = float(var_pc.iloc[:2].sum() * 100) if isinstance(var_pc, pd.Series) else float("nan")
                title = f"{method.upper()} – cercle des corrélations (F1–F2)\nVariance {pct:.1f}%"
                figures[f"{method}_correlation"] = plot_correlation_circle(qcoords, title)
    for method, res in nonlin_results.items():
        emb = res.get("embeddings")
        if isinstance(emb, pd.DataFrame) and emb.shape[1] >= 2:
            fig = plot_scatter_2d(
                emb.iloc[:, :2],
                df_active,
                color_var,
                f"Projection des affaires – {method.upper()}",
            )
            figures[f"{method}_scatter_2d"] = fig
            if not first_3d_done and emb.shape[1] >= 3:
                figures[f"{method}_scatter_3d"] = plot_scatter_3d(
                    emb.iloc[:, :3],
                    df_active,
                    color_var,
                    f"Projection 3D – {method.upper()}",
                )
                first_3d_done = True
    return figures


def dunn_index(X: np.ndarray, labels: np.ndarray) -> float:
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
        for cj in unique[i + 1:]:
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
    n_clusters: int = 3,
) -> pd.DataFrame:
    rows = []
    n_features = len(quant_vars) + len(qual_vars)
    for method, info in results_dict.items():
        inertias = info.get("inertia")
        if inertias is None:
            inertias = []
        if isinstance(inertias, pd.Series):
            inertias = inertias.tolist()
        inertias = list(inertias)
        kaiser = int(sum(1 for eig in np.array(inertias) * n_features if eig > 1))
        cum_inertia = float(sum(inertias) * 100) if inertias else np.nan
        X_low = info["embeddings"].values
        labels = KMeans(n_clusters=n_clusters, random_state=None).fit_predict(X_low)
        sil = float(silhouette_score(X_low, labels))
        dunn = dunn_index(X_low, labels)
        try:
            enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        except TypeError:  # pragma: no cover - older scikit-learn
            enc = OneHotEncoder(sparse=False, handle_unknown="ignore")
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
        T = float(trustworthiness(X_high, X_low, n_neighbors=k_nn))
        C = float(trustworthiness(X_low, X_high, n_neighbors=k_nn))
        runtime = info.get("runtime_seconds") or info.get("runtime_s") or info.get("runtime")
        rows.append(
            {
                "method": method,
                "variance_cumulee_%": cum_inertia,
                "nb_axes_kaiser": kaiser,
                "silhouette": sil,
                "dunn_index": dunn,
                "trustworthiness": T,
                "continuity": C,
                "runtime_seconds": runtime,
            }
        )
    df_metrics = pd.DataFrame(rows).set_index("method")
    return df_metrics


def plot_methods_heatmap(df_metrics: pd.DataFrame, output_path: str | Path) -> None:
    output = Path(output_path)
    output.mkdir(parents=True, exist_ok=True)
    df_norm = df_metrics.copy()
    for col in df_norm.columns:
        cmin, cmax = df_norm[col].min(), df_norm[col].max()
        if pd.isna(cmin) or cmax == cmin:
            df_norm[col] = 0.0
        else:
            df_norm[col] = (df_norm[col] - cmin) / (cmax - cmin)
    plt.figure(figsize=(8, 0.4 * len(df_norm) + 2), dpi=200)
    ax = sns.heatmap(
        df_norm,
        annot=df_metrics,
        fmt=".2f",
        cmap="viridis",
        vmin=0,
        vmax=1,
    )
    ax.set_title("Comparaison des méthodes")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output / "methods_heatmap.png")
    plt.close()


# ---------------------------------------------------------------------------
# Comparison & Robustness
# ---------------------------------------------------------------------------

def compare_datasets_versions(
    datasets: Dict[str, pd.DataFrame], *, exclude_lost: bool = True, min_modalite_freq: int = 5
) -> Dict[str, Any]:
    if not isinstance(datasets, dict):
        raise TypeError("datasets must be a dictionary")
    results_by_version: Dict[str, Any] = {}
    metrics_frames: List[pd.DataFrame] = []
    for name, df in datasets.items():
        logging.info("Processing dataset version '%s'", name)
        df_prep = prepare_data(df, exclude_lost=exclude_lost)
        df_active, quant_vars, qual_vars = select_variables(df_prep, min_modalite_freq=min_modalite_freq)
        df_active = handle_missing_values(df_active, quant_vars, qual_vars)
        factor_results: Dict[str, Any] = {}
        if quant_vars:
            factor_results["pca"] = run_pca(df_active, quant_vars, optimize=True)
        if qual_vars:
            factor_results["mca"] = run_mca(df_active, qual_vars, optimize=True)
        if quant_vars and qual_vars:
            try:
                factor_results["famd"] = run_famd(df_active, quant_vars, qual_vars, optimize=True)
            except ValueError as exc:
                logging.getLogger(__name__).warning("FAMD skipped: %s", exc)
        groups = []
        if quant_vars:
            groups.append(quant_vars)
        if qual_vars:
            groups.append(qual_vars)
        if len(groups) > 1:
            factor_results["mfa"] = run_mfa(df_active, groups, optimize=True)
        nonlin_results = {
            "umap": run_umap(df_active),
            "phate": run_phate(df_active),
            "pacmap": run_pacmap(df_active),
        }
        cleaned_nonlin = {
            k: v
            for k, v in nonlin_results.items()
            if "embeddings" in v and isinstance(v["embeddings"], pd.DataFrame) and not v["embeddings"].empty
        }
        all_results = {**factor_results, **cleaned_nonlin}
        n_clusters = 3 if len(df_active) > 3 else 2
        metrics = evaluate_methods(all_results, df_active, quant_vars, qual_vars, n_clusters=n_clusters)
        metrics["dataset_version"] = name
        try:
            figures = generate_figures(factor_results, nonlin_results, df_active, quant_vars, qual_vars)
        except Exception as exc:  # pragma: no cover - visualization failure
            logging.getLogger(__name__).warning("Figure generation failed: %s", exc)
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
    combined = pd.concat(metrics_frames).reset_index().rename(columns={"index": "method"})
    return {"metrics": combined, "details": results_by_version}


def _find_date_column(df: pd.DataFrame) -> Optional[str]:
    for col in df.columns:
        if "date" in col.lower():
            return col
    return None


def _fit_preprocess(
    df: pd.DataFrame, quant_vars: Sequence[str], qual_vars: Sequence[str]
) -> Tuple[np.ndarray, Optional[StandardScaler], Optional[OneHotEncoder]]:
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
            encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
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
    sims = []
    for v1, v2 in zip(a, b):
        num = np.abs(np.dot(v1, v2))
        denom = np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-12
        sims.append(num / denom)
    return float(np.mean(sims)) if sims else float("nan")


def _distance_discrepancy(X1: np.ndarray, X2: np.ndarray) -> float:
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
    random_state: Optional[int] = None,
) -> pd.DataFrame:
    logger = logging.getLogger(__name__)
    if not isinstance(df_active, pd.DataFrame):
        raise TypeError("df_active must be a DataFrame")
    if n_splits < 2:
        raise ValueError("n_splits must be >= 2")
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    pca_axis_scores = []
    pca_dist_scores = []
    umap_dist_scores = []
    for train_idx, test_idx in kf.split(df_active):
        df_train = df_active.iloc[train_idx]
        df_test = df_active.iloc[test_idx]
        X_train, scaler, encoder = _fit_preprocess(df_train, quant_vars, qual_vars)
        X_test = _transform(df_test, quant_vars, qual_vars, scaler, encoder)
        n_comp = min(2, X_train.shape[1]) or 1
        pca_train = PCA(n_components=n_comp, random_state=random_state)
        pca_train.fit(X_train)
        emb_test_proj = pca_train.transform(X_test)
        pca_test = PCA(n_components=n_comp, random_state=random_state)
        emb_test = pca_test.fit_transform(X_test)
        pca_axis_scores.append(_axis_similarity(pca_train.components_, pca_test.components_))
        pca_dist_scores.append(_distance_discrepancy(emb_test_proj, emb_test))
        try:
            reducer_train = umap.UMAP(n_components=2, random_state=random_state)
            reducer_train.fit(X_train)
            emb_umap_proj = reducer_train.transform(X_test)
            reducer_test = umap.UMAP(n_components=2, random_state=random_state)
            emb_umap_test = reducer_test.fit_transform(X_test)
            umap_dist_scores.append(_distance_discrepancy(emb_umap_proj, emb_umap_test))
        except Exception as exc:  # pragma: no cover - optional dependency
            logger.warning("UMAP unavailable: %s", exc)
    pca_axis_cv = float(np.nanmean(pca_axis_scores)) if pca_axis_scores else float("nan")
    pca_dist_cv = float(np.mean(pca_dist_scores)) if pca_dist_scores else float("nan")
    umap_dist_cv = float(np.mean(umap_dist_scores)) if umap_dist_scores else float("nan")
    date_col = _find_date_column(df_active)
    pca_axis_temp = float("nan")
    pca_dist_temp = float("nan")
    umap_dist_temp = float("nan")
    if date_col:
        df_sorted = df_active.sort_values(date_col)
        split_point = len(df_sorted) // 2
        df_old = df_sorted.iloc[:split_point]
        df_new = df_sorted.iloc[split_point:]
        X_old, scaler, encoder = _fit_preprocess(df_old, quant_vars, qual_vars)
        X_new = _transform(df_new, quant_vars, qual_vars, scaler, encoder)
        n_comp = min(2, X_old.shape[1]) or 1
        pca_old = PCA(n_components=n_comp, random_state=random_state)
        pca_old.fit(X_old)
        emb_new_proj = pca_old.transform(X_new)
        pca_new = PCA(n_components=n_comp, random_state=random_state)
        emb_new = pca_new.fit_transform(X_new)
        pca_axis_temp = _axis_similarity(pca_old.components_, pca_new.components_)
        pca_dist_temp = _distance_discrepancy(emb_new_proj, emb_new)
        reducer_old = umap.UMAP(n_components=2, random_state=random_state)
        reducer_old.fit(X_old)
        emb_proj = reducer_old.transform(X_new)
        reducer_new = umap.UMAP(n_components=2, random_state=random_state)
        emb_new_umap = reducer_new.fit_transform(X_new)
        umap_dist_temp = _distance_discrepancy(emb_proj, emb_new_umap)
    rows = [
        {
            "method": "pca",
            "cv_axis_similarity": pca_axis_cv,
            "cv_distance_diff": pca_dist_cv,
            "temporal_axis_similarity": pca_axis_temp,
            "temporal_distance_diff": pca_dist_temp,
        },
        {
            "method": "umap",
            "cv_axis_similarity": float("nan"),
            "cv_distance_diff": umap_dist_cv,
            "temporal_axis_similarity": float("nan"),
            "temporal_distance_diff": umap_dist_temp,
        },
    ]
    return pd.DataFrame(rows).set_index("method")


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _table_to_figure(df: pd.DataFrame, title: str) -> plt.Figure:
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


def export_report_to_pdf(
    figures: Mapping[str, plt.Figure],
    tables: Mapping[str, pd.DataFrame],
    output_path: str | Path,
) -> Path:
    if not isinstance(output_path, (str, Path)):
        raise TypeError("output_path must be a path-like object")
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    logging.info("Exporting PDF report to %s", out)
    with PdfPages(out) as pdf:
        fig, ax = plt.subplots(figsize=(8.27, 11.69), dpi=200)
        today = datetime.datetime.now().strftime("%Y-%m-%d")
        ax.text(0.5, 0.6, "Rapport des analyses – Phase 4", fontsize=20, ha="center", va="center")
        ax.text(0.5, 0.4, f"Généré le {today}", fontsize=12, ha="center", va="center")
        ax.axis("off")
        pdf.savefig(fig, dpi=300)
        plt.close(fig)
        for name, fig in figures.items():
            if fig is None:
                continue
            try:
                fig.suptitle(name, fontsize=12)
                pdf.savefig(fig, dpi=300)
            finally:
                plt.close(fig)
        for name, table in tables.items():
            if not isinstance(table, pd.DataFrame):
                continue
            fig = _table_to_figure(table, name)
            pdf.savefig(fig, dpi=300)
            plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Pipeline Orchestration
# ---------------------------------------------------------------------------

def _params_for(method: str) -> Dict[str, Any]:
    return BEST_PARAMS.get(method.upper(), {}).copy()


def _filter_kwargs(func: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    sig = inspect.signature(func)
    return {k: v for k, v in params.items() if k in sig.parameters}


def _method_params(method: str, config: Mapping[str, Any]) -> Dict[str, Any]:
    params = _params_for(method)
    if method.lower() in config and isinstance(config[method.lower()], Mapping):
        params.update(config[method.lower()])
    prefix = f"{method.lower()}_"
    for key, value in config.items():
        if key.startswith(prefix):
            params[key[len(prefix) :]] = value
    return params


def _setup_logging(output_dir: Path, level: str = "INFO") -> logging.Logger:
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


def run_pipeline(config: Dict[str, Any]) -> Dict[str, Any]:
    rs = config.get("random_state")
    random_state = int(rs) if rs is not None else None
    output_dir = Path(config.get("output_dir", "phase4_output"))
    _setup_logging(output_dir)
    datasets = load_datasets(config)
    data_key = config.get("main_dataset", config.get("dataset", "raw"))
    if data_key not in datasets:
        raise KeyError(f"dataset '{data_key}' not found in config")
    logging.info("Running pipeline on dataset '%s'", data_key)
    df_prep = prepare_data(datasets[data_key], exclude_lost=bool(config.get("exclude_lost", True)))
    df_active, quant_vars, qual_vars = select_variables(
        df_prep, min_modalite_freq=int(config.get("min_modalite_freq", 5))
    )
    df_active = handle_missing_values(df_active, quant_vars, qual_vars)
    methods_cfg = [m.lower() for m in config.get(
        "methods",
        ["pca", "mca", "famd", "mfa", "umap", "phate", "pacmap"],
    )]
    if not quant_vars and not qual_vars:
        logging.getLogger(__name__).warning("No variables selected; skipping analysis")
        empty = pd.DataFrame()
        empty.to_csv(output_dir / "metrics.csv")
        return {"metrics": empty, "figures": {}}
    factor_results: Dict[str, Any] = {}
    if "pca" in methods_cfg and quant_vars:
        params = _filter_kwargs(run_pca, _method_params("pca", config))
        if "n_components" in params:
            params["n_components"] = min(params["n_components"], len(quant_vars), len(df_active))
        factor_results["pca"] = run_pca(
            df_active,
            quant_vars,
            optimize=True,
            random_state=random_state,
            **params,
        )
    if "mca" in methods_cfg and qual_vars:
        params = _filter_kwargs(run_mca, _method_params("mca", config))
        if "n_components" in params:
            params["n_components"] = min(params["n_components"], sum(df_active[q].nunique() - 1 for q in qual_vars))
        factor_results["mca"] = run_mca(
            df_active,
            qual_vars,
            optimize=True,
            random_state=random_state,
            **params,
        )
    if "famd" in methods_cfg and quant_vars and qual_vars:
        try:
            params = _filter_kwargs(run_famd, _method_params("famd", config))
            if "n_components" in params:
                params["n_components"] = min(params["n_components"], df_active.shape[1])
            factor_results["famd"] = run_famd(
                df_active,
                quant_vars,
                qual_vars,
                optimize=True,
                random_state=random_state,
                **params,
            )
        except ValueError as exc:
            logging.getLogger(__name__).warning("FAMD skipped: %s", exc)
    groups = []
    if quant_vars:
        groups.append(quant_vars)
    if qual_vars:
        groups.append(qual_vars)
    if "mfa" in methods_cfg and len(groups) > 1:
        params = _filter_kwargs(run_mfa, _method_params("mfa", config))
        if "n_components" in params:
            params["n_components"] = min(params["n_components"], df_active.shape[1])
        factor_results["mfa"] = run_mfa(
            df_active,
            groups,
            optimize=True,
            random_state=random_state,
            **params,
        )
    nonlin_results: Dict[str, Any] = {}
    if "umap" in methods_cfg:
        params = _filter_kwargs(run_umap, _method_params("umap", config))
        nonlin_results["umap"] = run_umap(df_active, random_state=random_state, **params)
    if "phate" in methods_cfg:
        params = _filter_kwargs(run_phate, _method_params("phate", config))
        nonlin_results["phate"] = run_phate(df_active, random_state=random_state, **params)
    if "pacmap" in methods_cfg:
        params = _filter_kwargs(run_pacmap, _method_params("pacmap", config))
        nonlin_results["pacmap"] = run_pacmap(df_active, random_state=random_state, **params)
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
    metrics = evaluate_methods({**factor_results, **valid_nonlin}, 
                               df_active, quant_vars, 
                               qual_vars, 
                               n_clusters=3 if len(df_active) > 3 else 2)
    plot_methods_heatmap(metrics, output_dir)
    metrics.to_csv(output_dir / "metrics.csv")
    figures = generate_figures(factor_results, 
                               nonlin_results, 
                               df_active, quant_vars, 
                               qual_vars)
    for name, fig in figures.items():
        fig.savefig(output_dir / f"{name}.png")
        plt.close(fig)
    comparison_metrics = None
    comparison_figures: Dict[str, plt.Figure] = {}
    if config.get("compare_versions"):
        compare_res = compare_datasets_versions(
            {k: v for k, v in datasets.items() if k != "raw"},
            exclude_lost=bool(config.get("exclude_lost", True)),
            min_modalite_freq=int(config.get("min_modalite_freq", 5)),
        )
        comparison_metrics = compare_res["metrics"]
        comparison_figures = {
            f"{ver}_{name}": fig
            for ver, det in compare_res["details"].items()
            for name, fig in det["figures"].items()
        }
        comparison_metrics.to_csv(output_dir / "comparison_metrics.csv", index=False)
    robustness_df = None
    if config.get("run_temporal_tests"):
        robustness_df = unsupervised_cv_and_temporal_tests(
            df_active,
            quant_vars,
            qual_vars,
            n_splits=int(config.get("n_splits", 5)),
            random_state=random_state,
        )
        robustness_df.to_csv(output_dir / "robustness.csv")
    if config.get("output_pdf"):
        all_figs = {**figures, **comparison_figures}
        tables: Dict[str, pd.DataFrame] = {"metrics": metrics}
        if comparison_metrics is not None:
            tables["comparison_metrics"] = comparison_metrics
        if robustness_df is not None:
            tables["robustness"] = robustness_df
        export_report_to_pdf(all_figs, tables, config["output_pdf"])
    logging.info("Analysis complete")
    return {
        "metrics": metrics,
        "figures": figures,
        "comparison_metrics": comparison_metrics,
        "robustness": robustness_df,
    }


def _load_config(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        if path.suffix.lower() in {".yaml", ".yml"}:
            return yaml.safe_load(fh)
        return json.load(fh)


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Phase 4 analysis pipeline")
    parser.add_argument("--config", required=True, help="Path to config YAML/JSON")
    args = parser.parse_args(argv)
    np.random.seed(0)
    random.seed(0)
    cfg = _load_config(Path(args.config))
    CONFIG.update(cfg)
    run_pipeline(cfg)


if __name__ == "__main__":  # pragma: no cover - manual execution
    main()
