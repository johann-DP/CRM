#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Export additional Phase 4 results as CSV files.

This script runs a lightweight version of the Phase 4 pipeline and writes
CSV tables summarising the projections, cluster assignments and distances
for the main dataset configured in ``config.yaml``.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml
from scipy.spatial.distance import pdist, squareform

import phase4
from phase4_functions import (
    load_datasets,
    prepare_data,
    select_variables,
    handle_missing_values,
    run_pca,
    run_mca,
    run_famd,
    run_mfa,
    run_umap,
    run_phate,
    run_pacmap,
    evaluate_methods,
)

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def load_config(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        if path.suffix.lower() in {".yaml", ".yml"}:
            return yaml.safe_load(fh)
        return json.load(fh)


def method_params(method: str, config: Mapping[str, Any]) -> Dict[str, Any]:
    """Return parameters for ``method`` from ``config``."""
    return phase4._method_params(method, config)  # type: ignore[attr-defined]


def encode_data(df: pd.DataFrame, quant_vars: Sequence[str], qual_vars: Sequence[str]) -> np.ndarray:
    """Return numeric matrix with one-hot encoded categoricals."""
    parts = []
    if quant_vars:
        parts.append(df[quant_vars].to_numpy(float))
    if qual_vars:
        dummies = pd.get_dummies(df[qual_vars], dummy_na=False)
        parts.append(dummies.to_numpy(float))
    if parts:
        return np.hstack(parts)
    return np.empty((len(df), 0))


def clean_thumbs(path: Path) -> None:
    """Remove any ``Thumbs.db`` files under ``path``."""
    for thumb in path.rglob("Thumbs.db"):
        try:
            thumb.unlink()
        except Exception:
            pass


def intra_inter_distances(X: np.ndarray, labels: Sequence[int]) -> tuple[float, float]:
    """Return mean intra and inter cluster distances for ``labels``."""
    if X.size == 0:
        return float("nan"), float("nan")
    dist = squareform(pdist(X))
    uniq = np.unique(labels)
    intra: list[float] = []
    for lab in uniq:
        idx = np.where(np.asarray(labels) == lab)[0]
        if len(idx) > 1:
            intra.append(dist[np.ix_(idx, idx)].mean())
        else:
            intra.append(0.0)
    centroids = [X[labels == lab].mean(axis=0) for lab in uniq]
    inter = pdist(np.vstack(centroids)) if len(centroids) > 1 else np.array([np.nan])
    return float(np.nanmean(intra)), float(np.nanmean(inter))


# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------

def run(config: Mapping[str, Any]) -> None:
    output_dir = Path(config.get("output_dir", "phase4_output"))
    output_dir.mkdir(parents=True, exist_ok=True)
    clean_thumbs(output_dir)
    phase4.set_blas_threads(int(config.get("n_jobs", -1)))

    logging.info("Loading datasets (output: %s)...", output_dir)

    datasets = load_datasets(config, ignore_schema=bool(config.get("ignore_schema", False)))
    name = config.get("dataset", config.get("main_dataset", "raw"))
    if name not in datasets:
        raise KeyError(f"dataset '{name}' not found")
    logging.info("Dataset %s loaded", name)

    df_prep = prepare_data(datasets[name], exclude_lost=bool(config.get("exclude_lost", True)))
    df_active, quant_vars, qual_vars = select_variables(
        df_prep, min_modalite_freq=int(config.get("min_modalite_freq", 5))
    )
    df_active = handle_missing_values(df_active, quant_vars, qual_vars)

    methods = [m.lower() for m in config.get(
        "methods",
        ["pca", "mca", "famd", "mfa", "umap", "phate", "pacmap"],
    )]

    factor_results: Dict[str, Any] = {}
    nonlin_results: Dict[str, Any] = {}

    logging.info("Running factor methods...")

    if "pca" in methods and quant_vars:
        factor_results["pca"] = run_pca(
            df_active, quant_vars, optimize=False, **method_params("pca", config)
        )
    if "mca" in methods and qual_vars:
        factor_results["mca"] = run_mca(
            df_active, qual_vars, optimize=False, **method_params("mca", config)
        )
    if "famd" in methods and quant_vars and qual_vars:
        factor_results["famd"] = run_famd(
            df_active, quant_vars, qual_vars, optimize=False, **method_params("famd", config)
        )
    if "mfa" in methods and (quant_vars or qual_vars):
        groups = []
        if quant_vars:
            groups.append(quant_vars)
        if qual_vars:
            groups.append(qual_vars)
        params = method_params("mfa", config)
        grp = params.pop("groups", None)
        if grp:
            groups = grp
        factor_results["mfa"] = run_mfa(df_active, groups, optimize=False, **params)

    logging.info("Running non-linear methods...")
    if "umap" in methods:
        nonlin_results["umap"] = run_umap(df_active, **method_params("umap", config))
    if "phate" in methods:
        nonlin_results["phate"] = run_phate(df_active, **method_params("phate", config))
    if "pacmap" in methods:
        nonlin_results["pacmap"] = run_pacmap(df_active, **method_params("pacmap", config))

    all_results = {**factor_results, **nonlin_results}
    k_max = min(10, max(2, len(df_active) - 1))
    metrics = evaluate_methods(
        all_results,
        df_active,
        quant_vars,
        qual_vars,
        k_range=range(2, k_max + 1),
    )
    logging.info("Clustering and metric evaluation done")

    # Coordinates and cluster labels ---------------------------------------
    coord_df = pd.DataFrame(index=df_active.index)
    labels_df = pd.DataFrame(index=df_active.index)
    for method, info in all_results.items():
        emb = info.get("embeddings")
        if isinstance(emb, pd.DataFrame) and not emb.empty:
            coord_df[f"{method}_1"] = emb.iloc[:, 0]
            if emb.shape[1] > 1:
                coord_df[f"{method}_2"] = emb.iloc[:, 1]
            labels = info.get("cluster_labels")
            if labels is not None:
                labels_df[f"{method}"] = labels

    coord_df.to_csv(output_dir / "coordinates.csv")
    labels_df.to_csv(output_dir / "cluster_labels.csv")
    logging.info("Saved coordinates and cluster labels")

    segmented = df_prep.loc[coord_df.index].copy()
    for col in labels_df.columns:
        segmented[f"cluster_{col}"] = labels_df[col]
    segmented.to_csv(output_dir / "dataset_segmented.csv", index=False)
    logging.info("Saved segmented dataset")

    # Cluster statistics ----------------------------------------------------
    for method, labels in labels_df.items():
        tmp = segmented.copy()
        tmp["cluster"] = labels
        stats = tmp.groupby("cluster")[quant_vars].agg(["mean", "std"])
        stats.to_csv(output_dir / f"{method}_cluster_stats.csv")
        logging.info("Saved statistics for %s", method)
  
    # Distances and compacity/separation indices ---------------------------
    X_high = encode_data(df_active, quant_vars, qual_vars)
    records = []
    for method, info in all_results.items():
        labels = info.get("cluster_labels")
        emb = info.get("embeddings")
        if labels is None or not isinstance(emb, pd.DataFrame):
            continue
        intra_h, inter_h = intra_inter_distances(X_high, labels)
        intra_l, inter_l = intra_inter_distances(emb.iloc[:, :2].to_numpy(), labels)
        rec = {
            "method": method,
            "intra_orig": intra_h,
            "inter_orig": inter_h,
            "intra_low": intra_l,
            "inter_low": inter_l,
        }
        rec.update(metrics.loc[method, [
            "silhouette",
            "dunn_index",
            "calinski_harabasz",
            "inv_davies_bouldin",
        ]].to_dict())
        records.append(rec)
    dist_df = pd.DataFrame(records).set_index("method")
    dist_df.to_csv(output_dir / "cluster_distance_metrics.csv")
    logging.info("Saved cluster distance metrics")

    # Method parameters -----------------------------------------------------
    params = {m: method_params(m, config) for m in methods}
    with open(output_dir / "method_params.json", "w", encoding="utf-8") as fh:
        json.dump(params, fh, ensure_ascii=False, indent=2)
    logging.info("Saved method parameters")
    logging.info("CSV export completed")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export Phase 4 CSV tables")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    parser.add_argument("--dataset", help="Dataset to process (override config)")
    args = parser.parse_args()

    logging.basicConfig(format="%(message)s", level=logging.INFO)

    cfg = load_config(Path(args.config))
    if args.dataset:
        cfg["dataset"] = args.dataset
    run(cfg)
