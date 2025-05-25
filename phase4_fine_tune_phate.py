#!/usr/bin/env python3
"""Grid search fine-tuning for PHATE on cleaned CRM data."""

from __future__ import annotations

import argparse
import itertools
import time
from pathlib import Path

import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import trustworthiness
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder

import phate

from phase4v2 import select_variables, handle_missing_values, scatter_all_segments

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# ---------------------------------------------------------------------------
# Helpers


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine tune PHATE")
    parser.add_argument("--multi", required=True, help="Cleaned multivariate CSV")
    parser.add_argument("--univ", required=False, help="Cleaned univariate CSV")
    parser.add_argument("--output", required=True, help="Output directory")
    return parser.parse_args()


def prepare_dataset(path: Path) -> tuple[pd.DataFrame, list[str], list[str], pd.Series]:
    """Load CSV and return processed dataset with target column."""

    df = pd.read_csv(path)

    for col in df.select_dtypes(include="object"):
        df[col] = df[col].astype("category")

    df_active, quant_vars, qual_vars = select_variables(df)
    df_active = handle_missing_values(df_active, quant_vars, qual_vars)

    y = (
        df_active["Statut commercial"].astype("category")
        if "Statut commercial" in df_active.columns
        else pd.Series()
    )

    return df_active, quant_vars, qual_vars, y


def preprocess(df: pd.DataFrame, quant_vars: list[str], qual_vars: list[str]) -> np.ndarray:
    """Return scaled numeric and encoded categorical matrix."""

    X_num = StandardScaler().fit_transform(df[quant_vars]) if quant_vars else pd.DataFrame()
    try:
        enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    except TypeError:  # older scikit-learn
        enc = OneHotEncoder(handle_unknown="ignore")
    X_cat = enc.fit_transform(df[qual_vars]) if qual_vars else pd.DataFrame()

    if quant_vars and qual_vars:
        X = np.hstack([X_num, X_cat])
    elif quant_vars:
        X = X_num
    else:
        X = X_cat
    return X


# ---------------------------------------------------------------------------
# Main logic


def main() -> None:
    args = parse_args()
    input_file = Path(args.multi)
    out_dir = Path(args.output)

    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    df_active, quant_vars, qual_vars, y = prepare_dataset(input_file)
    X = preprocess(df_active, quant_vars, qual_vars)

    # Parameter grid
    param_grid = {
        "n_components": [2, 3],
        "knn": [5, 15, 30],
        "t": [5, 20],
        "decay": [10, 20],
    }

    results: list[dict[str, float | int]] = []
    generated_files: list[Path] = []

    combos = itertools.product(
        param_grid["n_components"],
        param_grid["knn"],
        param_grid["t"],
        param_grid["decay"],
    )

    for nc, nn, tt, dc in combos:
        start = time.time()
        model = phate.PHATE(
            n_components=nc,
            knn=nn,
            t=tt,
            decay=dc,
            n_jobs=-1,
            random_state=42,
        )
        emb = model.fit_transform(X)
        runtime = time.time() - start

        tw = trustworthiness(X, emb)
        ct = trustworthiness(emb, X)
        if not y.empty:
            acc = float(
                cross_val_score(KNeighborsClassifier(n_neighbors=5), emb, y, cv=5).mean()
            )
        else:
            acc = float("nan")

        results.append(
            {
                "n_components": nc,
                "knn": nn,
                "t": tt,
                "decay": dc,
                "trustworthiness": tw,
                "continuity": ct,
                "knn_accuracy": acc,
                "runtime_s": runtime,
            }
        )

    metrics_df = pd.DataFrame(results)
    metrics_path = out_dir / "phate_tuning_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    generated_files.append(metrics_path)

    # Select best two configs by trustworthiness
    top = metrics_df.sort_values("trustworthiness", ascending=False).head(2)

    for _, row in top.iterrows():
        nc = int(row["n_components"])
        nn = int(row["knn"])
        tt = int(row["t"])
        dc = int(row["decay"])
        model = phate.PHATE(
            n_components=nc,
            knn=nn,
            t=tt,
            decay=dc,
            n_jobs=-1,
            random_state=42,
        )
        embedding = model.fit_transform(X)

        coord_df = pd.DataFrame(
            embedding,
            index=df_active.index,
            columns=[f"PHATE{i+1}" for i in range(nc)],
        )
        coords_path = out_dir / f"phate_coords_{nc}D_knn{nn}_t{tt}_decay{dc}.csv"
        coord_df.to_csv(coords_path, index=True)
        generated_files.append(coords_path)

        model_path = out_dir / f"phate_model_{nc}D_knn{nn}_t{tt}_decay{dc}.joblib"
        joblib.dump(model, model_path)
        generated_files.append(model_path)

        if nc >= 2 and not y.empty:
            plt.figure(figsize=(12, 6), dpi=200)
            cats = y.astype("category")
            palette = sns.color_palette("tab10", len(cats.cat.categories))
            for cat, color in zip(cats.cat.categories, palette):
                mask = cats == cat
                plt.scatter(
                    coord_df.loc[mask, "PHATE1"],
                    coord_df.loc[mask, "PHATE2"],
                    s=10,
                    alpha=0.7,
                    color=color,
                    label=str(cat),
                )
            plt.xlabel("PHATE1")
            plt.ylabel("PHATE2")
            plt.title("PHATE - 2D")
            plt.legend(title="Statut commercial", bbox_to_anchor=(1.05, 1), loc="upper left")
            plt.tight_layout()
            fig_path = fig_dir / f"phate_{nc}D_knn{nn}_t{tt}.png"
            plt.savefig(fig_path)
            plt.close()
            generated_files.append(fig_path)

            scatter_all_segments(
                coord_df[["PHATE1", "PHATE2"]],
                df_active,
                fig_dir,
                f"phate_{nc}D_knn{nn}_t{tt}",
            )

    index_path = out_dir / "index_phate.txt"
    with open(index_path, "w", encoding="utf-8") as f:
        for p in generated_files:
            f.write(str(Path(p).resolve()) + "\n")


if __name__ == "__main__":
    main()
