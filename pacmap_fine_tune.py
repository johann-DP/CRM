#!/usr/bin/env python3
"""Grid search fine-tuning for PaCMAP on cleaned CRM data."""

from __future__ import annotations

import argparse
import itertools
import inspect
import time
from pathlib import Path

import joblib
import pandas as pd
from sklearn.manifold import trustworthiness
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import pacmap

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


try:
    pacmap.PaCMAP(init="pca")
    _PACMAP_HAS_INIT = True
except TypeError:  # pragma: no cover - older pacmap
    _PACMAP_HAS_INIT = False

# Import helper functions for variable selection and missing value handling
from phase4v2 import select_variables, handle_missing_values



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine tune PaCMAP")
    parser.add_argument("--input", required=True, help="Cleaned multivariate CSV")
    parser.add_argument("--output", required=True, help="Output directory")
    return parser.parse_args()


def prepare_dataset(path: Path) -> tuple[pd.DataFrame, list[str], list[str], pd.Series]:
    """Load CSV and return processed dataset with target column."""

    df = pd.read_csv(path)

    # convert object columns to categories
    for col in df.select_dtypes(include="object"):
        df[col] = df[col].astype("category")

    df_active, quant_vars, qual_vars = select_variables(df)
    df_active = handle_missing_values(df_active, quant_vars, qual_vars)

    y = df_active["Statut commercial"].astype("category") if "Statut commercial" in df_active.columns else pd.Series()

    return df_active, quant_vars, qual_vars, y


def preprocess(df: pd.DataFrame, quant_vars: list[str], qual_vars: list[str]):
    """Return scaled numeric and encoded categorical matrix."""

    X_num = StandardScaler().fit_transform(df[quant_vars]) if quant_vars else pd.DataFrame()
    try:
        enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    except TypeError:  # older scikit-learn
        enc = OneHotEncoder(sparse=False, handle_unknown="ignore")
    X_cat = enc.fit_transform(df[qual_vars]) if qual_vars else pd.DataFrame()

    import numpy as np

    if quant_vars and qual_vars:
        X = np.hstack([X_num, X_cat])
    elif quant_vars:
        X = X_num
    else:
        X = X_cat
    return X


def main() -> None:
    args = parse_args()
    input_file = Path(args.input)
    out_dir = Path(args.output)

    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    df_active, quant_vars, qual_vars, y = prepare_dataset(input_file)
    X = preprocess(df_active, quant_vars, qual_vars)

    results = []
    generated_files = []

    supports_init = "init" in inspect.signature(pacmap.PaCMAP).parameters
    param_grid = {
        "n_neighbors": [5, 15, 30, 50],
        "MN_ratio": [0.5, 1.0, 2.0],
        "n_components": [2, 3],
    }
    has_init = "init" in pacmap.PaCMAP.__init__.__code__.co_varnames
    if has_init:
        param_grid["init"] = ["pca", "random"]

    iter_args = [param_grid["n_neighbors"], param_grid["MN_ratio"], param_grid["n_components"]]
    if has_init:
        iter_args.append(param_grid["init"])
    for combo in itertools.product(*iter_args):
        if has_init:
            nn, mn, nc, ini = combo
        else:
            nn, mn, nc = combo
            ini = None
        start = time.time()
        kwargs = dict(n_components=nc, n_neighbors=nn, MN_ratio=mn, FP_ratio=2.0, random_state=42)
        if has_init:
            kwargs["init"] = ini
        model = pacmap.PaCMAP(**kwargs)
        embedding = model.fit_transform(X)
        runtime = time.time() - start

        tw = trustworthiness(X, embedding)
        ct = trustworthiness(embedding, X)
        acc = float(
            cross_val_score(
                KNeighborsClassifier(n_neighbors=5),
                embedding,
                y,
                cv=5,
            ).mean()
        )

        rec = {
            "n_neighbors": nn,
            "MN_ratio": mn,
            "n_components": nc,
            "trustworthiness": tw,
            "continuity": ct,
            "knn_accuracy": acc,
            "runtime_s": runtime,
        }
        if ini is not None:
            rec["init"] = ini
        results.append(rec)

    metrics_df = pd.DataFrame(results)
    metrics_path = out_dir / "pacmap_tuning_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    generated_files.append(metrics_path)

    # Select best two configs by trustworthiness
    top = metrics_df.sort_values("trustworthiness", ascending=False).head(2)

    for _, row in top.iterrows():
        nn = int(row["n_neighbors"])
        mn = float(row["MN_ratio"])
        nc = int(row["n_components"])
        ini = row.get("init") if has_init else None
        kwargs = dict(n_components=nc, n_neighbors=nn, MN_ratio=mn, FP_ratio=2.0, random_state=42)
        if has_init:
            kwargs["init"] = ini
        model = pacmap.PaCMAP(**kwargs)
        embedding = model.fit_transform(X)

        # Save coordinates
        coord_df = pd.DataFrame(
            embedding,
            index=df_active.index,
            columns=[f"PACMAP{i+1}" for i in range(nc)],
        )
        coords_path = out_dir / f"pacmap_coords_{nn}_{mn}_{nc}D.csv"
        coord_df.to_csv(coords_path, index=True)
        generated_files.append(coords_path)

        # Save model
        model_path = out_dir / f"pacmap_model_{nn}_{mn}_{nc}D.joblib"
        joblib.dump(model, model_path)
        generated_files.append(model_path)

        # 2D scatter
        import matplotlib.pyplot as plt
        import seaborn as sns

        if nc >= 2:
            plt.figure(figsize=(12, 6), dpi=200)
            cats = y.astype("category")
            palette = sns.color_palette("tab10", len(cats.cat.categories))
            for cat, color in zip(cats.cat.categories, palette):
                mask = cats == cat
                plt.scatter(
                    coord_df.loc[mask, "PACMAP1"],
                    coord_df.loc[mask, "PACMAP2"],
                    s=10,
                    alpha=0.7,
                    color=color,
                    label=str(cat),
                )
            plt.xlabel("PACMAP1")
            plt.ylabel("PACMAP2")
            plt.title("PaCMAP - 2D")
            plt.legend(title="Statut commercial", bbox_to_anchor=(1.05, 1), loc="upper left")
            plt.tight_layout()
            fig_path = fig_dir / f"pacmap_{nn}_{mn}_2D.png"
            plt.savefig(fig_path)
            plt.close()
            generated_files.append(fig_path)

        if nc >= 3:
            from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

            fig = plt.figure(figsize=(12, 6), dpi=200)
            ax = fig.add_subplot(111, projection="3d")
            cats = y.astype("category")
            palette = sns.color_palette("tab10", len(cats.cat.categories))
            for cat, color in zip(cats.cat.categories, palette):
                mask = cats == cat
                ax.scatter(
                    coord_df.loc[mask, "PACMAP1"],
                    coord_df.loc[mask, "PACMAP2"],
                    coord_df.loc[mask, "PACMAP3"],
                    s=10,
                    alpha=0.7,
                    color=color,
                    label=str(cat),
                )
            ax.set_xlabel("PACMAP1")
            ax.set_ylabel("PACMAP2")
            ax.set_zlabel("PACMAP3")
            ax.set_title("PaCMAP - 3D")
            ax.legend(title="Statut commercial", bbox_to_anchor=(1.05, 1), loc="upper left")
            plt.tight_layout()
            fig_path = fig_dir / f"pacmap_{nn}_{mn}_3D.png"
            plt.savefig(fig_path)
            plt.close()
            generated_files.append(fig_path)

    # Write index file
    index_path = out_dir / "index_pacmap.txt"
    with open(index_path, "w", encoding="utf-8") as f:
        for p in generated_files:
            f.write(str(Path(p).resolve()) + "\n")
    

if __name__ == "__main__":
    main()
