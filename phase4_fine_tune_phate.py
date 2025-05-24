#!/usr/bin/env python3
"""Fine-tuning PHATE on Phase 3 cleaned datasets.

This script loads the cleaned CSV files produced at the end of phase 3,
performs basic preprocessing (imputation, scaling, encoding) and runs the
PHATE algorithm. The resulting embeddings and a simple scatter plot are
saved in the chosen output directory.
"""

import argparse
import os
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import phate


_DEF_DROP = {"id", "code", "client", "contact"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine tune PHATE on Phase 3 data")
    parser.add_argument("--multi", required=True, help="Cleaned multivariate CSV")
    parser.add_argument("--univ", required=False, help="Cleaned univariate CSV")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--knn", type=int, default=10, help="Number of neighbors")
    parser.add_argument("--decay", type=int, default=20, help="Kernel decay")
    parser.add_argument("--n_components", type=int, default=2, help="Dimensions")
    parser.add_argument("--t", type=int, default=20, help="Diffusion steps")
    return parser.parse_args()


def preprocess(df: pd.DataFrame) -> tuple[pd.DataFrame, List[str], List[str], np.ndarray]:
    drop_cols = [c for c in df.columns if any(p in c.lower() for p in _DEF_DROP) or "date" in c.lower()]
    df = df.drop(columns=drop_cols, errors="ignore")

    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = [c for c in df.columns if c not in num_cols]

    if num_cols:
        for c in num_cols:
            if df[c].isna().all():
                df[c] = 0
            else:
                df[c] = df[c].fillna(df[c].median())
    for c in cat_cols:
        df[c] = df[c].fillna("Non renseigné")
        if df[c].dtype == bool:
            df[c] = df[c].astype(str)

    scaler = StandardScaler()
    X_num = scaler.fit_transform(df[num_cols]) if num_cols else np.empty((len(df), 0))

    try:
        enc = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")
    except TypeError:  # pragma: no cover - older scikit-learn
        enc = OneHotEncoder(drop="first", sparse=False, handle_unknown="ignore")
    X_cat = enc.fit_transform(df[cat_cols]) if cat_cols else np.empty((len(df), 0))

    X = np.hstack([X_num, X_cat])
    return df, num_cols, cat_cols, X


def run_phate(X: np.ndarray, knn: int, decay: int, n_components: int, t: int) -> np.ndarray:
    ph = phate.PHATE(
        knn=knn,
        decay=decay,
        n_components=n_components,
        t=t,
        random_state=42,
    )
    return ph.fit_transform(X)


def main() -> None:
    args = parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    df_multi = pd.read_csv(args.multi)
    if args.univ:
        pd.read_csv(args.univ)  # loaded for completeness but not used further

    df_multi, num_cols, cat_cols, X = preprocess(df_multi)
    emb = run_phate(X, args.knn, args.decay, args.n_components, args.t)

    cols = [f"PHATE{i+1}" for i in range(args.n_components)]
    df_emb = pd.DataFrame(emb, columns=cols)
    df_emb.to_csv(out / "phate_coordinates.csv", index=False)

    if {"PHATE1", "PHATE2"}.issubset(df_emb.columns) and "Statut commercial" in df_multi.columns:
        plot_df = df_emb.join(df_multi["Statut commercial"])
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x="PHATE1", y="PHATE2", hue="Statut commercial", data=plot_df, palette="tab10")
        plt.title("PHATE – Statut commercial")
        plt.tight_layout()
        plt.savefig(out / "phate_scatter_statut_commercial.png")
        plt.close()

    readme = out / "README.md"
    with readme.open("w", encoding="utf-8") as fh:
        fh.write("# Fine-tuning PHATE\n")
        fh.write("Input files: ``{}``, ``{}``\n".format(args.multi, args.univ or "n/a"))
        fh.write("\n")
        fh.write("Hyperparameters:\n")
        fh.write(f"- knn = {args.knn}\n")
        fh.write(f"- decay = {args.decay}\n")
        fh.write(f"- n_components = {args.n_components}\n")
        fh.write(f"- t = {args.t}\n")
        fh.write("\n")
        fh.write("Run with:\n")
        fh.write("```bash\n")
        fh.write(
            f"python {Path(__file__).name} --multi '{args.multi}' --univ '{args.univ or ''}' --output '{args.output}'\n"
        )
        fh.write("```\n")


if __name__ == "__main__":
    main()
