#!/usr/bin/env python3
"""Fine tune PCA on cleaned CRM data."""

import argparse
import logging
from pathlib import Path
import itertools

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from phase4v2 import plot_correlation_circle

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine tune PCA")
    parser.add_argument("--input", required=True, help="Cleaned multivariate CSV")
    parser.add_argument("--output", required=True, help="Output directory")
    return parser.parse_args()

CANDIDATE_QUANT = [
    "Total recette actualisé",
    "Total recette réalisé",
    "Total recette produit",
    "Budget client estimé",
    "duree_projet_jours",
    "taux_realisation",
    "marge_estimee",
]

N_COMPONENTS = [2, 3, 5, 8, 10]
SVD_SOLVERS = ["auto", "full", "randomized"]
WHITEN_OPTIONS = [False, True]

plt.rcParams.update({"figure.figsize": (12, 6), "figure.dpi": 200})


def load_quantitative_data(path: str) -> tuple[pd.DataFrame, list[str]]:
    """Load CSV and return dataframe with numeric columns only."""
    df = pd.read_csv(path)
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    quant_cols = [c for c in CANDIDATE_QUANT if c in numeric_cols]
    if not quant_cols:
        quant_cols = numeric_cols
    df_quant = df[quant_cols].dropna()
    return df_quant, quant_cols


def run_pca_grid(X: np.ndarray, columns: list[str]):
    """Run PCA for all parameter combinations."""
    results = []
    expl_rows = []
    load_rows = []
    contrib_rows = []
    sil_rows = []
    config_id = 0
    max_comp = min(X.shape[0], X.shape[1])
    for n_comp, solver, whiten in itertools.product(N_COMPONENTS, SVD_SOLVERS, WHITEN_OPTIONS):
        if n_comp > max_comp:
            continue
        logging.info("PCA n_components=%d solver=%s whiten=%s", n_comp, solver, whiten)
        try:
            pca = PCA(n_components=n_comp, svd_solver=solver, whiten=whiten, random_state=0)
            X_pca = pca.fit_transform(X)
        except ValueError:
            continue

        cum_var = np.cumsum(pca.explained_variance_ratio_)
        for i, (var_ratio, sv) in enumerate(zip(pca.explained_variance_ratio_, pca.singular_values_), 1):
            expl_rows.append({
                "config_id": config_id,
                "n_components": n_comp,
                "svd_solver": solver,
                "whiten": whiten,
                "component": f"F{i}",
                "explained_variance_ratio": var_ratio,
                "cumulative_variance_ratio": cum_var[i-1],
                "singular_value": sv,
            })

        loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
        for j, col in enumerate(columns):
            for i in range(pca.n_components):
                loading_val = loadings[j, i]
                contrib_val = (loading_val ** 2) / np.sum(loadings[:, i] ** 2) * 100
                load_rows.append({
                    "config_id": config_id,
                    "n_components": n_comp,
                    "svd_solver": solver,
                    "whiten": whiten,
                    "variable": col,
                    "component": f"F{i+1}",
                    "loading": loading_val,
                })
                contrib_rows.append({
                    "config_id": config_id,
                    "n_components": n_comp,
                    "svd_solver": solver,
                    "whiten": whiten,
                    "variable": col,
                    "component": f"F{i+1}",
                    "contribution_pct": contrib_val,
                })

        for k in range(2, 11):
            km = KMeans(n_clusters=k, n_init=10, random_state=0)
            labels = km.fit_predict(X_pca)
            sil = silhouette_score(X_pca, labels)
            sil_rows.append({
                "config_id": config_id,
                "n_components": n_comp,
                "svd_solver": solver,
                "whiten": whiten,
                "n_clusters": k,
                "silhouette_score": sil,
            })

        results.append({
            "config_id": config_id,
            "n_components": n_comp,
            "svd_solver": solver,
            "whiten": whiten,
            "cum_variance": cum_var[-1],
            "scores": X_pca,
            "loadings": loadings,
        })
        config_id += 1

    return results, pd.DataFrame(expl_rows), pd.DataFrame(load_rows), pd.DataFrame(contrib_rows), pd.DataFrame(sil_rows)


def export_csvs(out_dir: Path, exp_df: pd.DataFrame, load_df: pd.DataFrame, contrib_df: pd.DataFrame, sil_df: pd.DataFrame) -> None:
    """Save all CSV outputs."""
    out_dir.mkdir(parents=True, exist_ok=True)
    exp_df.to_csv(out_dir / "explained_variance.csv", index=False)
    load_df.to_csv(out_dir / "loadings.csv", index=False)
    contrib_df.to_csv(out_dir / "contributions.csv", index=False)
    sil_mean = sil_df.groupby(["n_components", "n_clusters"])["silhouette_score"].mean().reset_index()
    sil_mean.to_csv(out_dir / "silhouette_scores.csv", index=False)


def make_scree_plots(out_dir: Path, exp_df: pd.DataFrame) -> None:
    """Generate scree plots for each solver/whiten."""
    for solver in SVD_SOLVERS:
        for whiten in WHITEN_OPTIONS:
            subset = exp_df[(exp_df.svd_solver == solver) & (exp_df.whiten == whiten) & (exp_df.n_components == max(N_COMPONENTS))]
            if subset.empty:
                continue
            ratios = subset.sort_values("component")["explained_variance_ratio"].values
            axes = range(1, len(ratios) + 1)
            plt.figure()
            plt.bar(axes, ratios * 100, edgecolor="black")
            plt.plot(axes, np.cumsum(ratios) * 100, "-o", color="orange")
            plt.xlabel("Composante")
            plt.ylabel("% variance expliquée")
            plt.title(f"Scree plot – solver={solver} whiten={whiten}")
            plt.xticks(axes)
            plt.tight_layout()
            fname = f"scree_{solver}_w{int(whiten)}.png"
            plt.savefig(out_dir / fname)
            plt.close()


def plot_correlation_and_contrib(out_dir: Path, loadings: np.ndarray, columns: list[str], suffix: str) -> None:
    if loadings.shape[1] < 2:
        return
    coords = pd.DataFrame(loadings[:, :2], index=columns, columns=["F1", "F2"])
    plt.figure()
    ax = plt.gca()
    plot_correlation_circle(ax, coords, "Cercle des corrélations F1–F2")
    plt.tight_layout()
    plt.savefig(out_dir / f"correlation_circle_{suffix}.png")
    plt.close()

    contrib = ((loadings[:, :2] ** 2) / np.sum(loadings[:, :2] ** 2, axis=0)) * 100
    contrib_df = pd.DataFrame(contrib, index=columns, columns=["F1", "F2"])
    fig, axes = plt.subplots(1, 2)
    for i, ax in enumerate(axes):
        axis = f"F{i+1}"
        if axis in contrib_df.columns:
            top = contrib_df[axis].sort_values(ascending=False)
            ax.bar(top.index.astype(str), top.values)
            ax.set_title(f"Contributions {axis}")
            ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    plt.savefig(out_dir / f"contributions_{suffix}.png")
    plt.close()


def plot_silhouette_curves(out_dir: Path, sil_df: pd.DataFrame) -> None:
    pivot_nc = sil_df.pivot_table(index="n_components", columns="n_clusters", values="silhouette_score", aggfunc="mean")
    plt.figure()
    pivot_nc.plot(marker="o")
    plt.xlabel("n_components")
    plt.ylabel("Silhouette moyen")
    plt.title("Silhouette vs n_components")
    plt.tight_layout()
    plt.savefig(out_dir / "silhouette_vs_components.png")
    plt.close()

    pivot_k = sil_df.pivot_table(index="n_clusters", columns="n_components", values="silhouette_score", aggfunc="mean")
    plt.figure()
    pivot_k.plot(marker="o")
    plt.xlabel("n_clusters")
    plt.ylabel("Silhouette moyen")
    plt.title("Silhouette vs n_clusters")
    plt.tight_layout()
    plt.savefig(out_dir / "silhouette_vs_clusters.png")
    plt.close()


def plot_best_clusters(out_dir: Path, scores: np.ndarray, n_clusters: int) -> None:
    if scores.shape[1] < 2:
        return
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=0)
    labels = km.fit_predict(scores[:, :2])
    plt.figure()
    sns.scatterplot(x=scores[:, 0], y=scores[:, 1], hue=labels, palette="tab10", s=10)
    plt.xlabel("F1")
    plt.ylabel("F2")
    plt.title("Projection PCA F1–F2 par cluster")
    plt.tight_layout()
    plt.savefig(out_dir / "clusters_scatter.png")
    plt.close()


def write_index(out_dir: Path) -> None:
    paths = []
    for p in out_dir.rglob("*"):
        if p.is_file():
            paths.append(p.relative_to(out_dir))
    with open(out_dir / "index_fine_tune_pca.txt", "w", encoding="utf-8") as f:
        for p in sorted(paths):
            f.write(str(p) + "\n")


def main() -> None:
    args = parse_args()
    data_path = args.input
    out_dir = Path(args.output)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    out_dir.mkdir(parents=True, exist_ok=True)

    df_quant, quant_cols = load_quantitative_data(data_path)
    logging.info("Variables quantitatives utilisées : %s", quant_cols)

    scaler = StandardScaler()
    X = scaler.fit_transform(df_quant)

    results, exp_df, load_df, contrib_df, sil_df = run_pca_grid(X, quant_cols)

    export_csvs(out_dir, exp_df, load_df, contrib_df, sil_df)
    make_scree_plots(out_dir, exp_df)
    plot_silhouette_curves(out_dir, sil_df)

    # Identify best configurations
    exp_summary = exp_df.groupby("config_id")["cumulative_variance_ratio"].max()
    eligible = exp_summary[exp_summary >= 0.80]
    best_var_id = eligible.idxmax() if not eligible.empty else exp_summary.idxmax()
    sil_max = sil_df.groupby("config_id")["silhouette_score"].max()
    best_sil_id = sil_max.idxmax()
    best_k = sil_df[sil_df.config_id == best_sil_id].sort_values("silhouette_score", ascending=False).iloc[0]["n_clusters"]

    best_var_res = next(r for r in results if r["config_id"] == best_var_id)
    best_sil_res = next(r for r in results if r["config_id"] == best_sil_id)

    plot_correlation_and_contrib(out_dir, best_var_res["loadings"], quant_cols, "best_variance")
    plot_correlation_and_contrib(out_dir, best_sil_res["loadings"], quant_cols, "best_silhouette")
    plot_best_clusters(out_dir, best_sil_res["scores"], int(best_k))

    write_index(out_dir)

    logging.info(
        "Meilleure config variance >=80%% : id=%d (solver=%s, whiten=%s, n_comp=%d, variance=%.2f%%)",
        best_var_res["config_id"], best_var_res["svd_solver"], best_var_res["whiten"], best_var_res["n_components"], best_var_res["cum_variance"]*100,
    )
    logging.info(
        "Meilleure config silhouette : id=%d (solver=%s, whiten=%s, n_comp=%d, silhouette=%.3f)",
        best_sil_res["config_id"], best_sil_res["svd_solver"], best_sil_res["whiten"], best_sil_res["n_components"], sil_max[best_sil_id],
    )


if __name__ == "__main__":
    main()
