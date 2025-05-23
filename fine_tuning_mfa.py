#!/usr/bin/env python3
"""Fine-tune a Multiple Factor Analysis on the cleaned CRM dataset."""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import prince

# -----------------------------------------------------------------------------
# Configuration
QUANT_VARS = [
    "total_recette_actualise",
    "total_recette_produit",
    "budget_client_estime",
    "taux_realisation",
]

QUAL_VARS = [
    "Categorie",
    "Sous-categorie",
    "Entite_operationnelle",
    "Statut_production",
    "Type_opportunite",
    "Statut_commercial",
    "Pilier",
]

DATA_FILE = Path("/mnt/data/phase3_cleaned_multivariate.csv")
OUTPUT_DIR = Path("/mnt/data/phase4_output/fine_tuning_mfa")

# -----------------------------------------------------------------------------

def load_dataset(path: Path) -> pd.DataFrame:
    """Read the CSV file with basic error handling."""
    try:
        df = pd.read_csv(path)
        logging.info("Loaded dataset: %s", path)
        return df
    except Exception as exc:
        logging.error("Failed to read %s: %s", path, exc)
        raise


def preprocess(df: pd.DataFrame):
    """Return standardized quantitative block, encoded qualitative block and group mapping."""
    df = df[QUANT_VARS + QUAL_VARS].copy()

    scaler_q = StandardScaler()
    df_quant = pd.DataFrame(
        scaler_q.fit_transform(df[QUANT_VARS]),
        index=df.index,
        columns=QUANT_VARS,
    )

    dummies = pd.get_dummies(df[QUAL_VARS].astype(str))
    scaler_c = StandardScaler()
    df_qual = pd.DataFrame(
        scaler_c.fit_transform(dummies),
        index=df.index,
        columns=dummies.columns,
    )

    groups = {"quant": list(df_quant.columns)}
    for var in QUAL_VARS:
        groups[var] = [c for c in df_qual.columns if c.startswith(f"{var}_")]

    return df_quant, df_qual, groups


def scale_blocks(blocks: dict, weighting: str) -> dict:
    """Scale each block depending on weighting scheme."""
    scaled = {}
    for name, block in blocks.items():
        pca = PCA(n_components=1).fit(block)
        if weighting == "uniform":
            scaled[name] = block / np.sqrt(pca.explained_variance_[0])
        else:
            scaled[name] = block.copy()
    return scaled


def fit_mfa(blocks: dict, n_components: int, weighting: str):
    """Fit MFA with the desired weighting."""
    scaled_blocks = scale_blocks(blocks, weighting)
    X = pd.concat(scaled_blocks.values(), axis=1)
    groups = {g: list(b.columns) for g, b in scaled_blocks.items()}
    mfa = prince.MFA(n_components=n_components, n_iter=3, engine="sklearn")
    mfa = mfa.fit(X, groups=groups)
    mfa.df_encoded_ = X
    mfa.groups_input_ = groups
    rows = mfa.row_coordinates(X)
    return mfa, rows


def compute_best_params(blocks: dict):
    """Grid search over n_components and weighting options."""
    best = None
    best_cum = -np.inf
    for weighting in ["uniform", "inertia"]:
        for n_comp in range(2, 9):
            model, _ = fit_mfa(blocks, n_comp, weighting)
            inertia = np.cumsum(model.percentage_of_variance_)
            cum = inertia[min(n_comp, len(inertia)) - 1]
            logging.info(
                "Test weight=%s n_components=%d -> %.1f%% variance",
                weighting,
                n_comp,
                cum,
            )
            if cum >= 80 and (best is None or n_comp < best[2]):
                best = (weighting, n_comp, n_comp)
                return best[0], best[1]
            if cum > best_cum:
                best = (weighting, n_comp, n_comp)
                best_cum = cum
    return best[0], best[1]


def export_results(model, rows, out_dir: Path, df_active: pd.DataFrame):
    """Export CSV files and figures."""
    out_dir.mkdir(parents=True, exist_ok=True)

    axes = [f"F{i+1}" for i in range(rows.shape[1])]
    rows.columns = axes

    cols = model.column_coordinates_.copy()
    cols.columns = axes[: cols.shape[1]]
    contrib = (cols ** 2).div((cols ** 2).sum(axis=0), axis=1) * 100

    inertia = pd.DataFrame(
        {
            "eigenvalue": model.eigenvalues_,
            "variance_%": model.percentage_of_variance_,
        }
    )
    inertia["cumulative_%"] = inertia["variance_%"].cumsum()

    inertia.to_csv(out_dir / "mfa_eigenvalues.csv", index=False)
    rows.to_csv(out_dir / "mfa_row_coordinates.csv")
    cols.to_csv(out_dir / "mfa_col_coordinates.csv")
    contrib.to_csv(out_dir / "mfa_contributions.csv")

    # Scree
    plt.figure(figsize=(6, 4), dpi=200)
    plt.bar(range(1, len(inertia) + 1), inertia["variance_%"])
    plt.xlabel("Component")
    plt.ylabel("% Variance")
    plt.title("Scree plot")
    plt.tight_layout()
    plt.savefig(out_dir / "mfa_scree_plot.png")
    plt.close()

    plt.figure(figsize=(6, 4), dpi=200)
    plt.plot(range(1, len(inertia) + 1), inertia["cumulative_%"], marker="o")
    plt.xlabel("Component")
    plt.ylabel("Cumulative %")
    plt.title("Cumulative inertia")
    plt.tight_layout()
    plt.savefig(out_dir / "mfa_cumulative_inertia.png")
    plt.close()

    if {"F1", "F2"}.issubset(rows.columns):
        plt.figure(figsize=(6, 6), dpi=200)
        if "Statut_commercial" in df_active.columns:
            cats = df_active["Statut_commercial"].astype("category")
            palette = sns.color_palette("tab10", len(cats.cat.categories))
            for cat, color in zip(cats.cat.categories, palette):
                mask = cats == cat
                plt.scatter(
                    rows.loc[mask, "F1"],
                    rows.loc[mask, "F2"],
                    s=10,
                    alpha=0.7,
                    color=color,
                    label=str(cat),
                )
            plt.legend(title="Statut_commercial")
        else:
            plt.scatter(rows["F1"], rows["F2"], s=10, alpha=0.7)
        plt.xlabel("F1")
        plt.ylabel("F2")
        plt.title("Individus")
        plt.tight_layout()
        plt.savefig(out_dir / "mfa_indiv_scatter_2d.png")
        plt.close()

    if {"F1", "F2", "F3"}.issubset(rows.columns):
        fig = plt.figure(figsize=(8, 6), dpi=200)
        ax = fig.add_subplot(111, projection="3d")
        if "Statut_commercial" in df_active.columns:
            cats = df_active["Statut_commercial"].astype("category")
            palette = sns.color_palette("tab10", len(cats.cat.categories))
            for cat, color in zip(cats.cat.categories, palette):
                mask = cats == cat
                ax.scatter(
                    rows.loc[mask, "F1"],
                    rows.loc[mask, "F2"],
                    rows.loc[mask, "F3"],
                    s=10,
                    alpha=0.7,
                    color=color,
                    label=str(cat),
                )
            ax.legend(title="Statut_commercial")
        else:
            ax.scatter(rows["F1"], rows["F2"], rows["F3"], s=10, alpha=0.7)
        ax.set_xlabel("F1")
        ax.set_ylabel("F2")
        ax.set_zlabel("F3")
        ax.set_title("Individus (3D)")
        plt.tight_layout()
        plt.savefig(out_dir / "mfa_indiv_scatter_3d.png")
        plt.close()

    if {"F1", "F2"}.issubset(cols.columns):
        vars_coords = cols.loc[[v for v in QUANT_VARS if v in cols.index]]
        if not vars_coords.empty:
            fig, ax = plt.subplots(figsize=(6, 6), dpi=200)
            circle = plt.Circle((0, 0), 1, color="grey", fill=False, linestyle="dashed")
            ax.add_patch(circle)
            ax.axhline(0, color="grey", lw=0.5)
            ax.axvline(0, color="grey", lw=0.5)
            for var in vars_coords.index:
                x, y = vars_coords.loc[var, ["F1", "F2"]]
                ax.arrow(0, 0, x, y, head_width=0.02, length_includes_head=True)
                ax.text(x * 1.1, y * 1.1, var, fontsize=8, ha="center", va="center")
            ax.set_xlim(-1.1, 1.1)
            ax.set_ylim(-1.1, 1.1)
            ax.set_xlabel("F1")
            ax.set_ylabel("F2")
            ax.set_title("Correlation circle")
            ax.set_aspect("equal")
            plt.tight_layout()
            plt.savefig(out_dir / "mfa_correlation_circle.png")
            plt.close()

        mods = cols.drop(index=[v for v in QUANT_VARS if v in cols.index], errors="ignore")
        if not mods.empty:
            plt.figure(figsize=(6, 6), dpi=200)
            plt.scatter(mods["F1"], mods["F2"], s=20, alpha=0.7)
            for mod in mods.index:
                plt.text(mods.loc[mod, "F1"], mods.loc[mod, "F2"], mod, fontsize=8)
            plt.xlabel("F1")
            plt.ylabel("F2")
            plt.title("Modalit\xe9s")
            plt.tight_layout()
            plt.savefig(out_dir / "mfa_modalities_scatter.png")
            plt.close()


# -----------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    try:
        df = load_dataset(DATA_FILE)
    except Exception:
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df_quant, df_qual, groups = preprocess(df)
    blocks = {"quant": df_quant}
    for var in QUAL_VARS:
        blocks[var] = df_qual[groups[var]]

    weighting, n_comp = compute_best_params(blocks)
    logging.info("Selected parameters: weights=%s, n_components=%d", weighting, n_comp)

    model, rows = fit_mfa(blocks, n_comp, weighting)
    export_results(model, rows, OUTPUT_DIR, df)
    logging.info("MFA fine-tuning complete")


if __name__ == "__main__":
    main()
