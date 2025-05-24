import argparse
import os
import time
import logging
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import prince

from phase4v2 import plot_correlation_circle

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# ----------------------------------------------------------------------
# Configuration paths
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine tune MCA")
    parser.add_argument("--phase1", required=True, help="Phase1 categorical CSV")
    parser.add_argument("--phase2", required=True, help="Phase2 categorical CSV")
    parser.add_argument("--phase3", required=True, help="Phase3 multivariate CSV")
    parser.add_argument("--output", required=True, help="Output directory")
    return parser.parse_args()

INPUT_FILES = []
OUTPUT_ROOT = Path()
FIG_DIR = Path()
CSV_DIR = Path()

QUAL_COLS = [
    "Statut commercial",
    "Statut production",
    "Type opportunité",
    "Catégorie",
    "Sous-catégorie",
    "Pilier",
    "Entité opérationnelle",
]


# ----------------------------------------------------------------------
def load_inputs(paths: list[str]) -> pd.DataFrame:
    """Load input CSV files and keep qualitative columns."""
    logger = logging.getLogger(__name__)
    frames = []
    for path in paths:
        try:
            df = pd.read_csv(path)
        except FileNotFoundError:
            logger.error("File not found: %s", path)
            continue
        frames.append(df)
        logger.info("Loaded %s with shape %s", path, df.shape)

    if not frames:
        raise FileNotFoundError("No input files were loaded")

    df_all = pd.concat(frames, ignore_index=True, sort=False)
    existing = [c for c in QUAL_COLS if c in df_all.columns]
    df_all = df_all[existing].copy()
    df_all.fillna("Inconnu", inplace=True)
    for col in existing:
        df_all[col] = df_all[col].astype(str)
    return df_all


# ----------------------------------------------------------------------
def ensure_dirs() -> None:
    global FIG_DIR, CSV_DIR
    FIG_DIR = OUTPUT_ROOT / "figures"
    CSV_DIR = OUTPUT_ROOT / "csv"
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    CSV_DIR.mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------------------------
def compute_contributions(coords: pd.DataFrame) -> pd.DataFrame:
    contrib = (coords ** 2)
    contrib = contrib.div(contrib.sum(axis=0), axis=1) * 100
    return contrib


# ----------------------------------------------------------------------
def plot_scree(inertia: np.ndarray, base: str) -> Path:
    axes = range(1, len(inertia) + 1)
    plt.figure(figsize=(12, 6), dpi=200)
    plt.bar(axes, inertia * 100, edgecolor="black")
    plt.plot(axes, np.cumsum(inertia) * 100, "-o", color="orange")
    plt.xlabel("Composante")
    plt.ylabel("% Inertie expliquée")
    plt.title("Éboulis MCA")
    plt.xticks(list(axes))
    plt.tight_layout()
    path = FIG_DIR / f"scree_{base}.png"
    plt.savefig(path)
    plt.close()
    return path


# ----------------------------------------------------------------------
def plot_correlation(coords: pd.DataFrame, base: str, axes_pair: tuple[str, str]) -> Path:
    if not set(axes_pair).issubset(coords.columns):
        return Path()
    fig = plt.figure(figsize=(12, 6), dpi=200)
    ax = plt.gca()
    subset = coords[list(axes_pair)].copy()
    subset.columns = ["F1", "F2"]
    plot_correlation_circle(
        ax,
        subset,
        f"Cercle des corrélations ({axes_pair[0]}–{axes_pair[1]})",
    )
    plt.tight_layout()
    path = FIG_DIR / f"circle_{axes_pair[0].lower()}_{axes_pair[1].lower()}_{base}.png"
    plt.savefig(path)
    plt.close(fig)
    return path


# ----------------------------------------------------------------------
def plot_individuals(df_coords: pd.DataFrame, df_source: pd.DataFrame, base: str) -> tuple[Path, Path]:
    p2 = p3 = None
    if {"F1", "F2"}.issubset(df_coords.columns):
        plt.figure(figsize=(12, 6), dpi=200)
        cats = df_source.loc[df_coords.index, "Statut commercial"].astype(str)
        palette = plt.get_cmap("tab10")(np.linspace(0, 1, cats.nunique()))
        for color, cat in zip(palette, cats.unique()):
            mask = cats == cat
            plt.scatter(df_coords.loc[mask, "F1"], df_coords.loc[mask, "F2"], s=10, alpha=0.7, color=color, label=str(cat))
        plt.legend(title="Statut commercial", bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.xlabel("F1"); plt.ylabel("F2")
        plt.title("MCA – individus (F1–F2)")
        plt.tight_layout()
        p2 = FIG_DIR / f"indiv_2d_{base}.png"
        plt.savefig(p2)
        plt.close()
    if {"F1", "F2", "F3"}.issubset(df_coords.columns):
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        fig = plt.figure(figsize=(12, 6), dpi=200)
        ax = fig.add_subplot(111, projection="3d")
        cats = df_source.loc[df_coords.index, "Statut commercial"].astype(str)
        palette = plt.get_cmap("tab10")(np.linspace(0, 1, cats.nunique()))
        for color, cat in zip(palette, cats.unique()):
            mask = cats == cat
            ax.scatter(
                df_coords.loc[mask, "F1"],
                df_coords.loc[mask, "F2"],
                df_coords.loc[mask, "F3"],
                s=10,
                alpha=0.7,
                color=color,
                label=str(cat),
            )
        ax.legend(title="Statut commercial", bbox_to_anchor=(1.05, 1), loc="upper left")
        ax.set_xlabel("F1"); ax.set_ylabel("F2"); ax.set_zlabel("F3")
        ax.set_title("MCA – individus (3D)")
        plt.tight_layout()
        p3 = FIG_DIR / f"indiv_3d_{base}.png"
        plt.savefig(p3)
        plt.close()
    return p2, p3


# ----------------------------------------------------------------------
def plot_modalities(col_coords: pd.DataFrame, base: str) -> tuple[Path, Path]:
    p1 = p2 = None
    if {"F1", "F2"}.issubset(col_coords.columns):
        plt.figure(figsize=(12, 6), dpi=200)
        var_names = [m.split("__", 1)[0] if "__" in m else m for m in col_coords.index]
        palette = plt.get_cmap("tab10")(np.linspace(0, 1, len(set(var_names))))
        color_map = {v: palette[i] for i, v in enumerate(sorted(set(var_names)))}
        for mod, var in zip(col_coords.index, var_names):
            plt.scatter(col_coords.loc[mod, "F1"], col_coords.loc[mod, "F2"], color=color_map[var], s=20, alpha=0.7)
        for mod, var in zip(col_coords.index, var_names):
            plt.text(col_coords.loc[mod, "F1"], col_coords.loc[mod, "F2"], mod.replace("__", "="), fontsize=8)
        plt.xlabel("F1"); plt.ylabel("F2")
        plt.title("MCA – modalités (F1–F2)")
        plt.tight_layout()
        p1 = FIG_DIR / f"modalities_{base}.png"
        plt.savefig(p1)
        plt.close()
        # Zoom on near-origin modalities
        radius = np.sqrt(col_coords["F1"] ** 2 + col_coords["F2"] ** 2)
        thresh = radius.quantile(0.1)
        near = col_coords[radius <= thresh]
        if not near.empty:
            plt.figure(figsize=(12, 6), dpi=200)
            for mod, var in zip(near.index, [m.split("__", 1)[0] if "__" in m else m for m in near.index]):
                plt.scatter(near.loc[mod, "F1"], near.loc[mod, "F2"], color=color_map[var], s=20, alpha=0.7)
                plt.text(near.loc[mod, "F1"], near.loc[mod, "F2"], mod.replace("__", "="), fontsize=8)
            plt.xlabel("F1"); plt.ylabel("F2")
            plt.title("MCA – modalités proches (zoom)")
            plt.tight_layout()
            p2 = FIG_DIR / f"modalities_zoom_{base}.png"
            plt.savefig(p2)
            plt.close()
    return p1, p2


# ----------------------------------------------------------------------
def save_csvs(base: str, inertia: np.ndarray, row_coords: pd.DataFrame, col_coords: pd.DataFrame, contrib: pd.DataFrame) -> None:
    df_var = pd.DataFrame({
        "axe": [f"F{i+1}" for i in range(len(inertia))],
        "variance_%": inertia * 100,
    })
    df_var["variance_cum_%"] = df_var["variance_%"].cumsum()
    df_var.to_csv(CSV_DIR / f"explained_inertia_{base}.csv", index=False)
    row_coords.to_csv(CSV_DIR / f"individuals_coords_{base}.csv", index=True)
    col_coords.to_csv(CSV_DIR / f"modalities_coords_{base}.csv", index=True)
    contrib.to_csv(CSV_DIR / f"modalities_contrib_{base}.csv", index=True)


# ----------------------------------------------------------------------
def assemble_pdf(figures: list[Path], pdf_path: Path) -> None:
    with PdfPages(pdf_path) as pdf:
        for fig_path in figures:
            if not fig_path or not fig_path.exists():
                continue
            img = plt.imread(fig_path)
            fig, ax = plt.subplots(figsize=(12, 6), dpi=200)
            ax.imshow(img)
            ax.axis("off")
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)


# ----------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    global INPUT_FILES, OUTPUT_ROOT
    INPUT_FILES = [args.phase1, args.phase2, args.phase3]
    OUTPUT_ROOT = Path(args.output)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    ensure_dirs()
    df = load_inputs(INPUT_FILES)

    results = []
    all_figs: list[Path] = []

    for n_comp in [5, 10, 15]:
        for normalize in [True, False]:
            corr = "benzecri" if normalize else None
            for n_iter in [3, 5, 10]:
                base = f"n{n_comp}_norm{normalize}_iter{n_iter}"
                logging.info("Running MCA %s", base)
                start = time.perf_counter()
                mca = prince.MCA(n_components=n_comp, n_iter=n_iter, correction=corr, random_state=42)
                mca = mca.fit(df)
                duration = time.perf_counter() - start

                inertia = mca.eigenvalues_ / mca.eigenvalues_.sum()
                rows = mca.row_coordinates(df)
                cols = mca.column_coordinates(df)
                cols.index = cols.index.str.replace("__", "=")
                rows.columns = [f"F{i+1}" for i in range(rows.shape[1])]
                cols.columns = [f"F{i+1}" for i in range(cols.shape[1])]
                contrib = compute_contributions(cols)

                # Save CSV
                save_csvs(base, inertia, rows, cols, contrib)

                # Figures
                fig_paths = []
                fig_paths.append(plot_scree(inertia, base))
                fig_path = plot_correlation(cols, base, ("F1", "F2"))
                if fig_path:
                    fig_paths.append(fig_path)
                if "F3" in cols.columns:
                    fig_paths.append(plot_correlation(cols, base, ("F1", "F3")))
                indiv2d, indiv3d = plot_individuals(rows, df, base)
                mod, mod_zoom = plot_modalities(cols, base)
                fig_paths.extend([indiv2d, indiv3d, mod, mod_zoom])
                all_figs.extend([p for p in fig_paths if p])

                results.append({
                    "n_components": n_comp,
                    "normalize": normalize,
                    "n_iter": n_iter,
                    "explained_variance_cum": float(np.cumsum(inertia)[-1]),
                    "time_seconds": duration,
                })

    assemble_pdf(all_figs, OUTPUT_ROOT / "mca_fine_tuning_results.pdf")
    pd.DataFrame(results).to_csv(OUTPUT_ROOT / "tuning_report.csv", index=False)
    logging.info("Fine-tuning MCA complete")


if __name__ == "__main__":
    main()
