import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

import phase4.functions as pf


def load_dataset(path: Path) -> pd.DataFrame:
    """Load CSV or Excel dataset based on extension."""
    if path.suffix.lower() in {".xls", ".xlsx"}:
        return pd.read_excel(path)
    return pd.read_csv(path)


def plot_individuals(df: pd.DataFrame, group: str | None, output: Path) -> None:
    quant_vars = df.select_dtypes(include=[np.number]).columns.tolist()
    qual_vars = [c for c in df.columns if c not in quant_vars]
    res = pf.run_famd(df, quant_vars, qual_vars, n_components=2)
    coords = res["embeddings"].iloc[:, :2].rename(columns={0: "Dim1", 1: "Dim2"})
    var1 = float(res["inertia"].iloc[0]) * 100
    var2 = float(res["inertia"].iloc[1]) * 100
    fig, ax = plt.subplots(figsize=(12, 6), dpi=200)
    if group is None or group not in df.columns:
        sns.scatterplot(x="Dim1", y="Dim2", data=coords, s=15, color="tab:blue", ax=ax)
    else:
        cats = df[group].astype("category")
        palette = sns.color_palette("deep", len(cats.cat.categories))
        for color, cat in zip(palette, cats.cat.categories):
            mask = cats == cat
            sns.scatterplot(
                x="Dim1",
                y="Dim2",
                data=coords[mask],
                color=color,
                s=15,
                label=str(cat),
                ax=ax,
            )
            if coords[mask].shape[0] > 2:
                sns.kdeplot(
                    x=coords.loc[mask, "Dim1"],
                    y=coords.loc[mask, "Dim2"],
                    color=color,
                    fill=False,
                    ax=ax,
                )
        ax.legend(title=group)
    ax.set_xlabel(f"Dim1 ({var1:.1f}%)")
    ax.set_ylabel(f"Dim2 ({var2:.1f}%)")
    ax.set_title("FAMD – Carte des individus (Axes 1-2)")
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="FAMD individual plot")
    parser.add_argument("input_file", type=Path, help="CSV or Excel file")
    parser.add_argument("--group", default=None, help="Column for colour groups")
    parser.add_argument("--output", type=Path, default=Path("famd_individuals.png"))
    args = parser.parse_args()

    df = load_dataset(args.input_file)
    plot_individuals(df, args.group, args.output)


if __name__ == "__main__":  # pragma: no cover
    main()
