#!/usr/bin/env python3
"""Generate a heatmap comparing two clustering solutions."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from phase4_functions import (
    cluster_confusion_table,
    plot_cluster_confusion_heatmap,
)


def _read_labels(path: Path) -> pd.Series:
    """Return a label series from ``path`` CSV/Excel file."""
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        df = pd.read_excel(path)
    if df.shape[1] == 1:
        return df.iloc[:, 0]
    for col in df.columns:
        if col.lower() in {"cluster", "label", "labels"}:
            return df[col]
    raise ValueError(f"cannot determine label column in {path}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Cross-tabulate clustering solutions")
    parser.add_argument("--labels_a", required=True, help="First labels CSV/Excel")
    parser.add_argument("--labels_b", required=True, help="Second labels CSV/Excel")
    parser.add_argument("--output", default="clusters_confusion.png", help="Output PNG path")
    parser.add_argument("--normalize", action="store_true", help="Show percentages instead of counts")
    args = parser.parse_args(argv)

    a = _read_labels(Path(args.labels_a))
    b = _read_labels(Path(args.labels_b))
    table = cluster_confusion_table(a, b)
    fig = plot_cluster_confusion_heatmap(
        table,
        "Correspondance des clusters",
        normalize=args.normalize,
    )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    print(out)


if __name__ == "__main__":  # pragma: no cover - CLI
    main()
