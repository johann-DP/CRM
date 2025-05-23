#!/usr/bin/env python3
"""Standalone UMAP analysis using functions from phase4v2."""

import argparse
import logging
from pathlib import Path

from phase4v2 import run_umap, export_umap_results
from standalone_utils import prepare_active_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Run UMAP only")
    parser.add_argument("--input", required=True, help="Excel file")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--n_components", type=int, default=2, help="Number of components")
    parser.add_argument("--n_neighbors", type=int, default=None, help="Number of neighbors")
    parser.add_argument("--min_dist", type=float, default=None, help="Minimum distance")
    parser.add_argument("--metric", default="euclidean", help="UMAP distance metric")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    out_dir = Path(args.output) / "UMAP"

    df_active, quant_vars, qual_vars = prepare_active_dataset(args.input, out_dir)

    model, emb = run_umap(
        df_active,
        quant_vars,
        qual_vars,
        out_dir,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        n_components=args.n_components,
        metric=args.metric,
    )

    export_umap_results(emb, df_active, out_dir)
    logging.info("UMAP analysis complete")


if __name__ == "__main__":
    main()
