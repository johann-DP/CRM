#!/usr/bin/env python3
"""Standalone PaCMAP analysis using functions from phase4v2."""

import argparse
import logging
from pathlib import Path

from phase4v2 import run_pacmap, export_pacmap_results
from standalone_utils import prepare_active_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PaCMAP only")
    parser.add_argument("--input", required=True, help="Excel file")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--n_components", type=int, default=2, help="Number of components")
    parser.add_argument("--n_neighbors", type=int, default=None, help="Number of neighbors")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    out_dir = Path(args.output) / "PaCMAP"

    df_active, quant_vars, qual_vars = prepare_active_dataset(args.input, out_dir)

    model, emb = run_pacmap(
        df_active,
        quant_vars,
        qual_vars,
        out_dir,
        n_components=args.n_components,
        n_neighbors=args.n_neighbors,
    )

    if model is not None:
        export_pacmap_results(emb, df_active, out_dir)
    logging.info("PaCMAP analysis complete")


if __name__ == "__main__":
    main()
