#!/usr/bin/env python3
"""Standalone t-SNE analysis using functions from phase4v2."""

import argparse
import logging
from pathlib import Path

from phase4v2 import run_famd, run_tsne, export_tsne_results
from standalone_utils import prepare_active_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Run t-SNE only (requires FAMD)")
    parser.add_argument("--input", required=True, help="Excel file")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--perplexity", type=int, default=None, help="t-SNE perplexity")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    out_dir = Path(args.output) / "TSNE"

    df_active, quant_vars, qual_vars = prepare_active_dataset(args.input, out_dir)

    famd_model, inertia, rows, cols, contrib = run_famd(df_active, quant_vars, qual_vars)

    tsne_model, tsne_df, metrics = run_tsne(
        rows,
        df_active,
        out_dir,
        perplexity=args.perplexity,
    )

    export_tsne_results(tsne_df, df_active, out_dir, metrics)
    logging.info("t-SNE analysis complete")


if __name__ == "__main__":
    main()
