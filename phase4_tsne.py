#!/usr/bin/env python3
"""Standalone t-SNE analysis using functions from phase4v2."""

import argparse
import logging
from pathlib import Path

from phase4v2 import (
    load_data,
    prepare_data,
    select_variables,
    sanity_check,
    handle_missing_values,
    segment_data,
    run_famd,
    run_tsne,
    export_tsne_results,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run t-SNE only (requires FAMD)")
    parser.add_argument("--input", required=True, help="Excel file")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--perplexity", type=int, default=None, help="t-SNE perplexity")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    out_dir = Path(args.output) / "TSNE"

    df_raw = load_data(args.input)
    df_clean = prepare_data(df_raw)
    df_active_tmp, quant_vars, qual_vars = select_variables(df_clean)
    quant_vars, qual_vars = sanity_check(df_active_tmp, quant_vars, qual_vars)
    df_active = df_active_tmp[quant_vars + qual_vars]
    df_active = handle_missing_values(df_active, quant_vars, qual_vars)
    segment_data(df_active, qual_vars, out_dir)
    df_full = df_clean.loc[df_active.index]

    famd_model, inertia, rows, cols, contrib = run_famd(df_active, quant_vars, qual_vars)

    tsne_model, tsne_df, metrics = run_tsne(
        rows,
        df_active,
        out_dir,
        perplexity=args.perplexity,
    )

    export_tsne_results(tsne_df, df_full, out_dir, metrics)
    logging.info("t-SNE analysis complete")


if __name__ == "__main__":
    main()
