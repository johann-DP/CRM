#!/usr/bin/env python3
"""Standalone FAMD analysis with optional fine-tuning.

This script loads the CRM Excel export, prepares the active dataset and runs
FAMD using helper functions from :mod:`phase4v2`.  It exposes a few command
line options to tune the analysis, notably the automatic selection of the
number of components.
"""

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
    export_famd_results,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run FAMD only")
    parser.add_argument("--input", required=True, help="Excel file")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--n_components", type=int, default=None,
                        help="Number of components")
    parser.add_argument("--optimize", action="store_true",
                        help="Automatically pick the number of components")
    parser.add_argument("--rule", choices=["variance", "kaiser", "elbow"],
                        help="Selection rule when optimizing")
    parser.add_argument("--variance_threshold", type=float, default=0.9,
                        help="Variance threshold for the variance rule")
    parser.add_argument("--weighting", choices=["balanced", "auto", "manual"],
                        default="balanced", help="Variable weighting")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    out_dir = Path(args.output) / "FAMD"

    df_raw = load_data(args.input)
    df_clean = prepare_data(df_raw)
    df_active_tmp, quant_vars, qual_vars = select_variables(df_clean)
    quant_vars, qual_vars = sanity_check(df_active_tmp, quant_vars, qual_vars)
    df_active = df_active_tmp[quant_vars + qual_vars]
    df_active = handle_missing_values(df_active, quant_vars, qual_vars)
    segment_data(df_active, qual_vars, out_dir)
    df_full = df_clean.loc[df_active.index]

    famd_cfg = {
        "weighting": args.weighting,
        "n_components_rule": args.rule,
        "variance_threshold": args.variance_threshold,
    }

    famd, inertia, rows, cols, contrib = run_famd(
        df_active,
        quant_vars,
        qual_vars,
        n_components=args.n_components,
        famd_cfg=famd_cfg,
        optimize=args.optimize,
    )

    export_famd_results(
        famd,
        inertia,
        rows,
        cols,
        contrib,
        quant_vars,
        qual_vars,
        out_dir,
        df_active=df_full,
    )
    logging.info("FAMD analysis complete")


if __name__ == "__main__":
    main()
