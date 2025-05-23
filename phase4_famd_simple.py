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

from phase4v2 import run_famd, export_famd_results
from standalone_utils import prepare_active_dataset


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

    df_active, quant_vars, qual_vars = prepare_active_dataset(args.input, out_dir)

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
        df_active=df_active,
    )
    logging.info("FAMD analysis complete")


if __name__ == "__main__":
    main()
