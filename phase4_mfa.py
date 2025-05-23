#!/usr/bin/env python3
"""Standalone MFA analysis using functions from phase4v2."""

import argparse
import logging
from pathlib import Path

from phase4v2 import run_mfa, export_mfa_results
from standalone_utils import prepare_active_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MFA only")
    parser.add_argument("--input", required=True, help="Excel file")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--n_components", type=int, default=None, help="Number of components")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    out_dir = Path(args.output) / "MFA"

    df_active, quant_vars, qual_vars = prepare_active_dataset(args.input, out_dir)

    model, rows = run_mfa(
        df_active,
        quant_vars,
        qual_vars,
        out_dir,
        n_components=args.n_components,
    )

    export_mfa_results(model, rows, out_dir, quant_vars, qual_vars, df_active=df_active)
    logging.info("MFA analysis complete")


if __name__ == "__main__":
    main()
