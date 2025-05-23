#!/usr/bin/env python3
"""Fine-tune FAMD using the simple Phase 4 pipeline.

This script is a wrapper around :mod:`phase4_famd_simple`. It loads the
Excel export from Everwin, prepares the active dataset and runs FAMD.
If ``--optimize`` is passed and ``--n_components`` is omitted, the number
of axes is automatically selected based on the explained inertia.
Results are exported using :func:`phase4v2.export_famd_results`.
"""
import argparse
import logging
from pathlib import Path

from standalone_utils import prepare_active_dataset
from phase4v2 import run_famd, export_famd_results


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fine tune FAMD on CRM data")
    p.add_argument("--input", required=True, help="Excel file from Everwin")
    p.add_argument("--output", required=True, help="Output directory")
    p.add_argument(
        "--n_components",
        type=int,
        default=None,
        help="Number of components to keep",
    )
    p.add_argument(
        "--optimize",
        action="store_true",
        help="Automatically select the number of components if not provided",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    df_active, quant_vars, qual_vars = prepare_active_dataset(args.input, out_dir)

    famd, inertia, rows, cols, contrib = run_famd(
        df_active,
        quant_vars,
        qual_vars,
        n_components=args.n_components,
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
    logging.info("FAMD fine-tuning complete")


if __name__ == "__main__":
    main()
