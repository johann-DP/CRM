#!/usr/bin/env python3
"""Fine-tune PCAmix using utilities from phase4v2."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from phase4v2 import run_pcamix, export_pcamix_results
from standalone_utils import prepare_active_dataset

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine tune PCAmix")
    parser.add_argument("--input", required=True, help="Cleaned multivariate CSV")
    parser.add_argument("--output", required=True, help="Output directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_path = Path(args.input)
    out_dir = Path(args.output)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    if not data_path.exists():
        logging.error("File not found: %s", data_path)
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    df_active, quant_vars, qual_vars = prepare_active_dataset(str(data_path), out_dir)

    model, inertia, rows, cols = run_pcamix(
        df_active,
        quant_vars,
        qual_vars,
        out_dir,
        optimize=True,
    )

    export_pcamix_results(
        model,
        inertia,
        rows,
        cols,
        out_dir,
        quant_vars,
        qual_vars,
        df_active=df_active,
    )
    logging.info("PCAmix fine-tuning complete")


if __name__ == "__main__":
    main()
