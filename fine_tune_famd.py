#!/usr/bin/env python3
"""Simple fine-tuning script for FAMD using phase4 utilities."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from phase4v2 import run_famd, export_famd_results
from standalone_utils import prepare_active_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine tune FAMD")
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

    famd, inertia, rows, cols, contrib = run_famd(
        df_active,
        quant_vars,
        qual_vars,
        optimize=True,
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
