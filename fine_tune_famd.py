#!/usr/bin/env python3
"""Simple fine-tuning script for FAMD using phase4 utilities."""

from __future__ import annotations

import logging
from pathlib import Path

from phase4v2 import run_famd, export_famd_results
from standalone_utils import prepare_active_dataset

DATA_PATH = Path("phase3_output/phase3_cleaned_multivariate.csv")
OUTPUT_DIR = Path("phase4_output/fine_tuning_famd")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    if not DATA_PATH.exists():
        logging.error("File not found: %s", DATA_PATH)
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df_active, quant_vars, qual_vars = prepare_active_dataset(str(DATA_PATH), OUTPUT_DIR)

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
        OUTPUT_DIR,
        df_active=df_active,
    )
    logging.info("FAMD fine-tuning complete")


if __name__ == "__main__":
    main()
