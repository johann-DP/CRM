#!/usr/bin/env python3
"""Fine-tune PCAmix using utilities from phase4v2."""

from __future__ import annotations

import logging
from pathlib import Path

from phase4v2 import run_pcamix, export_pcamix_results
from standalone_utils import prepare_active_dataset

DATA_PATH = Path("/mnt/data/phase3_cleaned_multivariate.csv")
OUTPUT_DIR = Path("/mnt/data/phase4_output/fine_tuning_pcamix")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    if not DATA_PATH.exists():
        logging.error("File not found: %s", DATA_PATH)
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df_active, quant_vars, qual_vars = prepare_active_dataset(str(DATA_PATH), OUTPUT_DIR)

    model, inertia, rows, cols = run_pcamix(
        df_active,
        quant_vars,
        qual_vars,
        OUTPUT_DIR,
        optimize=True,
    )

    export_pcamix_results(
        model,
        inertia,
        rows,
        cols,
        OUTPUT_DIR,
        quant_vars,
        qual_vars,
        df_active=df_active,
    )
    logging.info("PCAmix fine-tuning complete")


if __name__ == "__main__":
    main()
