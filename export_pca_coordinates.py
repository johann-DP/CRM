import argparse
import logging
from pathlib import Path
from typing import Any, Mapping

import yaml

from phase4.functions import (
    load_datasets,
    prepare_data,
    select_variables,
    handle_missing_values,
    run_pca,
)


logger = logging.getLogger(__name__)


def export_pca_coords(
    config: Mapping[str, Any],
    dataset: str = "raw",
    *,
    n_components: int = 3,
    output: Path = Path("ACP_coordonnees_individus.csv"),
    sep: str = ";",
) -> Path:
    """Export PCA coordinates for ``dataset`` according to ``config``.

    Parameters
    ----------
    config:
        Mapping with dataset paths and options.
    dataset:
        Name of the dataset to process (e.g. ``"raw"``).
    n_components:
        Number of PCA components to keep.
    output:
        Destination CSV file.
    sep:
        Field separator used in the CSV file.

    Returns
    -------
    pathlib.Path
        Path to the written CSV file.
    """
    datasets = load_datasets(config, ignore_schema=bool(config.get("ignore_schema", False)))
    if dataset not in datasets:
        raise KeyError(f"dataset '{dataset}' not found")

    df = datasets[dataset]
    df_prep = prepare_data(df, exclude_lost=bool(config.get("exclude_lost", True)))
    ids = df_prep.get("Code", df_prep.index)

    df_active, quant_vars, qual_vars = select_variables(df_prep)
    df_active = handle_missing_values(df_active, quant_vars, qual_vars)

    if not quant_vars:
        raise ValueError("No quantitative variables available for PCA")

    res = run_pca(df_active, quant_vars, n_components=n_components)
    coords = res["embeddings"].iloc[:, :n_components].copy()
    coords.insert(0, "ID", ids.values)

    output.parent.mkdir(parents=True, exist_ok=True)
    coords.to_csv(output, sep=sep, index=False)
    logger.info("PCA coordinates saved to %s", output)
    return output


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Export PCA coordinates")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    parser.add_argument("--dataset", default="raw", help="Dataset to process")
    parser.add_argument("--components", type=int, default=3, help="Number of PCA components")
    parser.add_argument("--output", default="ACP_coordonnees_individus.csv", help="Output CSV file")
    parser.add_argument("--sep", default=";", help="CSV separator")
    args = parser.parse_args(argv)

    with open(args.config, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    output = export_pca_coords(
        cfg,
        args.dataset,
        n_components=args.components,
        output=Path(args.output),
        sep=args.sep,
    )
    print(output)


if __name__ == "__main__":  # pragma: no cover - CLI
    logging.basicConfig(level=logging.INFO)
    main()
