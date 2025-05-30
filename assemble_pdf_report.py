import argparse
from pathlib import Path
import yaml
import pandas as pd
import phase4_functions as pf

DEFAULT_DATASETS = ["raw", "cleaned_1", "cleaned_3_multi", "cleaned_3_univ"]


def gather_figures(base_dir: Path, datasets: list[str]) -> dict[str, Path]:
    figures: dict[str, Path] = {}
    for ds in datasets:
        ds_dir = base_dir / ds
        if not ds_dir.exists():
            continue
        for img in ds_dir.rglob("*.png"):
            figures[f"{ds}_{img.stem}"] = img
    for img in base_dir.glob("*.png"):
        figures[img.stem] = img
    return figures


def gather_metrics(base_dir: Path, datasets: list[str]) -> pd.DataFrame | None:
    frames: list[pd.DataFrame] = []
    for ds in datasets:
        csv_path = base_dir / ds / "metrics.csv"
        if not csv_path.exists():
            continue
        try:
            df = pd.read_csv(csv_path, index_col=0)
        except Exception:
            df = pd.read_csv(csv_path)
        if "method" not in df.columns:
            df = df.rename(columns={df.columns[0]: "method"})
        df["dataset"] = ds
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else None


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Assemble a Phase 4 PDF report from pre-generated figures"
    )
    parser.add_argument("--config", default="config.yaml", help="Config file")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=DEFAULT_DATASETS,
        help="Datasets processed by phase4.py",
    )
    args = parser.parse_args(argv)

    with open(args.config, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    output_dir = Path(cfg.get("output_dir", "phase4_output"))
    output_pdf = Path(cfg.get("output_pdf", output_dir / "phase4_report.pdf"))

    figures = gather_figures(output_dir, args.datasets)
    metrics = gather_metrics(output_dir, args.datasets)

    tables = {"metrics": pf.format_metrics_table(metrics)} if metrics is not None else {}

    pf.export_report_to_pdf(figures, tables, output_pdf)


if __name__ == "__main__":
    main()
