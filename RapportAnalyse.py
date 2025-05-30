import yaml
from pathlib import Path
import phase4_functions as pf
from phase4 import build_pdf_report


def main():
    with open("config.yaml", "r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)
    datasets = pf.load_datasets(config, ignore_schema=True)
    result = pf.compare_datasets_versions(datasets, output_dir=Path("rapport_output"))

    figures = {}
    for ds, info in result["details"].items():
        for name, fig in info.get("figures", {}).items():
            figures[f"{ds}_{name}"] = fig

    tables = {"metrics": result["metrics"]}
    pf.export_report_to_pdf(figures, tables, Path("RapportAnalyse.pdf"))


if __name__ == "__main__":
    main()
