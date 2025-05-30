import yaml
from pathlib import Path
import phase4_functions as pf


def main():
    with open("config.yaml", "r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)
    datasets = pf.load_datasets(config, ignore_schema=True)
    results = pf.compare_datasets_versions(datasets)

    figures = {}
    for ds_name, det in results["details"].items():
        for name, fig in det.get("figures", {}).items():
            figures[f"{ds_name}_{name}"] = fig

    tables = {"metrics": results["metrics"]}
    output_pdf = Path(config.get("output_pdf", "RapportAnalyse.pdf"))
    pf.export_report_to_pdf(figures, tables, output_pdf)


if __name__ == "__main__":
    main()
