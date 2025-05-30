import yaml
from pathlib import Path
import phase4_functions as pf


def main():
    with open("config.yaml", "r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)
    datasets = pf.load_datasets(config, ignore_schema=True)
    result = pf.compare_datasets_versions(
        datasets,
        output_dir=Path("rapport_output"),
    )

    figures = {
        f"{ver}_{name}": fig
        for ver, det in result["details"].items()
        for name, fig in det.get("figures", {}).items()
    }
    tables = {"metrics": result["metrics"]}

    pf.export_report_to_pdf(figures, tables, Path("RapportAnalyse.pdf"))


if __name__ == "__main__":
    main()
