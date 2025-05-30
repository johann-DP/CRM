import yaml
from pathlib import Path
import phase4_functions as pf


def main():
    with open("config.yaml", "r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)
    datasets = pf.load_datasets(config, ignore_schema=True)
    out_dir = Path("rapport_output")
    out_dir.mkdir(parents=True, exist_ok=True)
    result = pf.compare_datasets_versions(datasets, output_dir=out_dir)

    all_figs = {}
    for ds, info in result["details"].items():
        for name, fig in info.get("figures", {}).items():
            all_figs[f"{ds}_{name}"] = fig

    tables = {"metrics": result["metrics"]}
    pf.export_report_to_pdf(all_figs, tables, Path("RapportAnalyse.pdf"))


if __name__ == "__main__":
    main()
