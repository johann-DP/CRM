import yaml
from pathlib import Path
import phase4_functions as pf


def main():
    with open("config.yaml", "r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)
    datasets = pf.load_datasets(config, ignore_schema=True)
    result = pf.compare_datasets_versions(datasets, output_dir=Path("rapport_output"))
    pf.build_pdf_report(Path("rapport_output"), Path("RapportAnalyse.pdf"), list(datasets))


if __name__ == "__main__":
    main()
