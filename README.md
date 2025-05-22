# CRM Project

This repository contains scripts for various phases of data analysis.

## Running `phase4v2.py`

The script requires a configuration file in YAML (or JSON) format. A template
is provided in `config.yaml`. Copy or modify it to suit your dataset and run:

```bash
python phase4v2.py --config config.yaml
```

Make sure the dependencies listed in `requirements.txt` are installed. The
packages `umap-learn` and `PyYAML` are required for the UMAP functionality and
for reading the YAML configuration. If `PyYAML` is not available, you may
provide the configuration as a JSON file instead.
