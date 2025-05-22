# CRM Project

This repository contains scripts for various phases of data analysis.

## Running `phase4v2.py`

The script requires a configuration file in YAML (or JSON) format. A template
is provided in `config.yaml`. Copy or modify it to suit your dataset and run:

```bash
python phase4v2.py --config config.yaml
```

Set `optimize_params: true` in the configuration to automatically tune the main
hyperparameters of each dimensionality reduction method (number of components
for FAMD/MFA/PCAmix, neighbors and distance for UMAP, perplexity for t-SNE).
When disabled, the script uses the provided values or sensible defaults.

Make sure the dependencies listed in `requirements.txt` are installed. The
package `umap-learn` is required for the UMAP functionality. Optional methods
require extra packages: `pacmap` for PaCMAP and `phate` for PHATE.

### UMAP warnings

UMAP emits a warning when `random_state` is provided while using multiple
threads. The provided configuration sets `n_jobs: 1` whenever a seed is used to
avoid this warning and keep results reproducible. You can remove the seed if you
prefer parallelism over determinism.

## Running `phase4_famd.py`

For a streamlined FAMD-only pipeline without a configuration file, you can use
`phase4_famd.py`. Provide the input Excel file and an output directory:

```bash
python phase4_famd.py --input "D:\DATAPREDICT\DATAPREDICT 2024\Missions\Digora\export_everwin (19).xlsx" \
                       --output "D:\DATAPREDICT\DATAPREDICT 2024\Missions\Digora\phase4_output"
```

This mirrors the paths used when the original script was created and will
generate figures and CSV results inside the specified output folder.
