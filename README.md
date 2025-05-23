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
You can control parallel execution with the `n_jobs` option in `config.yaml`. Set it to the number of worker threads to run independent methods concurrently.

Make sure the dependencies listed in `requirements.txt` are installed. The
file pins specific versions of NumPy, pandas and other libraries to avoid
binary incompatibilities. Create a fresh virtual environment and install the
packages with:

```bash
python -m pip install -r requirements.txt
```

If you encounter an error similar to::

    ValueError: numpy.dtype size changed, may indicate binary incompatibility

or::

    ImportError: Detected an incompatible combination of pandas and NumPy

it means the currently installed versions of NumPy and pandas do not match the
pinned versions. Reinstall the dependencies in a fresh virtual environment using
the command above (``python -m pip install -r requirements.txt``) or force a
reinstall with:

```bash
python -m pip install --force-reinstall -r requirements.txt
```

The package `umap-learn` is required for the UMAP functionality. `phate` and
`pacmap` are optional; install them if you want to run the corresponding
analyses. UMAP accepts several parameters in `config.yaml`, including
`n_neighbors`, `min_dist` and the distance `metric` (default `euclidean`).

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

## Fine-tuning FAMD

Use `phase4_fine_tune_famd.py` to run an optimised FAMD analysis. The script relies on the helper used by `phase4_famd_simple.py` and can automatically select the number of axes when `--optimize` is provided.

```bash
python phase4_fine_tune_famd.py --input /path/to/export_everwin.xlsx \
                                --output /path/to/output_dir \
                                --optimize
```

The coordinates, contributions and scree plot are saved in the chosen output directory.

