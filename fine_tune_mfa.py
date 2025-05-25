import argparse
import json
import logging
from pathlib import Path
from time import time
import yaml
import pandas as pd
from itertools import product
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score

from standalone_utils import prepare_active_dataset
from phase4v2 import (
    run_mfa,
    export_mfa_results,
    load_data,
    prepare_data,
)

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def load_config(path: Path) -> dict:
    if path.suffix.lower() in {'.yaml', '.yml'}:
        return yaml.safe_load(path.read_text())
    return json.loads(path.read_text())


def evaluate_embedding(emb: pd.DataFrame, k_range=range(2, 7)) -> tuple[float, float]:
    best_sil = -1.0
    best_ch = -1.0
    for k in k_range:
        labels = KMeans(n_clusters=k, random_state=0).fit_predict(emb.values)
        sil = silhouette_score(emb.values, labels)
        ch = calinski_harabasz_score(emb.values, labels)
        if sil > best_sil:
            best_sil = sil
        if ch > best_ch:
            best_ch = ch
    return best_sil, best_ch


def main() -> None:
    p = argparse.ArgumentParser(description="Fine tune MFA")
    p.add_argument("--config", help="YAML or JSON config file")
    p.add_argument("--input", help="Cleaned multivariate CSV")
    p.add_argument("--output", help="Output directory")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    cfg = {}
    if args.config:
        cfg = load_config(Path(args.config))

    if args.input:
        cfg["input_file"] = args.input
    if args.output:
        cfg["output_dir"] = args.output

    if "input_file" not in cfg or "output_dir" not in cfg:
        p.error("Provide --input and --output or a config file with those fields")

    input_file = cfg["input_file"]
    out_dir = Path(cfg.get("output_dir", "phase4_output/fine_tuning_mfa"))
    group_defs = cfg.get("group_defs")
    mfa_cfg = cfg.get("mfa_params", {})
    n_components_range = list(range(mfa_cfg.get("min_components", 2), mfa_cfg.get("max_components", 10) + 1))
    weight_options = mfa_cfg.get("weights", [None])
    n_iter = int(mfa_cfg.get("n_iter", 3))

    df_active, quant_vars, qual_vars = prepare_active_dataset(input_file, out_dir)

    # Keep segmentation columns by loading the full cleaned dataset
    df_full = prepare_data(load_data(input_file))
    df_full = df_full.loc[df_active.index]

    results = []
    best = None
    best_metrics = (-1.0, -1.0)
    for n_comp, weights in product(n_components_range, weight_options):
        start = time()
        model, rows = run_mfa(
            df_active,
            quant_vars,
            qual_vars,
            out_dir / f"mfa_{n_comp}",
            n_components=n_comp,
            groups=group_defs,
            weights=weights,
            n_iter=n_iter,
        )
        runtime = time() - start
        inertia = sum(model.explained_inertia_)
        sil, ch = evaluate_embedding(rows)
        results.append({
            "n_components": n_comp,
            "weights": weights,
            "runtime_s": runtime,
            "cum_inertia": inertia,
            "silhouette": sil,
            "calinski": ch,
        })
        if inertia >= 0.8 and (sil > best_metrics[0] or ch > best_metrics[1]):
            best_metrics = (sil, ch)
            best = (model, rows, n_comp, weights)

    pd.DataFrame(results).to_csv(out_dir / "mfa_grid_search.csv", index=False)

    if best is None:
        logging.warning("No configuration reached 80% inertia; keeping last one")
        model, rows, _, _ = model, rows, n_comp, weights
    else:
        model, rows, n_comp, weights = best

    export_mfa_results(
        model,
        rows,
        out_dir / "best",
        quant_vars,
        qual_vars,
        df_active=df_full,
    )
    best_params = {
        "method": "MFA",
        "params": {"n_components": int(n_comp), "weights": weights},
    }
    with open(out_dir / "best_params.json", "w", encoding="utf-8") as fh:
        json.dump(best_params, fh, indent=2)
    logging.info("Best MFA with %d components", n_comp)


if __name__ == "__main__":
    main()
