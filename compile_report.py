# compile_report.py
import yaml
import pandas as pd
from pathlib import Path
import phase4
import phase4_functions as pf

with open("config.yaml", "r", encoding="utf-8") as fh:
    cfg = yaml.safe_load(fh)

datasets = ["raw", "cleaned_1", "cleaned_3_multi", "cleaned_3_univ"]

figures = {}
metrics = []

for ds in datasets:
    run_cfg = dict(cfg, dataset=ds)
    run_cfg.pop("output_pdf", None)  # avoid old PDF builder
    res = phase4.run_pipeline(run_cfg)
    figures.update({f"{ds}_{name}": fig for name, fig in res["figures"].items()})
    m = res.get("metrics")
    if isinstance(m, pd.DataFrame):
        m = m.reset_index().rename(columns={"index": "method"})
        m["dataset"] = ds
        metrics.append(m)

tables = {"metrics": pd.concat(metrics, ignore_index=True)} if metrics else {}
pf.export_report_to_pdf(figures, tables, cfg["output_pdf"])
