from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict


def _parse_value(value: str) -> Any:
    v = value.strip()
    if v.lower() in {"", "null"}:
        return None
    if v.lower() == "true":
        return True
    if v.lower() == "false":
        return False
    if (v.startswith("\"") and v.endswith("\"")) or (v.startswith("'") and v.endswith("'")):
        return v[1:-1]
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        pass
    try:
        return json.loads(v)
    except Exception:
        return v


def load_best_params(csv_path: Path | str = Path(__file__).with_name("best_params.csv")) -> Dict[str, Dict[str, Any]]:
    csv_path = Path(csv_path)
    params: Dict[str, Dict[str, Any]] = {}
    if not csv_path.exists():
        return params
    with open(csv_path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            method = row["method"].strip().upper()
            param = row["param"].strip()
            value = _parse_value(row["value"])
            params.setdefault(method, {})[param] = value
    return params


BEST_PARAMS = load_best_params()
