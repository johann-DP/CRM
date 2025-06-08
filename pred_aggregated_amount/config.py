"""Shared configuration for revenue forecasting."""

from __future__ import annotations

from pathlib import Path
import yaml


# Path to the repository root
_REPO_ROOT = Path(__file__).resolve().parents[1]
_CFG_FILE = _REPO_ROOT / "config.yaml"

try:
    with open(_CFG_FILE, "r", encoding="utf-8") as fh:
        _CFG = yaml.safe_load(fh) or {}
except FileNotFoundError:  # pragma: no cover - fallback when missing
    _CFG = {}

INPUT_CSV: Path = Path(_CFG.get("input_file_cleaned_3_all", "phase3_cleaned_all.csv"))
OUTPUT_DIR: Path = Path(_CFG.get("output_dir", "output_dir"))

__all__ = ["INPUT_CSV", "OUTPUT_DIR"]
