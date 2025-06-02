"""Training utilities for lead scoring models."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

try:
    from catboost import CatBoostClassifier
except Exception:  # pragma: no cover - optional dependency safeguard
    CatBoostClassifier = None


def train_catboost_lead(cfg: Dict[str, Dict]) -> Tuple[CatBoostClassifier, float, float]:
    """Train a CatBoost model for lead scoring and return it with validation metrics."""
    if CatBoostClassifier is None:  # pragma: no cover - catboost missing
        raise ImportError("catboost is required for lead scoring")

    lead_cfg = cfg["lead_scoring"]
    data_dir = Path(lead_cfg.get("output_dir", cfg.get("output_dir", ".")))

    X_train = pd.read_csv(data_dir / "X_train.csv")
    y_train = pd.read_csv(data_dir / "y_train.csv").squeeze()
    X_val = pd.read_csv(data_dir / "X_val.csv")
    y_val = pd.read_csv(data_dir / "y_val.csv").squeeze()

    cat_features = lead_cfg["cat_features"]
    params = cfg["cat_params"]

    model_cat = CatBoostClassifier(
        iterations=params["iterations"],
        learning_rate=params["learning_rate"],
        depth=params["depth"],
        eval_metric="Logloss",
        loss_function="Logloss",
        task_type=params.get("task_type", "CPU"),
        thread_count=params["thread_count"],
        random_seed=params["random_seed"],
    )

    model_cat.fit(
        X_train,
        y_train,
        cat_features=cat_features,
        eval_set=(X_val, y_val),
        early_stopping_rounds=params["early_stopping_rounds"],
        verbose=params["verbose"],
    )

    models_dir = Path(cfg.get("output_dir", ".")) / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    model_cat.save_model(models_dir / "lead_catboost.cbm")

    proba_val = model_cat.predict_proba(X_val)[:, 1]
    logloss_val = log_loss(y_val, proba_val)
    auc_val = roc_auc_score(y_val, proba_val)

    return model_cat, logloss_val, auc_val


__all__ = ["train_catboost_lead"]
