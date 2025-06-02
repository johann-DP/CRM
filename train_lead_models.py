from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import log_loss, roc_auc_score
import pickle


def train_xgboost_lead(cfg: Dict) -> Tuple[float, float, XGBClassifier]:
    """Train an XGBoost model on the lead scoring dataset.

    Parameters
    ----------
    cfg : dict
        Global configuration dictionary loaded from YAML. It must contain a
        ``lead_scoring`` section with the dataset paths and an ``xgb_params``
        section for the model hyper-parameters.

    Returns
    -------
    Tuple[float, float, XGBClassifier]
        The validation log-loss, the validation AUC and the fitted model.
    """
    lead_cfg = cfg["lead_scoring"]
    out_dir = Path(lead_cfg.get("output_dir", cfg.get("output_dir", ".")))

    # Load datasets produced by ``preprocess_lead_scoring.py``
    X_train = pd.read_csv(out_dir / "X_train.csv")
    y_train = pd.read_csv(out_dir / "y_train.csv").squeeze()
    X_val = pd.read_csv(out_dir / "X_val.csv")
    y_val = pd.read_csv(out_dir / "y_val.csv").squeeze()

    params = cfg["xgb_params"]
    model_xgb = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=params["n_estimators"],
        learning_rate=params["learning_rate"],
        max_depth=params["max_depth"],
        subsample=params["subsample"],
        colsample_bytree=params["colsample_bytree"],
        use_label_encoder=False,
        n_jobs=params["n_jobs"],
    )

    model_xgb.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=params["early_stopping_rounds"],
        verbose=params["verbose"],
    )

    models_dir = Path(cfg.get("output_dir", ".")) / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    with open(models_dir / "lead_xgb.pkl", "wb") as fh:
        pickle.dump(model_xgb, fh)

    probs_val = model_xgb.predict_proba(X_val)[:, 1]
    logloss_val = log_loss(y_val, probs_val)
    auc_val = roc_auc_score(y_val, probs_val)

    return logloss_val, auc_val, model_xgb
