"""Generate comparative visualisations for dimensionality reduction methods.

This module implements ``generate_figures`` used in phase4v3. It creates
correlation circles for factorial methods and 2D/3D scatter plots for all
methods. Figures are returned in a dictionary and can optionally be saved by
the caller.
"""

from __future__ import annotations

from typing import Dict, Any, List, Optional

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 -- imported for 3D plots
import seaborn as sns


def _get_var_coords(name: str, res: Dict[str, Any], df: pd.DataFrame,
                    quant_vars: List[str], qual_vars: List[str]) -> Optional[pd.DataFrame]:
    """Extract variable coordinates for the correlation circle if available."""
    model = res.get("model")
    if model is None:
        return None

    coords = res.get("var_coords")
    if isinstance(coords, pd.DataFrame):
        return coords

    # Prince models
    for attr in ["column_correlations_", "column_coordinates_"]:
        if hasattr(model, attr):
            arr = getattr(model, attr)
            if isinstance(arr, pd.DataFrame):
                return arr
            if isinstance(arr, np.ndarray):
                cols = [f"F{i+1}" for i in range(arr.shape[1])]
                idx = quant_vars + qual_vars
                idx = idx[: arr.shape[0]]
                return pd.DataFrame(arr, index=idx, columns=cols)
    if hasattr(model, "column_coordinates"):
        try:
            tmp = model.column_coordinates(df[quant_vars + qual_vars])
            if isinstance(tmp, pd.DataFrame):
                return tmp
        except Exception:
            pass

    # scikit-learn PCA
    if name.lower() == "pca" and hasattr(model, "components_"):
        try:
            loadings = model.components_.T * np.sqrt(model.explained_variance_)
            cols = [f"F{i+1}" for i in range(loadings.shape[1])]
            return pd.DataFrame(loadings, index=quant_vars, columns=cols)
        except Exception:
            return None
    return None


def _plot_corr_circle(coords: pd.DataFrame, inertia: Optional[pd.Series], title: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 6))
    circle = plt.Circle((0, 0), 1, color="grey", fill=False, linestyle="dashed")
    ax.add_patch(circle)
    ax.axhline(0, color="grey", lw=0.5)
    ax.axvline(0, color="grey", lw=0.5)
    for var in coords.index:
        if {"F1", "F2"}.issubset(coords.columns):
            x, y = coords.loc[var, ["F1", "F2"]]
            ax.arrow(0, 0, x, y, head_width=0.02, length_includes_head=True)
            ax.text(x * 1.1, y * 1.1, str(var), ha="center", va="center", fontsize=8)
    pct = ""
    if inertia is not None and len(inertia) >= 2:
        pct = f" ({inertia.iloc[:2].sum() * 100:.1f}% var)"
    ax.set_title(title + pct)
    ax.set_xlabel("F1")
    ax.set_ylabel("F2")
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_aspect("equal")
    fig.tight_layout()
    return fig


def _scatter_2d(emb: pd.DataFrame, df: pd.DataFrame, color_var: Optional[str], title: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 6))
    if color_var:
        cats = df.loc[emb.index, color_var].astype("category")
        palette = sns.color_palette("tab10", len(cats.cat.categories))
        for cat, color in zip(cats.cat.categories, palette):
            mask = cats == cat
            ax.scatter(emb.loc[mask, emb.columns[0]], emb.loc[mask, emb.columns[1]],
                       s=10, alpha=0.7, color=color, label=str(cat))
        ax.legend(title=color_var, bbox_to_anchor=(1.05, 1), loc="upper left")
    else:
        ax.scatter(emb.iloc[:, 0], emb.iloc[:, 1], s=10, alpha=0.7)
    ax.set_xlabel(emb.columns[0])
    ax.set_ylabel(emb.columns[1])
    ax.set_title(title)
    fig.tight_layout()
    return fig


def _scatter_3d(emb: pd.DataFrame, df: pd.DataFrame, color_var: Optional[str], title: str) -> plt.Figure:
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")
    if color_var:
        cats = df.loc[emb.index, color_var].astype("category")
        palette = sns.color_palette("tab10", len(cats.cat.categories))
        for cat, color in zip(cats.cat.categories, palette):
            mask = cats == cat
            ax.scatter(
                emb.loc[mask, emb.columns[0]],
                emb.loc[mask, emb.columns[1]],
                emb.loc[mask, emb.columns[2]],
                s=10,
                alpha=0.7,
                color=color,
                label=str(cat),
            )
        ax.legend(title=color_var, bbox_to_anchor=(1.05, 1), loc="upper left")
    else:
        ax.scatter(emb.iloc[:, 0], emb.iloc[:, 1], emb.iloc[:, 2], s=10, alpha=0.7)
    ax.set_xlabel(emb.columns[0])
    ax.set_ylabel(emb.columns[1])
    ax.set_zlabel(emb.columns[2])
    ax.set_title(title)
    fig.tight_layout()
    return fig


def generate_figures(
    factor_results: Dict[str, Dict[str, Any]],
    nonlin_results: Dict[str, Dict[str, Any]],
    df_active: pd.DataFrame,
    quant_vars: List[str],
    qual_vars: List[str],
) -> Dict[str, plt.Figure]:
    """Produce key comparative visualisations for dimensionality reduction."""

    figures: Dict[str, plt.Figure] = {}
    color_var: Optional[str] = None
    for col in ["Statut production", "Type opportunité"]:
        if col in df_active.columns:
            color_var = col
            break

    # Factorial methods
    for name, res in factor_results.items():
        emb = res.get("embeddings")
        if isinstance(emb, pd.DataFrame) and emb.shape[1] >= 2:
            figures[f"{name}_2D"] = _scatter_2d(emb.iloc[:, :2], df_active, color_var,
                                                f"Projection des affaires – {name}")
        if isinstance(emb, pd.DataFrame) and emb.shape[1] >= 3:
            figures[f"{name}_3D"] = _scatter_3d(emb.iloc[:, :3], df_active, color_var,
                                                f"Projection des affaires – {name}")
        coords = _get_var_coords(name, res, df_active, quant_vars, qual_vars)
        inertia = res.get("inertia")
        if isinstance(coords, pd.DataFrame) and {"F1", "F2"}.issubset(coords.columns):
            figures[f"{name}_corr"] = _plot_corr_circle(coords, inertia, name)

    # Non-linear methods
    for name, res in nonlin_results.items():
        emb = res.get("embeddings")
        if not isinstance(emb, pd.DataFrame):
            continue
        label = name.upper()
        if emb.shape[1] >= 2:
            figures[f"{label}_2D"] = _scatter_2d(emb.iloc[:, :2], df_active, color_var,
                                                 f"Projection des affaires – {label}")
        if emb.shape[1] >= 3:
            figures[f"{label}_3D"] = _scatter_3d(emb.iloc[:, :3], df_active, color_var,
                                                 f"Projection des affaires – {label}")

    return figures

