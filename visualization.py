"""Comparative visualization utilities for dimensionality reduction results."""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from pathlib import Path
import pandas as pd
import seaborn as sns
import numpy as np
from typing import Dict, Any, List, Optional
from sklearn.cluster import KMeans


def plot_correlation_circle(coords: pd.DataFrame, title: str) -> plt.Figure:
    """Return a correlation circle figure for the provided coordinates.

    Parameters
    ----------
    coords : pandas.DataFrame
        DataFrame indexed by variable names with ``F1`` and ``F2`` columns.
    title : str
        Title of the figure.
    """
    fig, ax = plt.subplots(figsize=(12, 6), dpi=200)
    circle = plt.Circle((0, 0), 1, color="grey", fill=False, linestyle="dashed")
    ax.add_patch(circle)
    ax.axhline(0, color="grey", lw=0.5)
    ax.axvline(0, color="grey", lw=0.5)
    for var in coords.index:
        x, y = coords.loc[var, ["F1", "F2"]]
        ax.arrow(0, 0, x, y, head_width=0.02, length_includes_head=True, color="black")
        offset_x = x * 1.15 + 0.03 * np.sign(x)
        offset_y = y * 1.15 + 0.03 * np.sign(y)
        ax.text(offset_x, offset_y, str(var), fontsize=8, ha="center", va="center")
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_xlabel("F1")
    ax.set_ylabel("F2")
    ax.set_title(title)
    ax.set_aspect("equal")
    fig.tight_layout()
    return fig


def _choose_color_var(df: pd.DataFrame, qual_vars: List[str]) -> Optional[str]:
    """Return a qualitative variable available in ``df`` to colour scatter plots."""
    preferred = [
        "Statut production",
        "Statut commercial",
        "Type opportunité",
    ]
    for col in preferred:
        if col in df.columns:
            return col
    for col in qual_vars:
        if col in df.columns:
            return col
    return None


def plot_scatter_2d(
    emb_df: pd.DataFrame, df_active: pd.DataFrame, color_var: Optional[str], title: str
) -> plt.Figure:
    """Return a 2D scatter plot figure coloured by ``color_var``."""
    fig, ax = plt.subplots(figsize=(12, 6), dpi=200)
    if color_var is None or color_var not in df_active.columns:
        ax.scatter(emb_df.iloc[:, 0], emb_df.iloc[:, 1], s=10, alpha=0.7)
    else:
        cats = df_active.loc[emb_df.index, color_var].astype("category")
        palette = sns.color_palette("tab10", len(cats.cat.categories))
        for cat, color in zip(cats.cat.categories, palette):
            mask = cats == cat
            ax.scatter(
                emb_df.loc[mask, emb_df.columns[0]],
                emb_df.loc[mask, emb_df.columns[1]],
                s=10,
                alpha=0.7,
                color=color,
                label=str(cat),
            )
        ax.legend(title=color_var, bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.set_xlabel(emb_df.columns[0])
    ax.set_ylabel(emb_df.columns[1])
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_scatter_3d(
    emb_df: pd.DataFrame, df_active: pd.DataFrame, color_var: Optional[str], title: str
) -> plt.Figure:
    """Return a 3D scatter plot figure coloured by ``color_var``."""
    fig = plt.figure(figsize=(12, 6), dpi=200)
    ax = fig.add_subplot(111, projection="3d")
    if color_var is None or color_var not in df_active.columns:
        ax.scatter(emb_df.iloc[:, 0], emb_df.iloc[:, 1], emb_df.iloc[:, 2], s=10, alpha=0.7)
    else:
        cats = df_active.loc[emb_df.index, color_var].astype("category")
        palette = sns.color_palette("tab10", len(cats.cat.categories))
        for cat, color in zip(cats.cat.categories, palette):
            mask = cats == cat
            ax.scatter(
                emb_df.loc[mask, emb_df.columns[0]],
                emb_df.loc[mask, emb_df.columns[1]],
                emb_df.loc[mask, emb_df.columns[2]],
                s=10,
                alpha=0.7,
                color=color,
                label=str(cat),
            )
        ax.legend(title=color_var, bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.set_xlabel(emb_df.columns[0])
    ax.set_ylabel(emb_df.columns[1])
    ax.set_zlabel(emb_df.columns[2])
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_cluster_scatter(
    emb_df: pd.DataFrame, labels: np.ndarray, title: str
) -> plt.Figure:
    """Return a 2D scatter plot coloured by K-Means clusters.

    Parameters
    ----------
    emb_df : pandas.DataFrame
        Embedding coordinates with at least two columns.
    labels : array-like
        Cluster labels for each observation.
    title : str
        Title of the figure.
    """
    fig, ax = plt.subplots(figsize=(12, 6), dpi=200)
    unique = np.unique(labels)
    try:
        cmap = matplotlib.colormaps.get_cmap("tab10")
    except AttributeError:  # Matplotlib < 3.6
        cmap = matplotlib.cm.get_cmap("tab10")
    n_colors = cmap.N if hasattr(cmap, "N") else len(unique)
    for i, lab in enumerate(unique):
        mask = labels == lab
        ax.scatter(
            emb_df.loc[mask, emb_df.columns[0]],
            emb_df.loc[mask, emb_df.columns[1]],
            s=10,
            alpha=0.7,
            color=cmap(i % n_colors),
            label=str(lab),
        )
        centroid = emb_df.loc[mask, emb_df.columns[:2]].mean().values
        ax.scatter(
            centroid[0],
            centroid[1],
            marker="x",
            s=60,
            color=cmap(i % n_colors),
        )
    ax.legend(title="cluster", bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.set_xlabel(emb_df.columns[0])
    ax.set_ylabel(emb_df.columns[1])
    ax.set_title(title)
    fig.tight_layout()
    return fig


def _extract_quant_coords(coords: pd.DataFrame, quant_vars: List[str]) -> pd.DataFrame:
    """Extract F1/F2 coordinates for quantitative variables if available."""
    cols = [c for c in ["F1", "F2"] if c in coords.columns]
    if len(cols) < 2:
        # fall back to the first available columns
        extra = [c for c in coords.columns if c not in cols][: 2 - len(cols)]
        cols.extend(extra)
    if len(cols) < 2:
        return pd.DataFrame(columns=["F1", "F2"])
    subset = coords.loc[[v for v in quant_vars if v in coords.index], cols]
    subset = subset.rename(columns={cols[0]: "F1", cols[1]: "F2"})
    return subset


def plot_scree(inertia: pd.Series, title: str) -> plt.Figure:
    """Return a scree plot showing variance explained by each component."""
    axes = range(1, len(inertia) + 1)
    fig, ax = plt.subplots(figsize=(12, 6), dpi=200)
    ax.bar(axes, inertia.values * 100, edgecolor="black")
    ax.plot(axes, np.cumsum(inertia.values) * 100, "-o", color="orange")
    ax.set_xlabel("Composante")
    ax.set_ylabel("% Variance expliquée")
    ax.set_title(title)
    ax.set_xticks(list(axes))
    fig.tight_layout()
    return fig


def plot_famd_contributions(contrib: pd.DataFrame, n: int = 10) -> plt.Figure:
    """Return a bar plot of variable contributions to F1 and F2."""
    if not {"F1", "F2"}.issubset(contrib.columns):
        cols = contrib.columns[:2]
        contrib = contrib.rename(columns={cols[0]: "F1", cols[1]: "F2"})
    grouped: Dict[str, pd.Series] = {}
    for idx in contrib.index:
        var = idx.split("__", 1)[0]
        grouped.setdefault(var, pd.Series(dtype=float))
        grouped[var] = grouped[var].add(contrib.loc[idx, ["F1", "F2"]], fill_value=0)
    df = pd.DataFrame(grouped).T.fillna(0)
    df = df.sort_values(df.sum(axis=1).name if df.columns.size > 2 else 0, ascending=False)
    df = df.iloc[:n]
    fig, ax = plt.subplots(figsize=(12, 6), dpi=200)
    df[["F1", "F2"]].plot(kind="bar", stacked=True, ax=ax)
    ax.set_ylabel("% Contribution")
    ax.set_title("Contribution des variables à F1/F2 – FAMD")
    ax.legend(title="Axe")
    fig.tight_layout()
    return fig


def generate_figures(
    factor_results: Dict[str, Dict[str, Any]],
    nonlin_results: Dict[str, Dict[str, Any]],
    df_active: pd.DataFrame,
    quant_vars: List[str],
    qual_vars: List[str],
    output_dir: Optional[Path] = None,
    *,
    cluster_k: int = 3,
) -> Dict[str, plt.Figure]:
    """Generate and optionally save comparative visualization figures.

    Parameters
    ----------
    output_dir : Path or None, optional
        Directory where figures will be saved.
    cluster_k : int, default ``3``
        Number of K-Means clusters for the additional scatter plots.
    """
    color_var = _choose_color_var(df_active, qual_vars)
    figures: Dict[str, plt.Figure] = {}
    first_3d_done = False
    out = Path(output_dir) if output_dir is not None else None

    def _save(fig: plt.Figure, method: str, name: str) -> None:
        if out is None:
            return
        sub = out / method.lower()
        sub.mkdir(parents=True, exist_ok=True)
        fig.savefig(sub / f"{name}.png")

    for method, res in factor_results.items():
        emb = res.get("embeddings")
        if isinstance(emb, pd.DataFrame) and emb.shape[1] >= 2:
            fig = plot_scatter_2d(
                emb.iloc[:, :2],
                df_active,
                color_var,
                f"Projection des affaires – {method.upper()}",
            )
            figures[f"{method}_scatter_2d"] = fig
            _save(fig, method, f"{method}_scatter_2d")
            km = KMeans(n_clusters=cluster_k, random_state=0)
            labels = km.fit_predict(emb.iloc[:, :2].values)
            title = f"Projection {method.upper()} – coloration par clusters (k={cluster_k})"
            cfig = plot_cluster_scatter(emb.iloc[:, :2], labels, title)
            figures[f"{method}_clusters"] = cfig
            _save(cfig, method, f"{method}_clusters")
            if not first_3d_done and emb.shape[1] >= 3:
                fig3d = plot_scatter_3d(
                    emb.iloc[:, :3],
                    df_active,
                    color_var,
                    f"Projection 3D – {method.upper()}",
                )
                figures[f"{method}_scatter_3d"] = fig3d
                _save(fig3d, method, f"{method}_scatter_3d")
                first_3d_done = True
        coords = res.get("loadings")
        if coords is None:
            coords = res.get("column_coords")
        if isinstance(coords, pd.DataFrame):
            qcoords = _extract_quant_coords(coords, quant_vars)
            if not qcoords.empty:
                var_pc = res.get("inertia")
                pct = float(var_pc.iloc[:2].sum() * 100) if isinstance(var_pc, pd.Series) else float("nan")
                title = f"{method.upper()} – cercle des corrélations (F1–F2)\nVariance {pct:.1f}%"
                fig_corr = plot_correlation_circle(qcoords, title)
                figures[f"{method}_correlation"] = fig_corr
                _save(fig_corr, method, f"{method}_correlation")
        inertia = res.get("inertia")
        if isinstance(inertia, pd.Series) and not inertia.empty:
            fig_scree = plot_scree(inertia, f"Variance expliquée par composante – {method.upper()}")
            figures[f"{method}_scree"] = fig_scree
            _save(fig_scree, method, f"{method}_scree")
        if method == "famd":
            contrib = res.get("contributions")
            if isinstance(contrib, pd.DataFrame) and not contrib.empty:
                fig_contrib = plot_famd_contributions(contrib)
                figures[f"{method}_contributions"] = fig_contrib
                _save(fig_contrib, method, f"{method}_contributions")

    for method, res in nonlin_results.items():
        emb = res.get("embeddings")
        if isinstance(emb, pd.DataFrame) and emb.shape[1] >= 2:
            fig = plot_scatter_2d(
                emb.iloc[:, :2],
                df_active,
                color_var,
                f"Projection des affaires – {method.upper()}",
            )
            figures[f"{method}_scatter_2d"] = fig
            _save(fig, method, f"{method}_scatter_2d")
            km = KMeans(n_clusters=cluster_k, random_state=0)
            labels = km.fit_predict(emb.iloc[:, :2].values)
            title = f"Projection {method.upper()} – coloration par clusters (k={cluster_k})"
            cfig = plot_cluster_scatter(emb.iloc[:, :2], labels, title)
            figures[f"{method}_clusters"] = cfig
            _save(cfig, method, f"{method}_clusters")
            if not first_3d_done and emb.shape[1] >= 3:
                fig3d = plot_scatter_3d(
                    emb.iloc[:, :3],
                    df_active,
                    color_var,
                    f"Projection 3D – {method.upper()}",
                )
                figures[f"{method}_scatter_3d"] = fig3d
                _save(fig3d, method, f"{method}_scatter_3d")
                first_3d_done = True
    return figures

