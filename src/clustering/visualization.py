"""Visualizações da clusterização: cotovelo, silhouette, PCA 2D e distribuições.

Todas as figuras são salvas em `outputs/clustering/figures/` via `data.save_fig`.
O PCA é usado APENAS para visualização (projeta a matriz que o KMeans enxergou em
2D); nunca como espaço de entrada do algoritmo.
"""

from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.figure import Figure
from sklearn.decomposition import PCA

from src.clustering import config, data

logger = logging.getLogger(__name__)


def plot_elbow(ks: list[int], inertias: list[float], name: str, titulo: str) -> None:
    """Gráfico do Método do Cotovelo: SSE (inertia) × K."""
    fig, ax = plt.subplots(figsize=config.FIGSIZE)
    ax.plot(ks, inertias, marker="o", color=config.BAR_COLOR)
    ax.set_xlabel("Número de clusters (K)")
    ax.set_ylabel("Inertia (SSE intra-cluster)")
    ax.set_title(titulo)
    ax.set_xticks(ks)
    ax.grid(True, alpha=0.3)
    data.save_fig(fig, name)


def plot_silhouette(
    ks: list[int], silhouettes: list[float], name: str, titulo: str
) -> None:
    """Gráfico do Silhouette Score médio × K, destacando o pico."""
    fig, ax = plt.subplots(figsize=config.FIGSIZE)
    ax.plot(ks, silhouettes, marker="o", color="#b2182b")
    melhor = ks[int(np.argmax(silhouettes))]
    ax.axvline(melhor, color="grey", linestyle="--", alpha=0.7,
               label=f"Pico em K={melhor}")
    ax.set_xlabel("Número de clusters (K)")
    ax.set_ylabel("Silhouette Score médio")
    ax.set_title(titulo)
    ax.set_xticks(ks)
    ax.grid(True, alpha=0.3)
    ax.legend()
    data.save_fig(fig, name)


def plot_pca(X: np.ndarray, labels: np.ndarray, name: str, titulo: str) -> float:
    """Projeta `X` em 2D via PCA e colore os pontos pelos clusters.

    Returns:
        A fração de variância explicada por PC1+PC2 (registrada no relatório).
    """
    pca = PCA(n_components=2, random_state=config.RANDOM_STATE)
    coords = pca.fit_transform(X)
    var_ratio = float(pca.explained_variance_ratio_[:2].sum())

    fig, ax = plt.subplots(figsize=config.FIGSIZE)
    cmap = plt.get_cmap(config.CLUSTER_CMAP)
    for c in sorted(set(labels.tolist())):
        mask = labels == c
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            s=8, alpha=0.4, color=cmap(c % 10), label=f"Cluster {c}",
        )
    # Centróides projetados pelo mesmo PCA.
    centroids = np.array([X[labels == c].mean(axis=0) for c in sorted(set(labels.tolist()))])
    cent2d = pca.transform(centroids)
    ax.scatter(cent2d[:, 0], cent2d[:, 1], s=180, marker="X",
               c="black", edgecolors="white", zorder=5, label="Centróides")

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    ax.set_title(f"{titulo}\n(variância 2D explicada: {var_ratio:.1%})")
    ax.legend(markerscale=2, fontsize=8)
    data.save_fig(fig, name)
    return var_ratio


def plot_box_por_cluster(
    df: pl.DataFrame, col: str, cluster_col: str, name: str, titulo: str
) -> None:
    """Boxplot de uma variável numérica por cluster."""
    clusters = sorted(df.get_column(cluster_col).unique().to_list())
    dados = [
        df.filter(pl.col(cluster_col) == c).get_column(col).to_numpy()
        for c in clusters
    ]
    fig, ax = plt.subplots(figsize=config.FIGSIZE)
    ax.boxplot(dados, labels=[f"C{c}" for c in clusters], showfliers=False)
    ax.set_xlabel("Cluster")
    ax.set_ylabel(col)
    ax.set_title(titulo)
    ax.grid(True, axis="y", alpha=0.3)
    data.save_fig(fig, name)


def plot_barras_proporcao(
    matriz: pl.DataFrame, index_col: str, categorias: list[str], name: str, titulo: str
) -> None:
    """Barras empilhadas 100% de uma distribuição categórica por cluster.

    `matriz` deve ter uma coluna `index_col` (cluster) e uma coluna por categoria
    com a proporção (somando ~1 por linha).
    """
    clusters = matriz.get_column(index_col).to_list()
    fig, ax = plt.subplots(figsize=config.FIGSIZE)
    cmap = plt.get_cmap(config.CLUSTER_CMAP)
    base = np.zeros(len(clusters))
    for i, cat in enumerate(categorias):
        vals = matriz.get_column(cat).to_numpy()
        ax.bar([f"C{c}" for c in clusters], vals, bottom=base,
               label=cat, color=cmap(i % 10))
        base = base + vals
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Proporção")
    ax.set_title(titulo)
    ax.legend(fontsize=8, bbox_to_anchor=(1.02, 1), loc="upper left")
    data.save_fig(fig, name)
