"""Visualizações da etapa de ML — salvas em `outputs/ml/figures/`."""

from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from src.ml import config, data
from src.ml.evaluate import ComboResult

logger = logging.getLogger(__name__)


def plot_class_distribution(dist: pl.DataFrame, name: str, titulo: str) -> None:
    """Barras da distribuição das 4 classes do alvo."""
    fig, ax = plt.subplots(figsize=config.FIGSIZE)
    ax.bar(dist.get_column("classe_risco").to_list(),
           dist.get_column("pct").to_numpy(), color=config.CLASSE_CORES)
    ax.set_ylabel("% dos trechos")
    ax.set_title(titulo)
    ax.grid(True, axis="y", alpha=0.3)
    data.save_fig(fig, name)


def plot_confusion(cm: np.ndarray, name: str, titulo: str) -> None:
    """Matriz de confusão (contagens) com anotações."""
    labels = config.CLASSE_LABELS
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Previsto")
    ax.set_ylabel("Real")
    ax.set_title(titulo)
    thr = cm.max() / 2 if cm.max() else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thr else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    data.save_fig(fig, name)


def plot_feature_importance(
    importance: list[tuple[str, float]], name: str, titulo: str, top: int = 15
) -> None:
    """Barras horizontais das features mais importantes."""
    itens = importance[:top][::-1]
    if not itens:
        return
    nomes = [n for n, _ in itens]
    vals = [v for _, v in itens]
    fig, ax = plt.subplots(figsize=config.FIGSIZE)
    ax.barh(nomes, vals, color=config.BAR_COLOR)
    ax.set_xlabel("Importância")
    ax.set_title(titulo)
    ax.grid(True, axis="x", alpha=0.3)
    data.save_fig(fig, name)


def plot_variant_comparison(combos: list[ComboResult], name: str, titulo: str) -> None:
    """Barras agrupadas de CV F1 macro por (variante × algoritmo)."""
    variantes = config.VARIANTES
    algos = config.ALGOS
    x = np.arange(len(variantes))
    w = 0.8 / len(algos)
    fig, ax = plt.subplots(figsize=config.FIGSIZE)
    for i, algo in enumerate(algos):
        vals = [
            next((c.cv_f1_macro for c in combos
                  if c.variante == v and c.algo == algo), 0.0)
            for v in variantes
        ]
        ax.bar(x + i * w, vals, w, label=algo)
    ax.axhline(1 / len(config.CLASSE_LABELS), color="grey", linestyle="--",
               alpha=0.7, label="acaso (1/4)")
    ax.set_xticks(x + w * (len(algos) - 1) / 2, variantes)
    ax.set_ylabel("CV F1 macro")
    ax.set_title(titulo)
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    data.save_fig(fig, name)
