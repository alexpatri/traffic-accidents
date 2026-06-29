"""§8 Correlação entre variáveis numéricas (Pearson).

Inclui `indice_gravidade` para verificar como ele se relaciona com as contagens que o
compõem (mortos, feridos_*) e com variáveis independentes (km, veículos, pessoas).
Interpretação descritiva, não causal.
"""

from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import polars as pl

from src.eda_analytics import config, data

logger = logging.getLogger(__name__)


def correlations(df: pl.DataFrame) -> pl.DataFrame:
    """Calcula a matriz de correlação de Pearson e gera o heatmap."""
    cols = config.CORR_COLS
    corr = df.select(cols).corr()
    logger.info("Matriz de correlação:\n%s", corr)

    matrix = corr.to_numpy()
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(matrix, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)))
    ax.set_yticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=45, ha="right")
    ax.set_yticklabels(cols)
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center",
                    color="black", fontsize=7)
    ax.set_title("Matriz de correlação (Pearson) — numéricas + índice de gravidade")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    data.save_fig(fig, "08_correlacao_numerica")
    return corr
