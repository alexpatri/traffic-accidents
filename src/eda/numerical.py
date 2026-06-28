"""§4 Variáveis numéricas: estatísticas descritivas, histogramas e boxplots.

Responde: há distribuições assimétricas? Existem outliers? Alguma variável merece
transformação futura?
"""

from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import polars as pl

from src.eda import config, data

logger = logging.getLogger(__name__)


def _describe(df: pl.DataFrame) -> pl.DataFrame:
    """Estatísticas descritivas das colunas numéricas + assimetria (skew)."""
    desc = df.select(config.NUMERIC_COLS).describe()
    skew = df.select(
        [pl.col(c).skew().round(2).alias(c) for c in config.NUMERIC_COLS]
    ).row(0, named=True)
    logger.info("Assimetria (skewness): %s", skew)
    return desc


def _hist(df: pl.DataFrame, col: str) -> None:
    values = df[col].drop_nulls().to_list()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(values, bins=50, color=config.BAR_COLOR)
    ax.set_yscale("log")  # cauda longa: escala log no eixo de contagem
    ax.set_title(f"Histograma: {col} (eixo y log)")
    ax.set_xlabel(col)
    ax.set_ylabel("Frequência (log)")
    data.save_fig(fig, f"04_hist_{col}")


def _boxplot_counts(df: pl.DataFrame) -> None:
    """Boxplot conjunto das contagens de vítimas/veículos (mesma escala)."""
    cols = ["pessoas", "mortos", "feridos_leves", "feridos_graves",
            "ilesos", "ignorados", "feridos", "veiculos"]
    series = [df[c].drop_nulls().to_list() for c in cols]
    fig, ax = plt.subplots(figsize=config.FIGSIZE)
    ax.boxplot(series, labels=cols, vert=True, showfliers=True)
    ax.set_title("Boxplots das contagens (vítimas e veículos)")
    ax.set_ylabel("Valor")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    data.save_fig(fig, "04_boxplot_contagens")


def _boxplot_km(df: pl.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.boxplot(df["km"].drop_nulls().to_list(), labels=["km"])
    ax.set_title("Boxplot: km")
    data.save_fig(fig, "04_boxplot_km")


def numerical(df: pl.DataFrame) -> pl.DataFrame:
    """Executa estatísticas e distribuições das variáveis numéricas."""
    desc = _describe(df)
    logger.info("Describe numérico:\n%s", desc)

    for col in config.NUMERIC_COLS:
        _hist(df, col)
    _boxplot_counts(df)
    _boxplot_km(df)
    return desc
