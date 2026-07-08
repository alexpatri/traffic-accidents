"""§3 Variáveis categóricas: frequência absoluta/relativa + gráficos de barras.

Responde: quais categorias predominam? Existem categorias pouco representativas?
Há necessidade futura de agrupamento?
"""

from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import polars as pl

from src.analysis.eda import config, data

logger = logging.getLogger(__name__)


def _freq_table(df: pl.DataFrame, col: str) -> pl.DataFrame:
    """Frequência absoluta e relativa (%) ordenada por contagem."""
    return (
        df[col]
        .value_counts(sort=True)
        .with_columns((pl.col("count") / df.height * 100).round(2).alias("pct"))
    )


def _bar(table: pl.DataFrame, col: str, top_n: int | None) -> None:
    """Gráfico de barras horizontais a partir de uma tabela de frequência."""
    plot = table.head(top_n) if top_n else table
    labels = plot[col].to_list()[::-1]
    values = plot["count"].to_list()[::-1]

    fig, ax = plt.subplots(figsize=config.FIGSIZE)
    ax.barh(labels, values, color=config.BAR_COLOR)
    suffix = f" (top {top_n})" if top_n and table.height > top_n else ""
    ax.set_title(f"Frequência: {col}{suffix}")
    ax.set_xlabel("Registros")
    data.save_fig(fig, f"03_cat_{col}")


def categorical(df: pl.DataFrame) -> dict[str, pl.DataFrame]:
    """Gera tabelas de frequência e barras para todas as categóricas relevantes."""
    tables: dict[str, pl.DataFrame] = {}

    for col in config.LOW_CARD_CATS:
        t = _freq_table(df, col)
        tables[col] = t
        logger.info("[%s] frequências:\n%s", col, t)
        _bar(t, col, top_n=None)

    for col in config.HIGH_CARD_CATS:
        t = _freq_table(df, col)
        tables[col] = t
        logger.info(
            "[%s] cardinalidade=%d | top5:\n%s", col, t.height, t.head(5)
        )
        # categorias raras: quantas aparecem <= 5 vezes
        raras = t.filter(pl.col("count") <= 5).height
        logger.info("[%s] categorias com <=5 ocorrências: %d", col, raras)
        _bar(t, col, top_n=config.TOP_N)

    return tables
