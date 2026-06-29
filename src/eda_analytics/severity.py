"""§2 Avaliação do índice de gravidade.

Responde: o índice separa bem os níveis de severidade? Há desbalanceamento entre as
classes? Os pesos produzem uma distribuição coerente? Valida a monotonicidade do
% de acidentes fatais ao longo das classes Baixa -> Crítica.
"""

from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import polars as pl

from src.eda_analytics import config, data

logger = logging.getLogger(__name__)


def _hist(df: pl.DataFrame) -> None:
    """Histograma do índice (eixo y log: cauda longa esperada)."""
    values = df["indice_gravidade"].drop_nulls().to_list()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(values, bins=60, color=config.BAR_COLOR)
    ax.set_yscale("log")
    ax.set_title("Distribuição do índice de gravidade (eixo y log)")
    ax.set_xlabel("indice_gravidade")
    ax.set_ylabel("Frequência (log)")
    data.save_fig(fig, "02_hist_indice")


def _boxplot(df: pl.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.boxplot(df["indice_gravidade"].drop_nulls().to_list(), labels=["indice_gravidade"])
    ax.set_title("Boxplot do índice de gravidade")
    ax.set_ylabel("Valor")
    data.save_fig(fig, "02_boxplot_indice")


def _classes(df: pl.DataFrame) -> pl.DataFrame:
    """Distribuição das classes na ordem ordinal + barra de frequência."""
    cont = df["classe_gravidade"].value_counts()
    ordem = {c: i for i, c in enumerate(config.CLASSE_ORDEM)}
    tab = (
        cont.with_columns(
            pl.col("classe_gravidade").replace_strict(ordem, default=99).alias("ord"),
            (pl.col("count") / df.height * 100).round(2).alias("pct"),
        )
        .sort("ord")
        .drop("ord")
    )
    fig, ax = plt.subplots(figsize=config.FIGSIZE)
    bars = ax.bar(tab["classe_gravidade"].to_list(), tab["count"].to_list(),
                  color=config.BAR_COLOR)
    for b, p in zip(bars, tab["pct"].to_list()):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{p:.1f}%",
                ha="center", va="bottom")
    ax.set_title("Distribuição das classes de gravidade")
    ax.set_ylabel("Registros")
    data.save_fig(fig, "02_classe_gravidade")
    return tab


def _pct_fatal_por_classe(df: pl.DataFrame) -> pl.DataFrame:
    """% de acidentes fatais em cada classe — valida a separação por severidade."""
    ordem = {c: i for i, c in enumerate(config.CLASSE_ORDEM)}
    tab = (
        df.group_by("classe_gravidade")
        .agg(
            pl.len().alias("n"),
            (pl.col("fatal").mean() * 100).round(2).alias("pct_fatal"),
            pl.col("indice_gravidade").mean().round(2).alias("indice_medio"),
        )
        .with_columns(pl.col("classe_gravidade").replace_strict(ordem, default=99).alias("ord"))
        .sort("ord")
        .drop("ord")
    )
    fig, ax = plt.subplots(figsize=config.FIGSIZE)
    bars = ax.bar(tab["classe_gravidade"].to_list(), tab["pct_fatal"].to_list(),
                  color=config.CLASSE_CORES[::-1])
    for b, v in zip(bars, tab["pct_fatal"].to_list()):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{v:.1f}%",
                ha="center", va="bottom")
    ax.set_title("% de acidentes fatais por classe de gravidade")
    ax.set_ylabel("% fatais na classe")
    data.save_fig(fig, "02_pct_fatal_por_classe")
    return tab


def severity(df: pl.DataFrame) -> dict:
    """Avalia distribuição, classes e capacidade discriminante do índice."""
    desc = df.select("indice_gravidade").describe()
    logger.info("Describe do índice de gravidade:\n%s", desc)
    quantis = df.select(
        pl.col("indice_gravidade").quantile(q).alias(f"p{int(q * 100)}")
        for q in (0.5, 0.75, 0.9, 0.95, 0.99)
    )
    logger.info("Quantis do índice:\n%s", quantis)

    _hist(df)
    _boxplot(df)
    classes = _classes(df)
    logger.info("Distribuição das classes:\n%s", classes)
    fatal_classe = _pct_fatal_por_classe(df)
    logger.info("Índice médio e %% fatal por classe:\n%s", fatal_classe)

    return {"describe": desc, "classes": classes, "fatal_por_classe": fatal_classe}
