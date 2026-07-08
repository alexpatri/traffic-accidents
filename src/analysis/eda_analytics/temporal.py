"""§3 Features temporais × gravidade.

Diferente da 1ª EDA (que contou acidentes), aqui o foco é a GRAVIDADE: índice médio e
% fatal por hora, turno, mês e dia da semana, além do contraste útil × fim de semana.
Responde: há períodos mais críticos? Há sazonalidade da gravidade? As features
temporais parecem relevantes para a modelagem?
"""

from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import polars as pl

from src.analysis.eda_analytics import config, data

logger = logging.getLogger(__name__)


def _line_indice(tab: pl.DataFrame, x: str, title: str, xlabel: str, name: str) -> None:
    fig, ax = plt.subplots(figsize=config.FIGSIZE)
    ax.plot(tab[x].to_list(), tab["indice_medio"].to_list(), marker="o",
            color=config.BAR_COLOR)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Índice médio de gravidade")
    ax.grid(True, alpha=0.3)
    data.save_fig(fig, name)


def _ordenar_turno(tab: pl.DataFrame) -> pl.DataFrame:
    ordem = {t: i for i, t in enumerate(config.TURNO_ORDEM)}
    return (
        tab.with_columns(pl.col("turno").replace_strict(ordem, default=99).alias("ord"))
        .sort("ord")
        .drop("ord")
    )


def temporal(df: pl.DataFrame) -> dict[str, pl.DataFrame]:
    """Gravidade por hora, turno, mês, dia da semana e fim de semana."""
    out: dict[str, pl.DataFrame] = {}

    por_hora = data.severity_by(df, "hora")
    out["por_hora"] = por_hora
    _line_indice(por_hora, "hora", "Índice médio de gravidade por hora do dia",
                 "Hora", "03_indice_por_hora")
    logger.info("Gravidade por hora:\n%s", por_hora)

    # Turno: índice médio e % fatal lado a lado.
    por_turno = _ordenar_turno(data.severity_by(df, "turno"))
    out["por_turno"] = por_turno
    fig, ax = plt.subplots(figsize=config.FIGSIZE)
    x = por_turno["turno"].to_list()
    ax.bar(x, por_turno["indice_medio"].to_list(), color=config.BAR_COLOR,
           label="Índice médio")
    ax.set_ylabel("Índice médio de gravidade")
    ax2 = ax.twinx()
    ax2.plot(x, por_turno["pct_fatal"].to_list(), marker="o", color="#b2182b",
             label="% fatal")
    ax2.set_ylabel("% de acidentes fatais")
    ax.set_title("Gravidade por turno (índice médio vs. % fatal)")
    ax.legend(loc="upper left", fontsize=8)
    ax2.legend(loc="upper right", fontsize=8)
    data.save_fig(fig, "03_gravidade_por_turno")
    logger.info("Gravidade por turno:\n%s", por_turno)

    por_mes = data.severity_by(df, "mes")
    out["por_mes"] = por_mes
    _line_indice(por_mes, "mes", "Índice médio de gravidade por mês",
                 "Mês", "03_indice_por_mes")
    logger.info("Gravidade por mês:\n%s", por_mes)

    por_dia = data.severity_by(df, "dia_da_semana")
    out["por_dia_semana"] = por_dia
    _line_indice(por_dia, "dia_da_semana",
                 "Índice médio de gravidade por dia da semana (1=seg … 7=dom)",
                 "Dia da semana", "03_indice_por_dia_semana")
    logger.info("Gravidade por dia da semana:\n%s", por_dia)

    # Útil × fim de semana: índice médio e % fatal.
    fds = (
        df.group_by("fim_de_semana")
        .agg(
            pl.len().alias("n"),
            pl.col("indice_gravidade").mean().round(2).alias("indice_medio"),
            (pl.col("fatal").mean() * 100).round(2).alias("pct_fatal"),
        )
        .sort("fim_de_semana")
        .with_columns(
            pl.when(pl.col("fim_de_semana")).then(pl.lit("Fim de semana"))
            .otherwise(pl.lit("Dia útil")).alias("rotulo")
        )
    )
    out["fim_de_semana"] = fds
    fig, ax = plt.subplots(figsize=(7, 5))
    rot = fds["rotulo"].to_list()
    bars = ax.bar(rot, fds["indice_medio"].to_list(), color=config.BAR_COLOR)
    for b, p in zip(bars, fds["pct_fatal"].to_list()):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                f"{p:.1f}% fatais", ha="center", va="bottom")
    ax.set_title("Gravidade: dia útil vs. fim de semana")
    ax.set_ylabel("Índice médio de gravidade")
    data.save_fig(fig, "03_util_vs_fds")
    logger.info("Útil vs. fim de semana:\n%s", fds)

    return out
