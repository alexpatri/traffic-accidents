"""§5 Distribuição temporal: ano, mês, dia da semana e hora.

As colunas derivadas (mês, hora) são temporárias, criadas apenas para a análise e
NÃO persistidas. Responde: há padrões sazonais? Horários críticos? Concentração em
finais de semana?
"""

from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import polars as pl

from src.analysis.eda import config, data

logger = logging.getLogger(__name__)

# Ordem natural dos dias para o eixo do gráfico.
_DIAS_ORDEM = [
    "segunda-feira", "terça-feira", "quarta-feira", "quinta-feira",
    "sexta-feira", "sábado", "domingo",
]


def _line(x, y, title, xlabel, name, rotate=False) -> None:
    fig, ax = plt.subplots(figsize=config.FIGSIZE)
    ax.plot(x, y, marker="o", color=config.BAR_COLOR)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Registros")
    ax.grid(True, alpha=0.3)
    if rotate:
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    data.save_fig(fig, name)


def temporal(df: pl.DataFrame) -> dict[str, pl.DataFrame]:
    """Séries temporais de contagem de acidentes em várias granularidades."""
    # Colunas temporárias para análise (não persistidas).
    tmp = df.with_columns(
        pl.col("data").dt.month().alias("mes"),
        pl.col("horario").dt.hour().alias("hora"),
    )

    por_mes = tmp.group_by("mes").len().sort("mes")
    _line(por_mes["mes"].to_list(), por_mes["len"].to_list(),
          "Acidentes por mês", "Mês", "05_por_mes")
    logger.info("Acidentes por mês:\n%s", por_mes)

    por_hora = tmp.group_by("hora").len().sort("hora")
    _line(por_hora["hora"].to_list(), por_hora["len"].to_list(),
          "Acidentes por hora do dia", "Hora", "05_por_hora")
    logger.info("Acidentes por hora:\n%s", por_hora)

    por_dia = (
        tmp.group_by("dia_semana").len()
        .with_columns(
            pl.col("dia_semana").replace_strict(
                {d: i for i, d in enumerate(_DIAS_ORDEM)}, default=99
            ).alias("ord")
        )
        .sort("ord")
    )
    fig, ax = plt.subplots(figsize=config.FIGSIZE)
    ax.bar(por_dia["dia_semana"].to_list(), por_dia["len"].to_list(),
           color=config.BAR_COLOR)
    ax.set_title("Acidentes por dia da semana")
    ax.set_ylabel("Registros")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    data.save_fig(fig, "05_por_dia_semana")
    logger.info("Acidentes por dia da semana:\n%s", por_dia.drop("ord"))

    return {"por_mes": por_mes, "por_hora": por_hora, "por_dia": por_dia.drop("ord")}
