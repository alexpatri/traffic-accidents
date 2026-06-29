"""Índice e indicadores de gravidade do acidente.

Justificativa (EDA §correlações/§relações): as contagens de vítimas têm baixa correlação
independente entre si, de modo que um índice ponderado agrega mais informação que cada
coluna isolada; a classe fatal é minoritária (~7%), o que torna úteis tanto a flag `fatal`
quanto uma `classe_gravidade` ordinal. `periodo_noturno` captura a maior letalidade
observada em fases de baixa luminosidade.
"""

from __future__ import annotations

import logging

import polars as pl

from src.feature_engineering import config

logger = logging.getLogger(__name__)


def adicionar_indice_gravidade(df: pl.DataFrame) -> pl.DataFrame:
    """Adiciona `indice_gravidade` como combinação ponderada das vítimas.

    indice_gravidade = PESO_MORTOS·mortos + PESO_FERIDOS_GRAVES·feridos_graves
                       + PESO_FERIDOS_LEVES·feridos_leves.

    Args:
        df: DataFrame com as contagens de vítimas.

    Returns:
        DataFrame com a coluna `indice_gravidade` (Int32).
    """
    indice = (
        config.PESO_MORTOS * pl.col("mortos")
        + config.PESO_FERIDOS_GRAVES * pl.col("feridos_graves")
        + config.PESO_FERIDOS_LEVES * pl.col("feridos_leves")
    ).cast(pl.Int32)
    return df.with_columns(indice.alias("indice_gravidade"))


def _expr_classe_gravidade() -> pl.Expr:
    """Mapeia `indice_gravidade` em uma classe ordinal de severidade."""
    indice = pl.col("indice_gravidade")
    return (
        pl.when(indice <= config.CLASSE_BAIXA_MAX)
        .then(pl.lit(config.CLASSE_LABELS["baixa"]))
        .when(indice <= config.CLASSE_MEDIA_MAX)
        .then(pl.lit(config.CLASSE_LABELS["media"]))
        .when(indice <= config.CLASSE_ALTA_MAX)
        .then(pl.lit(config.CLASSE_LABELS["alta"]))
        .otherwise(pl.lit(config.CLASSE_LABELS["critica"]))
        .alias("classe_gravidade")
    )


def adicionar_flags_gravidade(df: pl.DataFrame) -> pl.DataFrame:
    """Adiciona `fatal`, `classe_gravidade` e `periodo_noturno`.

    Requer que `indice_gravidade` já exista no DataFrame.

    Args:
        df: DataFrame já enriquecido com `indice_gravidade`.

    Returns:
        DataFrame com as flags e a classe de gravidade.
    """
    df = df.with_columns(
        (pl.col("mortos") > 0).alias("fatal"),
        pl.col("fase_dia").is_in(list(config.FASE_DIA_NOTURNA)).alias("periodo_noturno"),
        _expr_classe_gravidade(),
    )
    logger.info(
        "Features de gravidade adicionadas: indice_gravidade, fatal, "
        "classe_gravidade, periodo_noturno"
    )
    return df
