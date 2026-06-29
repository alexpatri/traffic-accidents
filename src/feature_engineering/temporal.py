"""Features temporais derivadas de `data` e `horario`.

Justificativa (EDA §temporal): pico de acidentes no fim da tarde (17h–19h),
concentração no fim de semana e sazonalidade mensal (pico em dezembro). Variáveis como
`hora`, `turno`, `mes`, `trimestre` e `fim_de_semana` tornam esses padrões explícitos e
disponíveis para análise, clusterização e modelagem.
"""

from __future__ import annotations

import logging

import polars as pl

from src.feature_engineering import config

logger = logging.getLogger(__name__)


def _expr_turno() -> pl.Expr:
    """Constrói a expressão vetorizada que mapeia `hora` em `turno`."""
    hora = pl.col("hora")
    return (
        pl.when(hora <= config.TURNO_MADRUGADA_MAX)
        .then(pl.lit(config.TURNO_LABELS["madrugada"]))
        .when(hora <= config.TURNO_MANHA_MAX)
        .then(pl.lit(config.TURNO_LABELS["manha"]))
        .when(hora <= config.TURNO_TARDE_MAX)
        .then(pl.lit(config.TURNO_LABELS["tarde"]))
        .otherwise(pl.lit(config.TURNO_LABELS["noite"]))
        .alias("turno")
    )


def adicionar_features_temporais(df: pl.DataFrame) -> pl.DataFrame:
    """Adiciona atributos temporais derivados de `data` e `horario`.

    Cria `hora`, `mes`, `trimestre`, `dia_da_semana` (1=segunda … 7=domingo),
    `fim_de_semana` (bool) e `turno`.

    Args:
        df: DataFrame da camada Trusted.

    Returns:
        DataFrame com as novas colunas temporais.
    """
    df = df.with_columns(
        pl.col("horario").dt.hour().cast(pl.Int8).alias("hora"),
        pl.col("data").dt.month().cast(pl.Int8).alias("mes"),
        pl.col("data").dt.quarter().cast(pl.Int8).alias("trimestre"),
        pl.col("data").dt.weekday().cast(pl.Int8).alias("dia_da_semana"),
    )
    df = df.with_columns(
        pl.col("dia_da_semana")
        .is_in(list(config.FIM_DE_SEMANA_WEEKDAYS))
        .alias("fim_de_semana"),
        _expr_turno(),
    )
    logger.info(
        "Features temporais adicionadas: hora, mes, trimestre, dia_da_semana, "
        "fim_de_semana, turno"
    )
    return df
