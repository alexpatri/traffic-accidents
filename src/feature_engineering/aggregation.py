"""Agregação por trecho rodoviário (camada Analytics — uma linha por trecho).

Justificativa (objetivo do projeto): consolidar os acidentes por `trecho` produz as
métricas que sustentam o ranking dos segmentos mais críticos e a priorização de
investimentos em infraestrutura.
"""

from __future__ import annotations

import logging

import polars as pl

from src.feature_engineering import config

logger = logging.getLogger(__name__)


def agregar_por_trecho(df: pl.DataFrame) -> pl.DataFrame:
    """Agrega o dataset de acidentes em uma linha por trecho.

    Requer as features `trecho`, `km_faixa`, `indice_gravidade` e `fatal`.

    Args:
        df: DataFrame de acidentes já enriquecido.

    Returns:
        DataFrame com uma linha por trecho, ordenado por `indice_gravidade_total` desc.
    """
    trechos = (
        df.group_by(["trecho", "uf", "br", "km_faixa"])
        .agg(
            pl.len().alias("qtd_acidentes"),
            pl.col("mortos").sum().alias("mortos"),
            pl.col("feridos_graves").sum().alias("feridos_graves"),
            pl.col("feridos_leves").sum().alias("feridos_leves"),
            pl.col("indice_gravidade").mean().round(2).alias("indice_gravidade_medio"),
            pl.col("indice_gravidade").max().alias("indice_gravidade_maximo"),
            pl.col("indice_gravidade").sum().alias("indice_gravidade_total"),
            pl.col("fatal").sum().alias("qtd_acidentes_fatais"),
            (pl.col("fatal").mean() * 100).round(2).alias("pct_acidentes_fatais"),
        )
        .sort("indice_gravidade_total", descending=True)
    )
    logger.info(
        "Agregação por trecho concluída: %d trechos, %d colunas",
        trechos.height,
        trechos.width,
    )
    return trechos
