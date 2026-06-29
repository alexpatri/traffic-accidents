"""Identificador de trecho rodoviário.

Justificativa (EDA §espacial + objetivo do projeto): identificar os trechos mais
perigosos exige uma chave espacial estável. Como `km` é contínuo e o número da BR se
repete entre estados, o trecho é definido como `UF_BR_<faixa de km>` (faixa de
`KM_BIN_SIZE` km), base para agregação, ranking e priorização de investimentos.
"""

from __future__ import annotations

import logging

import polars as pl

from src.feature_engineering import config

logger = logging.getLogger(__name__)


def adicionar_trecho(df: pl.DataFrame) -> pl.DataFrame:
    """Adiciona `km_faixa` (km discretizado) e `trecho` (UF_BR_km_faixa).

    Args:
        df: DataFrame com `uf`, `br` e `km`.

    Returns:
        DataFrame com `km_faixa` (Int32) e `trecho` (String).
    """
    km_faixa = (
        (pl.col("km") / config.KM_BIN_SIZE).floor() * config.KM_BIN_SIZE
    ).cast(pl.Int32)
    df = df.with_columns(km_faixa.alias("km_faixa"))
    df = df.with_columns(
        pl.concat_str(
            [pl.col("uf"), pl.col("br"), pl.col("km_faixa")],
            separator="_",
        ).alias("trecho")
    )
    logger.info(
        "Trecho criado (faixa=%d km): %d trechos distintos",
        config.KM_BIN_SIZE,
        df["trecho"].n_unique(),
    )
    return df
