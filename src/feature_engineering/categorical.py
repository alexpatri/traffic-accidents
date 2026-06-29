"""Tratamento de variáveis categóricas: expansão multivalorada e agrupamento de raras.

Justificativa (EDA §categóricas): `tracado_via` é multivalorado (`Reta;Declive`),
respondendo por sua cardinalidade de 898; separá-lo em flags booleanas remove a
multivaloração e preserva cada conceito. `causa_acidente` tem cauda longa (69 categorias),
e agrupar as raras em `Outros` reduz ruído. `municipio` é deliberadamente preservado.
"""

from __future__ import annotations

import logging
import unicodedata

import polars as pl

from src.feature_engineering import config

logger = logging.getLogger(__name__)


def _slug(valor: str) -> str:
    """Converte um rótulo em sufixo de coluna: minúsculo, sem acento, com `_`."""
    sem_acento = (
        unicodedata.normalize("NFKD", valor)
        .encode("ascii", "ignore")
        .decode("ascii")
    )
    return "_".join(sem_acento.lower().split())


def _descobrir_tracados(df: pl.DataFrame) -> list[str]:
    """Descobre, a partir dos dados, os valores únicos contidos em `tracado_via`."""
    valores = (
        df.select(
            pl.col(config.TRACADO_COL)
            .str.split(config.TRACADO_SEP)
            .alias("_tok")
        )
        .explode("_tok")
        .with_columns(pl.col("_tok").str.strip_chars())
        .get_column("_tok")
        .unique()
        .sort()
        .to_list()
    )
    return [v for v in valores if v]


def expandir_tracado_via(df: pl.DataFrame) -> tuple[pl.DataFrame, list[tuple[str, str]]]:
    """Expande `tracado_via` em colunas booleanas `tem_<slug>` por categoria.

    A lista de categorias é construída automaticamente a partir dos dados. A pertinência
    usa correspondência exata por elemento (evita o falso positivo de substring entre
    `Aclive` e `Declive`).

    Args:
        df: DataFrame com a coluna `tracado_via`.

    Returns:
        Tupla com o DataFrame expandido e a lista de pares (categoria, nome_da_coluna).
    """
    categorias = _descobrir_tracados(df)
    lista = pl.col(config.TRACADO_COL).str.split(config.TRACADO_SEP)

    mapeamento: list[tuple[str, str]] = [
        (cat, f"tem_{_slug(cat)}") for cat in categorias
    ]
    exprs = [
        lista.list.contains(cat).alias(coluna) for cat, coluna in mapeamento
    ]
    df = df.with_columns(exprs)
    logger.info(
        "tracado_via expandido em %d flags booleanas: %s",
        len(mapeamento),
        [coluna for _, coluna in mapeamento],
    )
    return df, mapeamento


def agrupar_causas_raras(
    df: pl.DataFrame, threshold: float = config.RARE_THRESHOLD_PCT
) -> tuple[pl.DataFrame, list[str]]:
    """Cria `causa_acidente_agrupada` reunindo causas raras em `Outros`.

    Mantém a coluna original `causa_acidente` intacta. Não altera `municipio`.

    Args:
        df: DataFrame com `causa_acidente`.
        threshold: frequência relativa mínima para uma causa permanecer individual.

    Returns:
        Tupla com o DataFrame e a lista de causas agrupadas em `Outros`.
    """
    total = df.height
    freq = (
        df.group_by(config.CAUSA_COL)
        .len()
        .with_columns((pl.col("len") / total).alias("freq_rel"))
    )
    raras = (
        freq.filter(pl.col("freq_rel") < threshold)
        .get_column(config.CAUSA_COL)
        .to_list()
    )

    df = df.with_columns(
        pl.when(pl.col(config.CAUSA_COL).is_in(raras))
        .then(pl.lit(config.RARE_LABEL))
        .otherwise(pl.col(config.CAUSA_COL))
        .alias("causa_acidente_agrupada")
    )
    logger.info(
        "Causas raras agrupadas em '%s' (limiar=%.2f%%): %d de %d categorias",
        config.RARE_LABEL,
        threshold * 100,
        len(raras),
        freq.height,
    )
    return df, sorted(raras)
