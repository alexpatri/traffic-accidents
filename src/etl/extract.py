"""Extract: leitura dos CSVs brutos da PRF.

Os dados são lidos integralmente como texto (`infer_schema_length=0`). A tipagem é
feita explicitamente em `transform`, evitando que o Polars infira tipos diferentes
entre os anos (ex.: latitude com separador decimal distinto) e dando controle total
sobre a conversão.
"""

from __future__ import annotations

import logging
from pathlib import Path

import polars as pl

from src import config

logger = logging.getLogger(__name__)


def read_raw(path: Path, year: int) -> pl.DataFrame:
    """Lê um CSV bruto como texto e marca sua origem.

    Args:
        path: Caminho do arquivo CSV.
        year: Ano que o arquivo representa (vira a coluna de rastreio).

    Returns:
        DataFrame com todas as colunas como `Utf8` e a coluna de origem.
    """
    if not path.exists():
        raise FileNotFoundError(f"Arquivo de origem não encontrado: {path}")

    df = pl.read_csv(
        path,
        separator=config.SEPARATOR,
        encoding=config.ENCODING,
        infer_schema_length=0,  # tudo como string; tipagem é feita em transform
        null_values=config.NULL_TOKENS,
    )
    df = df.with_columns(pl.lit(year).cast(pl.Int32).alias(config.SOURCE_COL))

    logger.info("Lido %s: %d linhas, %d colunas", path.name, df.height, df.width)
    return df


def extract_all() -> dict[int, pl.DataFrame]:
    """Lê todos os datasets configurados em `RAW_FILES`.

    Returns:
        Dicionário ano -> DataFrame cru.
    """
    return {year: read_raw(path, year) for year, path in config.RAW_FILES.items()}
