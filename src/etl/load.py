"""Load: persistência da camada Trusted em Parquet."""

from __future__ import annotations

import logging
from pathlib import Path

import polars as pl

from src import config

logger = logging.getLogger(__name__)


def write_trusted(df: pl.DataFrame, path: Path = config.TRUSTED_FILE) -> Path:
    """Grava o DataFrame final em Parquet com compressão configurada.

    Args:
        df: DataFrame já transformado e unificado.
        path: Caminho de destino do Parquet.

    Returns:
        O caminho gravado.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path, compression=config.PARQUET_COMPRESSION)

    size_mb = path.stat().st_size / 1_048_576
    logger.info(
        "Parquet gravado em %s (%d linhas, %.2f MB, compressão=%s)",
        path, df.height, size_mb, config.PARQUET_COMPRESSION,
    )
    return path
