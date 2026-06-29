"""Helpers de I/O e plotagem reutilizados pelos módulos da EDA Analytics."""

from __future__ import annotations

import logging

import matplotlib

matplotlib.use("Agg")  # backend não-interativo: apenas salva figuras em arquivo

import matplotlib.pyplot as plt
import polars as pl

from src.eda_analytics import config

logger = logging.getLogger(__name__)


def load_analytics() -> pl.DataFrame:
    """Carrega a camada Analytics (nível acidente)."""
    if not config.ANALYTICS_FILE.exists():
        raise FileNotFoundError(
            f"Camada Analytics não encontrada: {config.ANALYTICS_FILE}. "
            "Execute a Feature Engineering (python -m src.feature_engineering.main) antes da EDA."
        )
    df = pl.read_parquet(config.ANALYTICS_FILE)
    logger.info("Analytics carregado: %d linhas, %d colunas", df.height, df.width)
    return df


def load_trechos() -> pl.DataFrame:
    """Carrega o dataset agregado por trecho."""
    if not config.TRECHOS_FILE.exists():
        raise FileNotFoundError(
            f"Dataset de trechos não encontrado: {config.TRECHOS_FILE}. "
            "Execute a Feature Engineering (python -m src.feature_engineering.main) antes da EDA."
        )
    df = pl.read_parquet(config.TRECHOS_FILE)
    logger.info("Trechos carregado: %d linhas, %d colunas", df.height, df.width)
    return df


def severity_by(df: pl.DataFrame, col: str) -> pl.DataFrame:
    """Agrega gravidade por categoria: frequência, índice médio e % fatal.

    Reutilizado por temporal/spatial/categorical/relationships — produz uma tabela com
    `n` (registros), `indice_medio` (média de `indice_gravidade`) e `pct_fatal`
    (percentual de acidentes com `fatal=True`) por valor de `col`.
    """
    return (
        df.group_by(col)
        .agg(
            pl.len().alias("n"),
            pl.col("indice_gravidade").mean().round(2).alias("indice_medio"),
            (pl.col("fatal").mean() * 100).round(2).alias("pct_fatal"),
        )
        .sort(col)
    )


def save_fig(fig: plt.Figure, name: str) -> None:
    """Salva a figura em `FIG_DIR/<name>.png` e a fecha para liberar memória."""
    config.FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = config.FIG_DIR / f"{name}.png"
    fig.savefig(path, dpi=config.DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Figura salva: %s", path.relative_to(config.PROJECT_ROOT))
