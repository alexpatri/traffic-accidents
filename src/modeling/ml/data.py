"""Helpers de I/O e plotagem da etapa de ML (espelha `src/modeling/clustering/data.py`)."""

from __future__ import annotations

import logging

import matplotlib

matplotlib.use("Agg")  # backend não-interativo: apenas salva figuras em arquivo

import matplotlib.pyplot as plt
import polars as pl

from src.modeling.ml import config

logger = logging.getLogger(__name__)


def load_acidentes() -> pl.DataFrame:
    """Carrega a camada Analytics no nível acidente."""
    if not config.ACIDENTES_FILE.exists():
        raise FileNotFoundError(
            f"Camada Analytics não encontrada: {config.ACIDENTES_FILE}. "
            "Execute a Feature Engineering (python -m src.modeling.feature_engineering.main) antes."
        )
    df = pl.read_parquet(config.ACIDENTES_FILE)
    logger.info("Acidentes carregado: %d linhas, %d colunas", df.height, df.width)
    return df


def load_trechos() -> pl.DataFrame:
    """Carrega o dataset agregado por trecho (fonte do alvo e do filtro de volume)."""
    if not config.TRECHOS_FILE.exists():
        raise FileNotFoundError(
            f"Dataset de trechos não encontrado: {config.TRECHOS_FILE}. "
            "Execute a Feature Engineering (python -m src.modeling.feature_engineering.main) antes."
        )
    df = pl.read_parquet(config.TRECHOS_FILE)
    logger.info("Trechos carregado: %d linhas, %d colunas", df.height, df.width)
    return df


def save_fig(fig: plt.Figure, name: str) -> None:
    """Salva a figura em `FIG_DIR/<name>.png` e a fecha para liberar memória."""
    config.FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = config.FIG_DIR / f"{name}.png"
    fig.savefig(path, dpi=config.DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Figura salva: %s", path.relative_to(config.PROJECT_ROOT))
