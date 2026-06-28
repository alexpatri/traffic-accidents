"""Helpers de I/O e plotagem reutilizados pelos módulos da EDA."""

from __future__ import annotations

import logging

import matplotlib

matplotlib.use("Agg")  # backend não-interativo: apenas salva figuras em arquivo

import matplotlib.pyplot as plt
import polars as pl

from src.eda import config

logger = logging.getLogger(__name__)


def load_trusted() -> pl.DataFrame:
    """Carrega a camada Trusted (única fonte permitida nesta etapa)."""
    if not config.TRUSTED_FILE.exists():
        raise FileNotFoundError(
            f"Camada Trusted não encontrada: {config.TRUSTED_FILE}. "
            "Execute o ETL (python -m src.main) antes da EDA."
        )
    df = pl.read_parquet(config.TRUSTED_FILE)
    logger.info("Trusted carregado: %d linhas, %d colunas", df.height, df.width)
    return df


def save_fig(fig: plt.Figure, name: str) -> None:
    """Salva a figura em `FIG_DIR/<name>.png` e a fecha para liberar memória."""
    config.FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = config.FIG_DIR / f"{name}.png"
    fig.savefig(path, dpi=config.DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Figura salva: %s", path.relative_to(config.PROJECT_ROOT))
