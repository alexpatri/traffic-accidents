"""§6 Distribuição espacial: UF, BR e município.

Objetivo: compreender a distribuição espacial (não criar rankings de risco).
"""

from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import polars as pl

from src.eda import config, data

logger = logging.getLogger(__name__)


def _count(df: pl.DataFrame, col: str) -> pl.DataFrame:
    return df.group_by(col).len().sort("len", descending=True)


def _barh(table: pl.DataFrame, col: str, title: str, name: str, top_n: int | None) -> None:
    plot = table.head(top_n) if top_n else table
    labels = plot[col].to_list()[::-1]
    values = plot["len"].to_list()[::-1]
    fig, ax = plt.subplots(figsize=config.FIGSIZE)
    ax.barh(labels, values, color=config.BAR_COLOR)
    ax.set_title(title)
    ax.set_xlabel("Registros")
    data.save_fig(fig, name)


def spatial(df: pl.DataFrame) -> dict[str, pl.DataFrame]:
    """Contagens por UF (todas), BR (top-N) e município (top-N)."""
    por_uf = _count(df, "uf")
    _barh(por_uf, "uf", "Acidentes por UF", "06_por_uf", top_n=None)
    logger.info("Acidentes por UF:\n%s", por_uf)

    por_br = _count(df, "br")
    _barh(por_br, "br", f"Acidentes por BR (top {config.TOP_N})",
          "06_por_br", top_n=config.TOP_N)
    logger.info("Top BRs:\n%s", por_br.head(config.TOP_N))

    por_mun = _count(df, "municipio")
    _barh(por_mun, "municipio", f"Acidentes por município (top {config.TOP_N})",
          "06_por_municipio", top_n=config.TOP_N)
    logger.info("Top municípios:\n%s", por_mun.head(config.TOP_N))

    return {"por_uf": por_uf, "por_br": por_br, "por_municipio": por_mun}
