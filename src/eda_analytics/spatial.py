"""§4 Features espaciais × gravidade (nível acidente).

Usa `uf`, `br`, `trecho` e `km_faixa` para avaliar concentração espacial da gravidade.
Responde: há trechos claramente mais críticos? Algumas BRs concentram acidentes mais
graves? O identificador de trecho mostrou-se adequado? (O ranking detalhado por trecho
fica em §9, sobre o dataset agregado.)
"""

from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import polars as pl

from src.eda_analytics import config, data

logger = logging.getLogger(__name__)


def _barh_indice(tab: pl.DataFrame, col: str, value: str, title: str,
                 xlabel: str, name: str) -> None:
    labels = tab[col].to_list()[::-1]
    values = tab[value].to_list()[::-1]
    fig, ax = plt.subplots(figsize=config.FIGSIZE)
    ax.barh(labels, values, color=config.BAR_COLOR)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    data.save_fig(fig, name)


def spatial(df: pl.DataFrame) -> dict[str, pl.DataFrame]:
    """Gravidade por BR e distribuição de acidentes por trecho."""
    out: dict[str, pl.DataFrame] = {}

    # Distribuição do nº de acidentes por trecho (a partir do nível acidente).
    por_trecho = df.group_by("trecho").len().rename({"len": "qtd_acidentes"})
    out["acidentes_por_trecho"] = por_trecho
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(por_trecho["qtd_acidentes"].to_list(), bins=60, color=config.BAR_COLOR)
    ax.set_yscale("log")
    ax.set_title("Distribuição do nº de acidentes por trecho (eixo y log)")
    ax.set_xlabel("Acidentes no trecho")
    ax.set_ylabel("Nº de trechos (log)")
    data.save_fig(fig, "04_acidentes_por_trecho")
    logger.info(
        "Trechos distintos=%d | acidentes/trecho: média=%.2f máx=%d",
        por_trecho.height,
        por_trecho["qtd_acidentes"].mean(),
        por_trecho["qtd_acidentes"].max(),
    )

    # Índice médio por BR (top-N por volume, para legibilidade).
    por_br = (
        df.group_by("br")
        .agg(
            pl.len().alias("n"),
            pl.col("indice_gravidade").mean().round(2).alias("indice_medio"),
            (pl.col("fatal").mean() * 100).round(2).alias("pct_fatal"),
            pl.col("fatal").sum().alias("qtd_fatais"),
        )
        .filter(pl.col("n") >= 500)  # evita BRs com pouquíssimos registros no ranking
    )
    out["por_br"] = por_br

    top_indice = por_br.sort("indice_medio", descending=True).head(config.TOP_N)
    _barh_indice(top_indice, "br", "indice_medio",
                 f"Índice médio de gravidade por BR (top {config.TOP_N}, n≥500)",
                 "Índice médio", "04_indice_por_br")
    logger.info("BRs por índice médio (top):\n%s", top_indice)

    # Acidentes fatais (contagem absoluta) por BR.
    top_fatais = por_br.sort("qtd_fatais", descending=True).head(config.TOP_N)
    _barh_indice(top_fatais, "br", "qtd_fatais",
                 f"Acidentes fatais por BR (top {config.TOP_N})",
                 "Nº de acidentes fatais", "04_fatais_por_br")
    logger.info("BRs por nº de acidentes fatais (top):\n%s", top_fatais)

    return out
