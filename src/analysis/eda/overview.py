"""§1 Visão geral + §2 Qualidade dos dados.

Responde: o dataset está balanceado entre 2024 e 2025? Há nulos remanescentes?
Quais variáveis têm alta/baixa cardinalidade?
"""

from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import polars as pl

from src.analysis.eda import config, data

logger = logging.getLogger(__name__)


def overview(df: pl.DataFrame) -> dict:
    """Panorama estrutural: shape, schema, memória e distribuição por ano."""
    info = {
        "linhas": df.height,
        "colunas": df.width,
        "memoria_mb": round(df.estimated_size("mb"), 2),
        "schema": {c: str(dt) for c, dt in df.schema.items()},
    }
    logger.info("Shape=%(linhas)dx%(colunas)d | memória=%(memoria_mb)s MB", info)
    logger.info("Schema: %s", info["schema"])

    por_ano = df["ano_arquivo"].value_counts().sort("ano_arquivo")
    info["por_ano"] = por_ano
    logger.info("Distribuição por ano:\n%s", por_ano)

    fig, ax = plt.subplots(figsize=(7, 5))
    anos = por_ano["ano_arquivo"].to_list()
    counts = por_ano["count"].to_list()
    ax.bar([str(a) for a in anos], counts, color=config.BAR_COLOR)
    for i, c in enumerate(counts):
        ax.text(i, c, f"{c:,}", ha="center", va="bottom")
    ax.set_title("Acidentes por ano")
    ax.set_xlabel("Ano")
    ax.set_ylabel("Registros")
    data.save_fig(fig, "01_acidentes_por_ano")
    return info


def quality(df: pl.DataFrame) -> dict:
    """Qualidade: nulos, % nulos, frequência de 'Não informado' e cardinalidade."""
    n = df.height

    nulls = df.null_count()
    null_cols = {c: nulls[c][0] for c in df.columns if nulls[c][0] > 0}
    logger.info("Colunas com nulos: %s", null_cols or "nenhuma")

    cats = config.LOW_CARD_CATS + config.HIGH_CARD_CATS
    nao_inf = {
        c: df.filter(pl.col(c) == "Não informado").height
        for c in cats
        if df.filter(pl.col(c) == "Não informado").height > 0
    }
    logger.info("Frequência de 'Não informado': %s", nao_inf or "nenhuma")

    card = (
        pl.DataFrame(
            {
                "coluna": cats,
                "cardinalidade": [df[c].n_unique() for c in cats],
            }
        )
        .with_columns((pl.col("cardinalidade") / n * 100).round(3).alias("pct_distintos"))
        .sort("cardinalidade", descending=True)
    )
    logger.info("Cardinalidade das categóricas:\n%s", card)

    fig, ax = plt.subplots(figsize=config.FIGSIZE)
    ax.barh(card["coluna"].to_list()[::-1], card["cardinalidade"].to_list()[::-1],
            color=config.BAR_COLOR)
    ax.set_xscale("log")
    ax.set_title("Cardinalidade das variáveis categóricas (escala log)")
    ax.set_xlabel("Nº de valores distintos")
    data.save_fig(fig, "02_cardinalidade_categoricas")

    return {"null_cols": null_cols, "nao_informado": nao_inf, "cardinalidade": card}
