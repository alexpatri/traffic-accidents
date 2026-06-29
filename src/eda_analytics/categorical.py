"""§5 Traçado da via + §6 Causa do acidente agrupada.

§5 avalia as flags booleanas `tem_*` (frequência, índice médio e % fatal de cada
característica geométrica). §6 avalia `causa_acidente_agrupada` (frequência, índice
médio e % fatal), verificando se o agrupamento reduziu cardinalidade sem perder poder
de discriminação. Responde: alguma geometria/causa concentra maior gravidade?
"""

from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import polars as pl

from src.eda_analytics import config, data

logger = logging.getLogger(__name__)


def _tracado(df: pl.DataFrame) -> pl.DataFrame:
    """Para cada flag de traçado: frequência, índice médio e % fatal quando presente."""
    linhas = []
    for flag in config.TRACADO_FLAGS:
        sub = df.filter(pl.col(flag))
        if sub.height == 0:
            continue
        linhas.append(
            {
                "tracado": flag.replace("tem_", ""),
                "frequencia": sub.height,
                "pct_registros": round(sub.height / df.height * 100, 2),
                "indice_medio": round(sub["indice_gravidade"].mean(), 2),
                "pct_fatal": round(sub["fatal"].mean() * 100, 2),
            }
        )
    tab = pl.DataFrame(linhas).sort("indice_medio", descending=True)

    # Gráfico: índice médio por característica geométrica, anotando frequência.
    fig, ax = plt.subplots(figsize=config.FIGSIZE)
    plot = tab.sort("indice_medio")
    labels = plot["tracado"].to_list()
    bars = ax.barh(labels, plot["indice_medio"].to_list(), color=config.BAR_COLOR)
    for b, n in zip(bars, plot["frequencia"].to_list()):
        ax.text(b.get_width(), b.get_y() + b.get_height() / 2, f" n={n:,}",
                va="center", fontsize=7)
    ax.set_title("Índice médio de gravidade por característica do traçado")
    ax.set_xlabel("Índice médio de gravidade")
    data.save_fig(fig, "05_indice_por_tracado")

    # Gráfico: % fatal por característica geométrica.
    fig, ax = plt.subplots(figsize=config.FIGSIZE)
    plot = tab.sort("pct_fatal")
    ax.barh(plot["tracado"].to_list(), plot["pct_fatal"].to_list(), color="#b2182b")
    ax.set_title("% de acidentes fatais por característica do traçado")
    ax.set_xlabel("% fatais")
    data.save_fig(fig, "05_pct_fatal_por_tracado")
    return tab


def _causa(df: pl.DataFrame) -> pl.DataFrame:
    """Causa agrupada: frequência, índice médio e % fatal (top-N por índice médio)."""
    tab = (
        df.group_by("causa_acidente_agrupada")
        .agg(
            pl.len().alias("frequencia"),
            pl.col("indice_gravidade").mean().round(2).alias("indice_medio"),
            (pl.col("fatal").mean() * 100).round(2).alias("pct_fatal"),
        )
        .sort("frequencia", descending=True)
    )
    logger.info(
        "causa_acidente_agrupada: %d categorias (orig. causa_acidente=%d)",
        tab.height, df["causa_acidente"].n_unique(),
    )

    # Frequência (top-N).
    freq = tab.head(config.TOP_N).sort("frequencia")
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(freq["causa_acidente_agrupada"].to_list(), freq["frequencia"].to_list(),
            color=config.BAR_COLOR)
    ax.set_title(f"Frequência por causa agrupada (top {config.TOP_N})")
    ax.set_xlabel("Registros")
    data.save_fig(fig, "06_freq_causa")

    # Índice médio (top-N mais graves, com volume mínimo para não destacar ruído).
    graves = (
        tab.filter(pl.col("frequencia") >= 200)
        .sort("indice_medio", descending=True)
        .head(config.TOP_N)
        .sort("indice_medio")
    )
    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh(graves["causa_acidente_agrupada"].to_list(),
                   graves["indice_medio"].to_list(), color="#b2182b")
    for b, p in zip(bars, graves["pct_fatal"].to_list()):
        ax.text(b.get_width(), b.get_y() + b.get_height() / 2, f" {p:.1f}% fatal",
                va="center", fontsize=7)
    ax.set_title(f"Índice médio de gravidade por causa agrupada (top {config.TOP_N}, n≥200)")
    ax.set_xlabel("Índice médio de gravidade")
    data.save_fig(fig, "06_indice_causa")
    return tab


def categorical(df: pl.DataFrame) -> dict[str, pl.DataFrame]:
    """Executa as análises de traçado (§5) e causa agrupada (§6)."""
    tracado = _tracado(df)
    logger.info("Traçado × gravidade:\n%s", tracado)
    causa = _causa(df)
    logger.info("Causa agrupada (top por índice médio):\n%s",
                causa.sort("indice_medio", descending=True).head(10))
    return {"tracado": tracado, "causa": causa}
