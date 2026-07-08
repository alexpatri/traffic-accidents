"""§9 Avaliação do dataset agregado por trecho (trechos_analytics.parquet).

Responde: o risco se concentra em poucos trechos? O ranking parece coerente? Produz
histogramas, boxplots, curva de concentração (Pareto) e o ranking dos trechos críticos.
"""

from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import polars as pl

from src.analysis.eda_analytics import config, data

logger = logging.getLogger(__name__)


def _hist(serie: list[float], titulo: str, xlabel: str, name: str, logy: bool = True) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(serie, bins=60, color=config.BAR_COLOR)
    if logy:
        ax.set_yscale("log")
    ax.set_title(titulo)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Nº de trechos" + (" (log)" if logy else ""))
    data.save_fig(fig, name)


def _pareto(trechos: pl.DataFrame) -> float:
    """Curva de concentração: % acumulado do índice total vs. % de trechos."""
    ordenado = trechos.sort("indice_gravidade_total", descending=True)
    total = ordenado["indice_gravidade_total"].sum()
    cum = ordenado["indice_gravidade_total"].cum_sum() / total * 100
    n = ordenado.height
    pct_trechos = [(i + 1) / n * 100 for i in range(n)]
    cum_list = cum.to_list()

    # % do risco concentrado nos 10% piores trechos.
    idx10 = max(0, int(n * 0.10) - 1)
    risco_top10 = cum_list[idx10]

    fig, ax = plt.subplots(figsize=config.FIGSIZE)
    ax.plot(pct_trechos, cum_list, color=config.BAR_COLOR)
    ax.axvline(10, color="#b2182b", linestyle="--", alpha=0.7)
    ax.axhline(risco_top10, color="#b2182b", linestyle="--", alpha=0.7)
    ax.set_title("Concentração do risco (índice de gravidade total acumulado)")
    ax.set_xlabel("% dos trechos (ordenados do pior ao melhor)")
    ax.set_ylabel("% acumulado do índice de gravidade total")
    ax.grid(True, alpha=0.3)
    ax.text(12, risco_top10 - 6, f"10% dos trechos = {risco_top10:.1f}% do risco",
            color="#b2182b", fontsize=9)
    data.save_fig(fig, "09_concentracao_risco")
    return risco_top10


def _ranking(trechos: pl.DataFrame) -> pl.DataFrame:
    """Top-N trechos por índice de gravidade total (gráfico de barras horizontais)."""
    top = trechos.sort("indice_gravidade_total", descending=True).head(config.TOP_N)
    plot = top.sort("indice_gravidade_total")
    fig, ax = plt.subplots(figsize=(10, 8))
    bars = ax.barh(plot["trecho"].to_list(), plot["indice_gravidade_total"].to_list(),
                   color="#b2182b")
    for b, m, n in zip(bars, plot["mortos"].to_list(), plot["qtd_acidentes"].to_list()):
        ax.text(b.get_width(), b.get_y() + b.get_height() / 2,
                f" {m} mortos / {n} acid.", va="center", fontsize=7)
    ax.set_title(f"Top {config.TOP_N} trechos por índice de gravidade total")
    ax.set_xlabel("Índice de gravidade total")
    data.save_fig(fig, "09_top_trechos")
    return top


def trechos(df: pl.DataFrame) -> dict:
    """Distribuições, concentração e ranking sobre o dataset por trecho."""
    desc = df.select(config.TRECHO_METRICAS).describe()
    logger.info("Describe das métricas por trecho:\n%s", desc)

    _hist(df["qtd_acidentes"].to_list(), "Distribuição do nº de acidentes por trecho",
          "Acidentes no trecho", "09_hist_qtd_acidentes")
    _hist(df["indice_gravidade_total"].to_list(),
          "Distribuição do índice de gravidade total por trecho",
          "Índice de gravidade total", "09_hist_indice_total")
    _hist(df["indice_gravidade_medio"].to_list(),
          "Distribuição do índice de gravidade médio por trecho",
          "Índice de gravidade médio", "09_hist_indice_medio")

    # Boxplot da taxa de acidentes fatais (apenas trechos com >1 acidente: a taxa é
    # informativa quando há denominador; trechos com 1 acidente são 0% ou 100%).
    fig, ax = plt.subplots(figsize=(6, 5))
    multi = df.filter(pl.col("qtd_acidentes") > 1)
    ax.boxplot(multi["pct_acidentes_fatais"].to_list(), labels=["pct_fatais"])
    ax.set_title("Taxa de acidentes fatais por trecho (trechos com >1 acidente)")
    ax.set_ylabel("% de acidentes fatais")
    data.save_fig(fig, "09_boxplot_pct_fatais")

    risco_top10 = _pareto(df)
    logger.info("Concentração: 10%% dos trechos acumulam %.1f%% do índice total",
                risco_top10)

    ranking = _ranking(df)
    logger.info("Top trechos por índice total:\n%s", ranking)

    # Estatísticas de concentração para o relatório.
    zero_fatal = df.filter(pl.col("mortos") == 0).height
    logger.info(
        "Trechos: total=%d | com 1 acidente=%d (%.1f%%) | sem mortos=%d (%.1f%%)",
        df.height,
        df.filter(pl.col("qtd_acidentes") == 1).height,
        df.filter(pl.col("qtd_acidentes") == 1).height / df.height * 100,
        zero_fatal, zero_fatal / df.height * 100,
    )

    return {"describe": desc, "ranking": ranking, "risco_top10": risco_top10}
