"""§1 Visão geral da camada Analytics.

Responde: quantas features foram criadas? Como a Analytics difere da Trusted?
Apresenta linhas/colunas, schema, memória e o contraste de largura entre as camadas.
"""

from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import polars as pl

from src.analysis.eda_analytics import config, data

logger = logging.getLogger(__name__)


def overview(df: pl.DataFrame) -> dict:
    """Panorama estrutural da Analytics + comparação de largura com a Trusted."""
    info = {
        "linhas": df.height,
        "colunas": df.width,
        "memoria_mb": round(df.estimated_size("mb"), 2),
        "n_novas_features": len(config.NOVAS_FEATURES),
        "schema": {c: str(dt) for c, dt in df.schema.items()},
    }
    logger.info(
        "Analytics: shape=%(linhas)dx%(colunas)d | memória=%(memoria_mb)s MB | "
        "novas features=%(n_novas_features)d",
        info,
    )
    logger.info("Schema completo: %s", info["schema"])

    # Confere quais features esperadas estão de fato presentes.
    presentes = [c for c in config.NOVAS_FEATURES if c in df.columns]
    ausentes = [c for c in config.NOVAS_FEATURES if c not in df.columns]
    logger.info("Features novas presentes (%d): %s", len(presentes), presentes)
    if ausentes:
        logger.warning("Features esperadas ausentes: %s", ausentes)

    # Largura da Trusted derivada por subtração (sem ler a camada Trusted): as colunas
    # da Analytics que não são features novas correspondem às herdadas da Trusted.
    n_trusted = df.width - len(presentes)
    info["colunas_trusted"] = n_trusted

    fig, ax = plt.subplots(figsize=(7, 5))
    camadas = ["Trusted", "Analytics"]
    larguras = [n_trusted, df.width]
    cores = ["#9ecae1", config.BAR_COLOR]
    bars = ax.bar(camadas, larguras, color=cores)
    for b, v in zip(bars, larguras):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v}", ha="center", va="bottom")
    ax.set_title("Largura do dataset: Trusted vs. Analytics")
    ax.set_ylabel("Nº de colunas")
    ax.text(
        1, df.width / 2, f"+{df.width - n_trusted}\nfeatures",
        ha="center", va="center", color="white", fontsize=11, fontweight="bold",
    )
    data.save_fig(fig, "01_colunas_trusted_vs_analytics")

    return info
