"""Orquestração da EDA: carrega a Trusted e executa todos os módulos de análise.

Gera as figuras em outputs/eda/figures/ e registra as estatísticas no log.

Execução:
    python -m src.analysis.eda.main
"""

from __future__ import annotations

import logging

from src.analysis.eda import (
    categorical,
    correlations,
    data,
    numerical,
    overview,
    relationships,
    spatial,
    temporal,
)

logger = logging.getLogger("eda")


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        handlers=[logging.StreamHandler()],
    )


def run() -> None:
    """Executa a EDA completa na ordem das seções do relatório."""
    _setup_logging()
    logger.info("=== Início da EDA — Acidentes PRF (camada Trusted) ===")

    df = data.load_trusted()

    overview.overview(df)       # §1 visão geral
    overview.quality(df)        # §2 qualidade
    categorical.categorical(df) # §3 categóricas
    numerical.numerical(df)     # §4 numéricas
    temporal.temporal(df)       # §5 temporal
    spatial.spatial(df)         # §6 espacial
    relationships.relationships(df)  # §7 relações
    correlations.correlations(df)    # §8 correlações

    logger.info("=== EDA concluída — figuras em outputs/eda/figures/ ===")


if __name__ == "__main__":
    run()
