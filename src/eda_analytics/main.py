"""Orquestração da EDA Analytics: valida as features da camada Analytics.

Carrega os dois Parquet da camada Analytics e executa todos os módulos de análise,
gerando as figuras em outputs/eda_analytics/figures/ e registrando as estatísticas no log.

Execução:
    python -m src.eda_analytics.main
"""

from __future__ import annotations

import logging

from src.eda_analytics import (
    categorical,
    correlation,
    data,
    overview,
    relationships,
    report,
    severity,
    spatial,
    temporal,
    trechos,
)

logger = logging.getLogger("eda_analytics")


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        handlers=[logging.StreamHandler()],
    )


def run() -> None:
    """Executa a EDA Analytics completa na ordem das seções do relatório."""
    _setup_logging()
    logger.info("=== Início da EDA Analytics — Acidentes PRF (camada Analytics) ===")

    df = data.load_analytics()
    trechos_df = data.load_trechos()

    overview.overview(df)              # §1 visão geral
    severity.severity(df)              # §2 índice de gravidade
    temporal.temporal(df)             # §3 features temporais
    spatial.spatial(df)               # §4 features espaciais
    categorical.categorical(df)       # §5 traçado + §6 causa
    relationships.relationships(df)   # §7 relações entre features
    correlation.correlations(df)      # §8 correlação numérica
    trechos.trechos(trechos_df)       # §9 dataset por trecho
    report.report(df)                 # §10 clusterização + §11 ML

    logger.info("=== EDA Analytics concluída — figuras em outputs/eda_analytics/figures/ ===")


if __name__ == "__main__":
    run()
