"""Orquestração da etapa de Clusterização.

Executa as duas análises independentes (trechos e acidentes) e consolida o relatório.

Execução:
    python -m src.clustering.main
"""

from __future__ import annotations

import logging

from src.clustering import acidentes, report, trechos

logger = logging.getLogger("clustering")


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        handlers=[logging.StreamHandler()],
    )


def run() -> None:
    """Executa a clusterização completa: trechos + acidentes + relatório."""
    _setup_logging()
    logger.info("=== Início da Clusterização — Acidentes PRF ===")

    trechos_result = trechos.run()
    acidentes_result = acidentes.run()
    report.report(trechos_result, acidentes_result)

    logger.info("=== Clusterização concluída — artefatos em outputs/clustering/ ===")


if __name__ == "__main__":
    run()
