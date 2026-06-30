"""Orquestração da etapa de Clusterização.

Executa a clusterização de trechos rodoviários e consolida o relatório. A clusterização
de acidentes foi investigada e não incluída como entrega (ver relatório/README): o KMeans
não formou perfis multivariados nítidos, apenas redescobre `tipo_pista`.

Execução:
    python -m src.clustering.main
"""

from __future__ import annotations

import logging

from src.clustering import report, trechos

logger = logging.getLogger("clustering")


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        handlers=[logging.StreamHandler()],
    )


def run() -> None:
    """Executa a clusterização de trechos e gera o relatório consolidado."""
    _setup_logging()
    logger.info("=== Início da Clusterização — Acidentes PRF ===")

    trechos_result = trechos.run()
    report.report(trechos_result)

    logger.info("=== Clusterização concluída — artefatos em outputs/clustering/ ===")


if __name__ == "__main__":
    run()
