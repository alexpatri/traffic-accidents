"""Orquestração do ETL: Extract -> Transform -> Load.

Execução:
    python -m src.main
"""

from __future__ import annotations

import logging
from datetime import datetime

from src import config, extract, load, transform

logger = logging.getLogger("etl")


def _setup_logging() -> None:
    """Configura logging simultâneo em arquivo (logs/) e stdout."""
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = config.LOG_DIR / f"etl_{timestamp}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    logger.info("Log desta execução: %s", log_file)


def run() -> None:
    """Executa o pipeline completo de ETL."""
    _setup_logging()
    logger.info("=== Início do ETL — Acidentes PRF (Trusted) ===")

    # Extract
    frames = extract.extract_all()

    # Transform
    df, reports = transform.transform(frames)
    logger.info("Relatório de validação por ano: %s", reports)

    # Load
    destino = load.write_trusted(df)

    # Resumo final
    logger.info("Schema final: %s", dict(df.schema))
    logger.info("=== ETL concluído com sucesso -> %s ===", destino)


if __name__ == "__main__":
    run()
