"""Orquestração da Feature Engineering: Trusted → camada Analytics.

Execução: ``python -m src.feature_engineering.main``.
Gera ``acidentes_analytics.parquet``, ``trechos_analytics.parquet`` e o relatório.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import polars as pl

from src.feature_engineering import aggregation, config, pipeline, report

logger = logging.getLogger("feature_engineering")


def _setup_logging() -> None:
    """Configura logging simultâneo em arquivo (logs/) e stdout."""
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = config.LOG_DIR / f"feature_engineering_{timestamp}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )


def _load_trusted() -> pl.DataFrame:
    """Carrega a camada Trusted (fonte da Feature Engineering)."""
    if not config.TRUSTED_FILE.exists():
        raise FileNotFoundError(
            f"Camada Trusted não encontrada: {config.TRUSTED_FILE}. Rode o ETL antes."
        )
    df = pl.read_parquet(config.TRUSTED_FILE)
    logger.info("Trusted carregado: %d linhas, %d colunas", df.height, df.width)
    return df


def _write_parquet(df: pl.DataFrame, path: Path) -> Path:
    """Grava um DataFrame em Parquet com a compressão configurada."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path, compression=config.PARQUET_COMPRESSION)
    size_mb = path.stat().st_size / 1_048_576
    logger.info(
        "Parquet gravado em %s (%d linhas, %d colunas, %.2f MB)",
        path.relative_to(config.PROJECT_ROOT),
        df.height,
        df.width,
        size_mb,
    )
    return path


def run() -> None:
    """Executa o pipeline completo de Feature Engineering."""
    _setup_logging()
    logger.info("=== Início da Feature Engineering — Acidentes PRF ===")

    df = _load_trusted()

    df_acidentes, meta = pipeline.construir_features(df)
    _write_parquet(df_acidentes, config.ACIDENTES_ANALYTICS_FILE)

    df_trechos = aggregation.agregar_por_trecho(df_acidentes)
    _write_parquet(df_trechos, config.TRECHOS_ANALYTICS_FILE)

    report.gerar_relatorio(meta, df_acidentes, df_trechos)

    logger.info("=== Feature Engineering concluída — camada Analytics gerada ===")


if __name__ == "__main__":
    run()
