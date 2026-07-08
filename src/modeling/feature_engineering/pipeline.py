"""Pipeline da Feature Engineering: encadeia as transformações por acidente.

Produz o DataFrame analítico por acidente e os metadados (decisões e categorias
descobertas) consumidos pelo relatório.
"""

from __future__ import annotations

import logging
from typing import Any

import polars as pl

from src.modeling.feature_engineering import categorical, config, severity, spatial, temporal

logger = logging.getLogger(__name__)


def construir_features(df: pl.DataFrame) -> tuple[pl.DataFrame, dict[str, Any]]:
    """Aplica todas as features de domínio à camada Trusted.

    Ordem: temporais → gravidade → trecho → expansão de `tracado_via` →
    agrupamento de causas raras.

    Args:
        df: DataFrame da camada Trusted.

    Returns:
        Tupla com o DataFrame enriquecido (por acidente) e um dicionário de metadados
        (pesos, limiares, categorias de `tracado_via`, causas agrupadas).
    """
    colunas_iniciais = df.width

    df = temporal.adicionar_features_temporais(df)
    df = severity.adicionar_indice_gravidade(df)
    df = severity.adicionar_flags_gravidade(df)
    df = spatial.adicionar_trecho(df)
    df, tracados = categorical.expandir_tracado_via(df)
    df, causas_raras = categorical.agrupar_causas_raras(df)

    meta: dict[str, Any] = {
        "pesos_indice": {
            "mortos": config.PESO_MORTOS,
            "feridos_graves": config.PESO_FERIDOS_GRAVES,
            "feridos_leves": config.PESO_FERIDOS_LEVES,
        },
        "km_bin_size": config.KM_BIN_SIZE,
        "rare_threshold_pct": config.RARE_THRESHOLD_PCT,
        "tracados": tracados,
        "causas_raras": causas_raras,
        "n_trechos": df["trecho"].n_unique(),
    }

    logger.info(
        "Feature engineering concluída: %d → %d colunas (%d novas), %d linhas",
        colunas_iniciais,
        df.width,
        df.width - colunas_iniciais,
        df.height,
    )
    return df, meta
