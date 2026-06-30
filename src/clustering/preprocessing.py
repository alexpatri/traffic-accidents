"""Pré-processamento da matriz de clusterização de trechos (Polars -> NumPy).

`build_matrix_trechos` aplica `log1p` nas features assimétricas (contagens/somas com
cauda longa) e padroniza todas com `StandardScaler`.

Nenhuma linha é descartada (a camada Analytics não tem nulos nas colunas usadas),
de modo que a ordem das linhas de `X` coincide com a de `df` retornado — essencial
para juntar os rótulos de cluster de volta ao DataFrame de contexto.

Obs.: a clusterização de acidentes foi investigada e não incluída como entrega — ver
a documentação (`outputs/clustering/relatorio_clustering.md` e o README).
"""

from __future__ import annotations

import logging
from typing import NamedTuple

import numpy as np
import polars as pl
from sklearn.preprocessing import StandardScaler

from src.clustering import config

logger = logging.getLogger(__name__)


class ClusterMatrix(NamedTuple):
    """Matriz pronta para clusterização e seu contexto.

    Attributes:
        X: matriz (n_linhas × n_features) escalada que o algoritmo enxerga.
        feature_names: nome de cada coluna de `X`, na ordem.
        blocks: mapeamento bloco conceitual -> lista de features (para o relatório).
        df: DataFrame de contexto, alinhado linha a linha com `X` (inclui
            identificador, descritores e desfecho retido para caracterização).
    """

    X: np.ndarray
    feature_names: list[str]
    blocks: dict[str, list[str]]
    df: pl.DataFrame


# --------------------------------------------------------------------------- #
# Clusterização 1 — Trechos
# --------------------------------------------------------------------------- #
def build_matrix_trechos(
    df: pl.DataFrame,
    features: list[str] | None = None,
    log_features: list[str] | None = None,
) -> ClusterMatrix:
    """Constrói a matriz de trechos: log1p nas assimétricas + StandardScaler.

    Args:
        df: dataset por trecho (camada Analytics).
        features: features de formação (default `config.TRECHOS_FEATURES`).
        log_features: subconjunto que recebe log1p (default `config.TRECHOS_LOG_FEATURES`).

    Returns:
        `ClusterMatrix` com `X` padronizada (z-score) e `df` de contexto.
    """
    features = features or config.TRECHOS_FEATURES
    log_features = config.TRECHOS_LOG_FEATURES if log_features is None else log_features
    log_features = [c for c in log_features if c in features]

    exprs = [
        (pl.col(c).log1p() if c in log_features else pl.col(c).cast(pl.Float64)).alias(c)
        for c in features
    ]
    transformado = df.select(exprs)

    scaler = StandardScaler()
    X = scaler.fit_transform(transformado.to_numpy())

    logger.info(
        "Matriz de trechos: %d×%d | log1p em %s | StandardScaler aplicado",
        X.shape[0], X.shape[1], log_features or "(nenhuma)",
    )
    return ClusterMatrix(
        X=X,
        feature_names=list(features),
        blocks={"numericas": list(features)},
        df=df,
    )
