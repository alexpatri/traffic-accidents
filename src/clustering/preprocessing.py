"""Pré-processamento das matrizes de clusterização (Polars -> NumPy).

Duas funções públicas constroem a matriz numérica que o KMeans enxerga:

- `build_matrix_trechos`: 4 features numéricas, log1p nas assimétricas, StandardScaler.
- `build_matrix_acidentes`: dados mistos (cíclica + numérica + one-hot + booleanos),
  com tratamento explícito do problema de escala/peso do KMeans em dados mistos.

Decisões de escala (acidentes):
- Numéricas -> log1p (quando assimétricas) -> MinMax [0, 1].
- `hora` -> sin/cos reescalados para [0, 1] (respeita 23h ≈ 0h).
- One-hot / booleanos -> 0/1 cru (NÃO padronizados: z-score de uma dummy rara
  explodiria seu peso, ex. flag a 0.06% ≈ 40 desvios).
- Peso de bloco: cada coluna é dividida por √(nº de colunas do seu bloco conceitual),
  para que um one-hot de 9 níveis (causa) não supere um único booleano.

Nenhuma linha é descartada (a camada Analytics não tem nulos nas colunas usadas),
de modo que a ordem das linhas de `X` coincide com a de `df` retornado — essencial
para juntar os rótulos de cluster de volta ao DataFrame de contexto.
"""

from __future__ import annotations

import logging
import math
from typing import NamedTuple

import numpy as np
import polars as pl
from sklearn.preprocessing import MinMaxScaler, StandardScaler

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


# --------------------------------------------------------------------------- #
# Clusterização 2 — Acidentes
# --------------------------------------------------------------------------- #
def _assert_sem_vazamento(cols_formacao: list[str]) -> None:
    """Falha cedo se qualquer coluna de desfecho entrar na matriz de formação."""
    vazou = sorted(set(cols_formacao) & set(config.ACIDENTES_LEAKAGE))
    if vazou:
        raise ValueError(
            f"Vazamento de desfecho na clusterização de acidentes: {vazou}. "
            "Essas colunas só podem caracterizar grupos APÓS a clusterização."
        )


def _encode_acidentes(df: pl.DataFrame) -> tuple[pl.DataFrame, dict[str, list[str]]]:
    """Codifica os blocos de features dos acidentes em colunas numéricas.

    Numéricas ficam em escala bruta (MinMax é aplicado depois, em NumPy); one-hot e
    booleanos saem como 0/1. Retorna o DataFrame codificado e o mapa bloco->colunas.
    """
    # --- Temporal: hora cíclica (sin/cos -> [0,1]) + fim de semana ---------- #
    ang = (pl.col(config.ACIDENTES_CICLICA).cast(pl.Float64) / 24.0) * (2 * math.pi)
    temporal = df.select(
        ((ang.sin() + 1) / 2).alias("hora_sin"),
        ((ang.cos() + 1) / 2).alias("hora_cos"),
        *[pl.col(c).cast(pl.Int8).alias(c) for c in config.ACIDENTES_BOOL],
    )

    # --- Operacional: veículos (log1p; MinMax depois) ----------------------- #
    operacional = df.select(
        [pl.col(c).cast(pl.Float64).log1p().alias(c) for c in config.ACIDENTES_NUM_LOG]
    )

    # --- Via: tipo de pista (one-hot) + uso do solo (booleano) -------------- #
    tipo_pista = df.select(pl.col(config.ACIDENTES_TIPO_PISTA)).to_dummies()
    uso = df.select(
        (pl.col(config.ACIDENTES_USO_SOLO) == "Sim").cast(pl.Int8).alias("uso_solo_sim")
    )
    via = pl.concat([tipo_pista, uso], how="horizontal")

    # --- Meteorologia: colapsa para níveis mantidos + "Outros" (one-hot) ---- #
    meteo_col = (
        pl.when(pl.col(config.ACIDENTES_METEO).is_in(config.ACIDENTES_METEO_MANTIDAS))
        .then(pl.col(config.ACIDENTES_METEO))
        .otherwise(pl.lit("Outros"))
        .alias("meteo")
    )
    meteo = df.select(meteo_col).to_dummies()

    # --- Causa: top-N por frequência + "Outros" (one-hot) ------------------- #
    top_causas = (
        df.group_by(config.ACIDENTES_CAUSA)
        .len()
        .sort("len", descending=True)
        .head(config.ACIDENTES_CAUSA_TOP_N)
        .get_column(config.ACIDENTES_CAUSA)
        .to_list()
    )
    causa_col = (
        pl.when(pl.col(config.ACIDENTES_CAUSA).is_in(top_causas))
        .then(pl.col(config.ACIDENTES_CAUSA))
        .otherwise(pl.lit("Outros"))
        .alias("causa")
    )
    causa = df.select(causa_col).to_dummies()

    # --- Traçado: flags mantidas (prevalência ≥ ~3%) ------------------------ #
    tracado = df.select(
        [pl.col(c).cast(pl.Int8).alias(c) for c in config.ACIDENTES_TRACADO_FLAGS]
    )

    blocks: dict[str, list[str]] = {
        "temporal": temporal.columns,
        "operacional": operacional.columns,
        "via": via.columns,
        "meteorologia": meteo.columns,
        "causa": causa.columns,
        "tracado": tracado.columns,
    }
    encoded = pl.concat(
        [temporal, operacional, via, meteo, causa, tracado], how="horizontal"
    )
    return encoded, blocks


def build_matrix_acidentes(df: pl.DataFrame) -> ClusterMatrix:
    """Constrói a matriz de acidentes (dados mistos, sem vazamento de desfecho).

    Args:
        df: camada Analytics no nível acidente.

    Returns:
        `ClusterMatrix` com `X` escalada (MinMax nas numéricas, 0/1 nos one-hot)
        e ponderada por bloco; `df` é o contexto original alinhado às linhas.
    """
    encoded, blocks = _encode_acidentes(df)
    feature_names = encoded.columns
    _assert_sem_vazamento(feature_names)

    X = encoded.to_numpy().astype(np.float64)

    # MinMax [0,1] apenas nas numéricas (log1p(veiculos)); o resto já é 0/1.
    num_idx = [feature_names.index(c) for c in config.ACIDENTES_NUM_LOG]
    if num_idx:
        X[:, num_idx] = MinMaxScaler().fit_transform(X[:, num_idx])

    # Peso de bloco: cada coluna ÷ √(nº de colunas do bloco) -> blocos comparáveis.
    col_block = {c: b for b, cols in blocks.items() for c in cols}
    weights = np.array(
        [1.0 / math.sqrt(len(blocks[col_block[c]])) for c in feature_names]
    )
    X = X * weights

    logger.info(
        "Matriz de acidentes: %d×%d | blocos: %s",
        X.shape[0], X.shape[1],
        {b: len(cols) for b, cols in blocks.items()},
    )
    logger.info("Sanidade anti-vazamento OK (nenhuma feature de desfecho na matriz).")
    return ClusterMatrix(X=X, feature_names=feature_names, blocks=blocks, df=df)
