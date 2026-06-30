"""Seleção do número de clusters: Método do Cotovelo (inertia) + Silhouette.

Para cada K em `config.K_RANGE` ajusta KMeans e calcula:
- `inertia`: soma dos erros quadráticos intra-cluster (SSE) — base do cotovelo;
- `silhouette`: coesão/separação média — base da escolha fina.

O silhouette é O(n²) em memória/tempo; mesmo nos 33k trechos é avaliado numa amostra
fixa (mesmos índices em todos os K, para que a curva seja comparável), enquanto o KMeans
é ajustado na base inteira.
"""

from __future__ import annotations

import logging
from typing import NamedTuple

import numpy as np
from sklearn.metrics import silhouette_score

from src.clustering import cluster, config

logger = logging.getLogger(__name__)


class SelectionResult(NamedTuple):
    """Curvas de seleção de K.

    Attributes:
        ks: valores de K avaliados.
        inertias: SSE por K (cotovelo).
        silhouettes: Silhouette médio por K.
        sample_size: nº de pontos usados no silhouette (n se base completa).
    """

    ks: list[int]
    inertias: list[float]
    silhouettes: list[float]
    sample_size: int


def evaluate_k(
    X: np.ndarray,
    k_range: range = config.K_RANGE,
    silhouette_sample_size: int | None = None,
) -> SelectionResult:
    """Calcula inertia e silhouette para uma faixa de K.

    Args:
        X: matriz de features escalada.
        k_range: valores de K a testar.
        silhouette_sample_size: se informado e menor que n, avalia o silhouette numa
            amostra fixa desse tamanho; caso contrário usa a base completa.

    Returns:
        `SelectionResult` com as curvas alinhadas a `ks`.
    """
    n = X.shape[0]
    if silhouette_sample_size and silhouette_sample_size < n:
        rng = np.random.default_rng(config.RANDOM_STATE)
        sample_idx = rng.choice(n, size=silhouette_sample_size, replace=False)
        sample_size = silhouette_sample_size
    else:
        sample_idx = None
        sample_size = n

    ks, inertias, silhouettes = [], [], []
    for k in k_range:
        labels, model = cluster.fit_cluster(X, algo="kmeans", k=k)
        if sample_idx is not None:
            sil = silhouette_score(X[sample_idx], labels[sample_idx])
        else:
            sil = silhouette_score(X, labels)
        ks.append(k)
        inertias.append(float(model.inertia_))
        silhouettes.append(float(sil))
        logger.info("K=%d | inertia=%.1f | silhouette=%.4f", k, model.inertia_, sil)

    return SelectionResult(ks, inertias, silhouettes, sample_size)


def best_k_by_silhouette(result: SelectionResult) -> int:
    """Retorna o K de maior Silhouette (referência; a escolha final é documentada)."""
    return result.ks[int(np.argmax(result.silhouettes))]
